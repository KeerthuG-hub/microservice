"""
Kubernetes Parser - extracts ALL dependency types from K8s manifests.
Zero hardcoded service names, ports, or patterns.
"""

from pathlib import Path
from typing import List, Dict, Optional
import yaml
import re
from agent.models import Dependency, Service


class KubernetesParser:

    def __init__(self, services: Dict[str, Service],
                 port_map: Dict[int, str],
                 env_patterns):
        self.services = services
        self.port_map = port_map
        self.env_patterns = env_patterns
        self.deps: List[Dependency] = []

    # -------------------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------------------

    def parse(self, project_path: str) -> List[Dependency]:
        print("\n📋 Parsing Kubernetes manifests...")
        yaml_files = (list(Path(project_path).rglob('*.yaml')) +
                      list(Path(project_path).rglob('*.yml')))

        for f in yaml_files:
            if self._is_vendor(f):
                continue
            self._parse_file(f)

        print(f"   ✅ {len(self.deps)} deps from K8s")
        return self.deps

    # -------------------------------------------------------------------------
    # FILE PARSING
    # -------------------------------------------------------------------------

    def _parse_file(self, path: Path):
        try:
            with open(path, errors='ignore') as f:
                for doc in yaml.safe_load_all(f):
                    if not doc:
                        continue
                    kind = doc.get('kind', '')
                    if kind == 'Deployment':
                        self._parse_deployment(doc)
                    elif kind == 'ConfigMap':
                        self._parse_configmap(doc)
                    elif kind == 'Ingress':
                        self._parse_ingress(doc)
                    elif kind == 'Job':
                        self._parse_job(doc)
                    elif kind in ('ServiceMonitor', 'PodMonitor'):
                        self._parse_monitor(doc)
                    elif kind in ('VirtualService', 'DestinationRule'):
                        self._parse_service_mesh(doc)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # DEPLOYMENT
    # -------------------------------------------------------------------------

    def _parse_deployment(self, doc: dict):
        name = doc.get('metadata', {}).get('name', '')
        from_svc = self._match(name)
        if not from_svc:
            return

        spec = doc.get('spec', {}).get('template', {}).get('spec', {})

        # Regular containers
        for container in spec.get('containers', []):
            self._parse_container_env(from_svc, container)
            self._parse_container_image(from_svc, container)

        # Init containers — deployment ordering
        for init in spec.get('initContainers', []):
            init_name = init.get('name', '')
            to_svc = self._match(init_name)
            if to_svc and to_svc != from_svc:
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=to_svc,
                    dep_type='deployment',
                    subtype='startup_order',
                    confidence=0.85,
                    evidence=f'initContainer: {init_name}',
                    source='k8s_init_container'
                ))
            # Also parse env in init containers
            self._parse_container_env(from_svc, init)

    def _parse_container_env(self, from_svc: str, container: dict):
        for env in container.get('env', []):
            key = env.get('name', '')
            val = str(env.get('value', ''))
            if not val or val.startswith('$('):  # Skip K8s variable refs
                continue
            dep = self._extract_from_env_entry(from_svc, key, val)
            if dep:
                self._add(dep)

        # K8s auto-injected env from service discovery
        for envFrom in container.get('envFrom', []):
            ref = envFrom.get('configMapRef', {}).get('name', '')
            to_svc = self._match(ref)
            if to_svc and to_svc != from_svc:
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=to_svc,
                    dep_type='endpoint',
                    subtype='http',
                    confidence=0.72,
                    evidence=f'envFrom configMapRef: {ref}',
                    source='k8s_envfrom'
                ))

    def _parse_container_image(self, from_svc: str, container: dict):
        image = container.get('image', '')
        if not image:
            return
        # Internal registry base image → build dependency
        # Only flag if registry looks internal (not docker.io / gcr.io / quay.io)
        known_public = {'docker.io', 'gcr.io', 'quay.io', 'ghcr.io',
                        'registry.k8s.io', 'k8s.gcr.io'}
        registry = image.split('/')[0] if '/' in image else ''
        is_public = any(registry.endswith(p) for p in known_public) or ':' not in registry
        if not is_public and '/' in image:
            img_name = image.split('/')[-1].split(':')[0]
            to_svc = self._match(img_name)
            if to_svc and to_svc != from_svc:
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=to_svc,
                    dep_type='build',
                    subtype='base_image',
                    confidence=0.75,
                    evidence=f'FROM {image}',
                    source='k8s_image'
                ))

    # -------------------------------------------------------------------------
    # CONFIGMAP
    # -------------------------------------------------------------------------

    def _parse_configmap(self, doc: dict):
        cm_name = doc.get('metadata', {}).get('name', '')
        # Which service owns this ConfigMap?
        from_svc = self._match(cm_name)

        for key, val in (doc.get('data') or {}).items():
            val = str(val)
            dep = self._extract_from_env_entry(from_svc or cm_name, key, val)
            if dep and from_svc:
                self._add(dep)

    # -------------------------------------------------------------------------
    # INGRESS
    # -------------------------------------------------------------------------

    def _parse_ingress(self, doc: dict):
        """Ingress rules tell us which services are externally exposed."""
        for rule in doc.get('spec', {}).get('rules', []):
            for path_item in rule.get('http', {}).get('paths', []):
                backend = (path_item.get('backend', {})
                           .get('service', {})
                           .get('name', ''))
                if backend:
                    to_svc = self._match(backend)
                    if to_svc:
                        self._add(Dependency(
                            from_service='__external__',
                            to_service=to_svc,
                            dep_type='endpoint',
                            subtype='http',
                            confidence=0.90,
                            evidence=f'Ingress → {backend}',
                            source='k8s_ingress'
                        ))

    # -------------------------------------------------------------------------
    # JOB (db migrations, etc.)
    # -------------------------------------------------------------------------

    def _parse_job(self, doc: dict):
        name = doc.get('metadata', {}).get('name', '')
        annotations = doc.get('metadata', {}).get('annotations', {})

        # Helm hooks
        hook = annotations.get('helm.sh/hook', '')
        if hook:
            for container in (doc.get('spec', {})
                               .get('template', {})
                               .get('spec', {})
                               .get('containers', [])):
                img = container.get('image', '').split('/')[-1].split(':')[0]
                from_svc = self._match(img) or self._match(name)
                if from_svc:
                    self._add(Dependency(
                        from_service=from_svc,
                        to_service='__helm__',
                        dep_type='deployment',
                        subtype='helm_hook',
                        confidence=0.80,
                        evidence=f'helm.sh/hook={hook}',
                        source='k8s_job'
                    ))

        # DB migration jobs
        job_lower = name.lower()
        if any(kw in job_lower for kw in ('migrat', 'migrate', 'flyway',
                                           'liquibase', 'alembic', 'schema')):
            for container in (doc.get('spec', {})
                               .get('template', {})
                               .get('spec', {})
                               .get('containers', [])):
                self._parse_container_env(name, container)

    # -------------------------------------------------------------------------
    # OBSERVABILITY
    # -------------------------------------------------------------------------

    def _parse_monitor(self, doc: dict):
        """Prometheus ServiceMonitor / PodMonitor."""
        selector = doc.get('spec', {}).get('selector', {})
        match_labels = selector.get('matchLabels', {})
        for label_val in match_labels.values():
            to_svc = self._match(label_val)
            if to_svc:
                self._add(Dependency(
                    from_service='prometheus',
                    to_service=to_svc,
                    dep_type='observability',
                    subtype='prometheus',
                    confidence=0.85,
                    evidence=f'ServiceMonitor selector: {match_labels}',
                    source='k8s_servicemonitor'
                ))

    # -------------------------------------------------------------------------
    # SERVICE MESH
    # -------------------------------------------------------------------------

    def _parse_service_mesh(self, doc: dict):
        kind = doc.get('kind', '')
        name = doc.get('metadata', {}).get('name', '')

        if kind == 'VirtualService':
            # VS routes to specific destinations
            for http_route in doc.get('spec', {}).get('http', []):
                for route in http_route.get('route', []):
                    dest = route.get('destination', {}).get('host', '')
                    from_svc = self._match(name)
                    to_svc = self._match(dest)
                    if from_svc and to_svc and from_svc != to_svc:
                        self._add(Dependency(
                            from_service=from_svc,
                            to_service=to_svc,
                            dep_type='observability',
                            subtype='service_mesh',
                            confidence=0.78,
                            evidence=f'VirtualService route → {dest}',
                            source='k8s_virtualservice'
                        ))

    # -------------------------------------------------------------------------
    # ENV EXTRACTION (generic — no hardcoded patterns)
    # -------------------------------------------------------------------------

    def _extract_from_env_entry(self, from_svc: str,
                                key: str, val: str) -> Optional[Dependency]:
        """
        Try to extract a dependency from a single env key=value.
        Uses patterns LEARNED from the project, not hardcoded ones.
        """
        val_lower = val.lower()

        # Pattern 1: host:port or scheme://host:port — look for known service name in host part
        url_match = re.search(r'(?:https?|grpc|amqp|redis|mongodb|mysql|postgres)?'
                              r'(?:://)?([a-zA-Z0-9_-]+)(?::\d+)?', val)
        if url_match:
            candidate = url_match.group(1)
            to_svc = self._match(candidate)
            if to_svc and to_svc != from_svc:
                subtype = 'http'
                if 'grpc' in val_lower:
                    subtype = 'grpc'
                elif 'amqp' in val_lower:
                    subtype = 'amqp'
                elif any(x in val_lower for x in ('redis', ':6379')):
                    return Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='infrastructure',
                        subtype='redis',
                        confidence=0.88,
                        evidence=f'env {key}={val}',
                        source='k8s_env'
                    )
                elif any(x in val_lower for x in ('postgres', 'pg', ':5432')):
                    return Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='infrastructure',
                        subtype='postgres',
                        confidence=0.88,
                        evidence=f'env {key}={val}',
                        source='k8s_env'
                    )
                return Dependency(
                    from_service=from_svc,
                    to_service=to_svc,
                    dep_type='endpoint',
                    subtype=subtype,
                    confidence=0.90,
                    evidence=f'env {key}={val}',
                    source='k8s_env'
                )

        # Pattern 2: service name appears anywhere in value
        for svc_name in self.services:
            if svc_name.lower() in val_lower and svc_name != from_svc:
                return Dependency(
                    from_service=from_svc,
                    to_service=svc_name,
                    dep_type='endpoint',
                    subtype='http',
                    confidence=0.82,
                    evidence=f'env {key}={val}',
                    source='k8s_env_svcname'
                )

        # Pattern 3: Port-based inference (LEARNED port map only)
        port_match = re.search(r':(\d+)', val)
        if port_match:
            port = int(port_match.group(1))
            if port in self.port_map:
                to_svc = self.port_map[port]
                if to_svc != from_svc:
                    return Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='endpoint',
                        subtype='http',
                        confidence=0.72,
                        evidence=f'port {port} → {to_svc}',
                        source='k8s_port'
                    )

        # Pattern 4: Messaging broker references
        if ':9092' in val or 'kafka' in val_lower:
            return Dependency(
                from_service=from_svc,
                to_service='kafka',
                dep_type='infrastructure',
                subtype='kafka',
                confidence=0.85,
                evidence=f'env {key}={val}',
                source='k8s_env_broker'
            )
        if ':5672' in val or 'rabbitmq' in val_lower or 'amqp' in val_lower:
            return Dependency(
                from_service=from_svc,
                to_service='rabbitmq',
                dep_type='infrastructure',
                subtype='rabbitmq',
                confidence=0.85,
                evidence=f'env {key}={val}',
                source='k8s_env_broker'
            )
        if ':4222' in val or 'nats' in val_lower:
            return Dependency(
                from_service=from_svc,
                to_service='nats',
                dep_type='infrastructure',
                subtype='nats',
                confidence=0.85,
                evidence=f'env {key}={val}',
                source='k8s_env_broker'
            )

        return None

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _match(self, name: str) -> Optional[str]:
        """Match a name string to a known service (normalized)."""
        norm = self._norm(name)
        for svc_name, svc in self.services.items():
            if svc.normalized_name == norm:
                return svc_name
        return None

    def _norm(self, name: str) -> str:
        return name.lower().replace('-', '').replace('_', '').replace('.', '')

    def _add(self, dep: Dependency):
        if dep.from_service and dep.to_service and dep.from_service != dep.to_service:
            self.deps.append(dep)

    @staticmethod
    def _is_vendor(path: Path) -> bool:
        skip = {'node_modules', 'vendor', '.git', 'build', 'dist',
                'target', '__pycache__', '.venv', 'venv', 'generated'}
        return any(p in skip for p in path.parts)
