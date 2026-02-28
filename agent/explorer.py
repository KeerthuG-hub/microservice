"""
Service Explorer - discovers services with ZERO hardcoded names.
Works for ANY domain. Learns conventions FROM the project.
"""

from pathlib import Path
from typing import Dict, Set, List
import yaml
import re
from agent.models import Service, ProjectContext


class ServiceExplorer:
    """Discover services from project - NO assumptions about names or structure"""

    def __init__(self, context: ProjectContext):
        self.context = context
        self.project_path = Path(context.project_root)
        self.services: Dict[str, Service] = {}

        # LEARNED from project (NOT hardcoded)
        self.port_map: Dict[int, str] = {}
        self.env_patterns: Set[str] = set()

    # -------------------------------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------------------------------

    def discover(self) -> Dict[str, Service]:
        print("\n🔍 Discovering services...")

        # Priority order (higher = more authoritative)
        self._discover_from_k8s()
        self._match_src_directories()   
        self._discover_from_anchors()
        if self.context.has_docker_compose:
            self._discover_from_compose()
        if self.context.has_grpc:
            self._discover_from_proto()

        # Assign languages to services discovered without one
        self._infer_languages()

        # Classify infrastructure vs real services (purely structural — zero hardcoding)
        k8s_deployments = self._gather_k8s_deployment_names()
        self._classify_infrastructure(k8s_deployments)

        # Learn project conventions
        self._learn_port_map()
        self._learn_env_patterns()

        infra = [n for n, s in self.services.items() if s.is_infrastructure]
        real  = [n for n, s in self.services.items() if not s.is_infrastructure]
        print(f"   ✅ Found {len(self.services)} nodes: "
              f"{len(real)} services + {len(infra)} infrastructure")
        for name, svc in self.services.items():
            tag = " [INFRA]" if svc.is_infrastructure else ""
            print(f"      • {name}  [{svc.language}]  anchor={svc.anchor}{tag}")

        return self.services


    def _match_src_directories(self):
        for svc_name, svc in self.services.items():
            for subdir in ['src', 'services', 'apps']:
                candidate = self.project_path / subdir / svc_name
                if candidate.exists():
                    svc.path = str(candidate)
                    break

    # -------------------------------------------------------------------------
    # DISCOVERY STRATEGIES
    # -------------------------------------------------------------------------

    def _discover_from_k8s(self):
        """K8s Service objects are the most authoritative source."""
        for yaml_file in self._yaml_files():
            try:
                with open(yaml_file, errors='ignore') as f:
                    for doc in yaml.safe_load_all(f):
                        if not doc or doc.get('kind') != 'Service':
                            continue
                        name = doc.get('metadata', {}).get('name', '')
                        if not name:
                            continue

                        ports = []
                        for ps in doc.get('spec', {}).get('ports', []):
                            if 'port' in ps:
                                ports.append(ps['port'])

                        self._register(Service(
                            name=name,
                            path=str(yaml_file.parent),
                            language='unknown',
                            anchor=f'k8s:Service:{yaml_file.name}',
                            ports=ports,
                            normalized_name=self._norm(name)
                        ))
            except Exception:
                pass

    def _discover_from_anchors(self):
        """Anchor files prove a directory is a microservice."""
        anchor_lang = {
            'Dockerfile': 'docker',
            'go.mod': 'go',
            'package.json': 'javascript',
            'pom.xml': 'java',
            'build.gradle': 'java',
            'build.gradle.kts': 'kotlin',
            'requirements.txt': 'python',
            'Pipfile': 'python',
            'pyproject.toml': 'python',
            'Cargo.toml': 'rust',
            'Gemfile': 'ruby',
            'composer.json': 'php',
            'Program.cs': 'csharp',
            'main.go': 'go',
            'main.py': 'python',
            'index.js': 'javascript',
            'app.py': 'python',
            'server.go': 'go',
            'server.js': 'javascript',
            'server.ts': 'typescript',
            'app.js': 'javascript',
            'app.ts': 'typescript',
        }

        for anchor, lang in anchor_lang.items():
            for anchor_path in self.project_path.rglob(anchor):
                if self._is_vendor(anchor_path):
                    continue

                svc_dir = anchor_path.parent
                # Skip project root itself
                if svc_dir == self.project_path:
                    continue

                svc_name = svc_dir.name
                # Skip shared/lib dirs — not real services
                if svc_name.lower() in {
                    'shared', 'lib', 'libs', 'common', 'util', 'utils',
                    'helpers', 'internal', 'pkg',
                    # Filesystem layout dirs — NOT services
                    'src', 'app', 'apps', 'cmd', 'api',
                    'proto', 'protos', 'pb', 'gen', 'generated',
                    'test', 'tests', 'testdata', 'mocks', 'mock',
                    'scripts', 'hack', 'deploy', 'docs', 'examples',
                }:
                    continue

                norm = self._norm(svc_name)
                if not self._norm_exists(norm):
                    self._register(Service(
                        name=svc_name,
                        path=str(svc_dir),
                        language=lang,
                        anchor=f'file:{anchor}',
                        normalized_name=norm
                    ))

    def _discover_from_compose(self):
        """docker-compose services section."""
        for compose_file in self._compose_files():
            try:
                with open(compose_file) as f:
                    data = yaml.safe_load(f)
                if 'services' not in data:
                    continue

                for svc_name, svc_cfg in data['services'].items():
                    norm = self._norm(svc_name)
                    if self._norm_exists(norm):
                        continue

                    ports = []
                    for p in (svc_cfg or {}).get('ports', []):
                        try:
                            host_port = str(p).split(':')[0]
                            ports.append(int(host_port))
                        except Exception:
                            pass

                    self._register(Service(
                        name=svc_name,
                        path=str(compose_file.parent),
                        language='unknown',
                        anchor=f'compose:{compose_file.name}',
                        ports=ports,
                        normalized_name=norm
                    ))
            except Exception:
                pass

    def _discover_from_proto(self):
        """gRPC service definitions in .proto files."""
        for proto_file in self.project_path.rglob('*.proto'):
            if self._is_vendor(proto_file):
                continue
            try:
                content = proto_file.read_text(errors='ignore')
                for svc_name in re.findall(r'service\s+(\w+)\s*\{', content):
                    norm = self._norm(svc_name)
                    if not self._norm_exists(norm):
                        self._register(Service(
                            name=svc_name,
                            path=str(proto_file.parent),
                            language='proto',
                            anchor=f'proto:{proto_file.name}',
                            normalized_name=norm
                        ))
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # INFRASTRUCTURE CLASSIFICATION (zero hardcoding)
    # -------------------------------------------------------------------------

    def _gather_k8s_deployment_names(self) -> set:
        """
        Scan all K8s manifests in the project and return the set of
        normalized names of every Deployment and StatefulSet found.
        These are used as 'has an owned workload' signals.
        """
        workload_kinds = {'Deployment', 'StatefulSet', 'DaemonSet'}
        names: set = set()
        for yaml_file in self._yaml_files():
            try:
                with open(yaml_file, errors='ignore') as f:
                    for doc in yaml.safe_load_all(f):
                        if not doc or doc.get('kind') not in workload_kinds:
                            continue
                        name = doc.get('metadata', {}).get('name', '')
                        if name:
                            names.add(self._norm(name))
                        # Also capture selector app label values
                        selector = (doc.get('spec', {})
                                      .get('selector', {})
                                      .get('matchLabels', {}))
                        for v in selector.values():
                            names.add(self._norm(str(v)))
            except Exception:
                pass
        return names

    def _classify_infrastructure(self, k8s_deployments: set):
        """
        Flag services as infrastructure if ALL three structural conditions hold:
          1. No owned source code found (language stayed 'unknown' after infer_languages)
          2. No matching K8s Deployment / StatefulSet / DaemonSet in the project
          3. Anchor is only from k8s:Service or compose — never a real code file anchor

        All signals come from the project's own files — zero hardcoded names/ports.
        """
        for svc in self.services.values():
            # Condition 1: no owned code
            no_code = svc.language in ('unknown', 'docker')

            # Condition 2: no Deployment-class workload found for this name
            no_deployment = svc.normalized_name not in k8s_deployments

            # Condition 3: anchor is not from a code file — only from k8s or compose
            code_anchors = {'file:', 'proto:', 'go.mod', 'package.json',
                            'pom.xml', 'requirements.txt', 'Cargo.toml'}
            anchor_is_infra_only = not any(
                svc.anchor.startswith(a) for a in code_anchors
            )

            if no_code and no_deployment and anchor_is_infra_only:
                svc.is_infrastructure = True

    # -------------------------------------------------------------------------
    # LEARNING CONVENTIONS FROM THE PROJECT
    # -------------------------------------------------------------------------

    def _learn_port_map(self):
        """Build port → service mapping FROM discovered K8s Service objects."""
        for svc in self.services.values():
            for port in svc.ports:
                self.port_map[port] = svc.name
        print(f"   📍 Learned {len(self.port_map)} port mappings")

    def _learn_env_patterns(self):
        """
        Learn env var naming patterns FROM this project.
        Example: if env var ORDER_SERVICE_URL value = 'http://order-service:8080'
        then the pattern learned is '_SERVICE_URL' (suffix after service name)
        """
        for yaml_file in self._yaml_files():
            try:
                with open(yaml_file, errors='ignore') as f:
                    for doc in yaml.safe_load_all(f):
                        if not doc or doc.get('kind') != 'Deployment':
                            continue
                        containers = (doc.get('spec', {})
                                        .get('template', {})
                                        .get('spec', {})
                                        .get('containers', []))
                        for container in containers:
                            for env_entry in container.get('env', []):
                                key = env_entry.get('name', '')
                                val = str(env_entry.get('value', ''))
                                for svc_name in self.services:
                                    if svc_name.lower() in val.lower():
                                        suffix = re.sub(
                                            re.escape(svc_name.upper().replace('-', '_')),
                                            '', key.upper()
                                        )
                                        if suffix:
                                            self.env_patterns.add(suffix)
            except Exception:
                pass
        if self.env_patterns:
            print(f"   📝 Learned env patterns: {self.env_patterns}")

    def _infer_languages(self):
        """For services without a detected language, scan their directory."""
        ext_lang = {
            '.go': 'go', '.py': 'python', '.java': 'java',
            '.js': 'javascript', '.ts': 'typescript', '.rs': 'rust',
            '.rb': 'ruby', '.php': 'php', '.cs': 'csharp',
        }
        for svc in self.services.values():
            if svc.language not in ('unknown', 'docker', 'proto'):
                continue
            svc_path = Path(svc.path)
            counts: Dict[str, int] = {}
            for ext, lang in ext_lang.items():
                hits = [
                    p for p in svc_path.rglob(f'*{ext}')
                    if not self._is_vendor(p)
                ]
                n = len(hits)

                if n:
                    counts[lang] = n

            if counts:
                svc.language = max(counts, key=counts.get)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _register(self, service: Service):
        """Register if not already present (first discoverer wins for name)."""
        if service.name not in self.services:
            self.services[service.name] = service

    def _norm_exists(self, norm: str) -> bool:
        return any(s.normalized_name == norm for s in self.services.values())

    def _norm(self, name: str) -> str:
        return name.lower().replace('-', '').replace('_', '').replace('.', '')

    def match_service(self, name: str):
        """Return canonical service name for any variant, or None."""
        norm = self._norm(name)
        for svc_name, svc in self.services.items():
            if svc.normalized_name == norm:
                return svc_name
        return None

    def _yaml_files(self):
        files = list(self.project_path.rglob('*.yaml')) + \
                list(self.project_path.rglob('*.yml'))
        return [f for f in files if not self._is_vendor(f)]

    def _compose_files(self):
        names = ['docker-compose.yml', 'docker-compose.yaml',
                 'docker-compose.override.yml', 'docker-compose.override.yaml']
        result = []
        for name in names:
            for p in self.project_path.rglob(name):
                if not self._is_vendor(p):
                    result.append(p)
        return result

    @staticmethod
    def _is_vendor(path: Path) -> bool:
        skip = {'node_modules', 'vendor', '.git', 'build', 'dist',
                'target', '__pycache__', '.venv', 'venv', 'env',
                '.idea', '.vscode', 'bin', 'obj', 'generated', '.tox'}
        return any(p in skip for p in path.parts)
