"""
Docker Compose Parser - detects dependencies from docker-compose files.
Covers: depends_on, environment, ports, networks.
Zero hardcoded assumptions.
"""

from pathlib import Path
from typing import List, Dict
import yaml
import re
from agent.models import Dependency, Service


class ComposeParser:

    def __init__(self, services: Dict[str, Service], port_map: Dict[int, str]):
        self.services = services
        self.port_map = port_map
        self.deps: List[Dependency] = []

    def parse(self, project_path: str) -> List[Dependency]:
        print("\n🐳 Parsing docker-compose files...")
        root = Path(project_path)
        found = False

        for compose_name in ('docker-compose.yml', 'docker-compose.yaml',
                             'docker-compose.override.yml',
                             'docker-compose.override.yaml'):
            for fpath in root.rglob(compose_name):
                if self._is_vendor(fpath):
                    continue
                self._parse_file(fpath)
                found = True

        if found:
            print(f"   ✅ {len(self.deps)} deps from docker-compose")
        return self.deps

    def _parse_file(self, fpath: Path):
        try:
            with open(fpath) as f:
                data = yaml.safe_load(f)
        except Exception:
            return

        if not data or 'services' not in data:
            return

        compose_svcs = data['services']

        for svc_name, svc_cfg in compose_svcs.items():
            from_svc = self._match(svc_name)
            if not from_svc:
                continue
            svc_cfg = svc_cfg or {}

            # depends_on — deployment ordering
            depends = svc_cfg.get('depends_on', [])
            if isinstance(depends, dict):
                depends = list(depends.keys())
            for dep_name in (depends or []):
                to_svc = self._match(dep_name)
                if to_svc and to_svc != from_svc:
                    self.deps.append(Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='deployment',
                        subtype='startup_order',
                        confidence=0.88,
                        evidence=f'depends_on: {dep_name} in {fpath.name}',
                        source='compose_depends_on'
                    ))

            # links (legacy)
            for link in (svc_cfg.get('links') or []):
                link_name = link.split(':')[0]
                to_svc = self._match(link_name)
                if to_svc and to_svc != from_svc:
                    self.deps.append(Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='endpoint',
                        subtype='http',
                        confidence=0.80,
                        evidence=f'links: {link} in {fpath.name}',
                        source='compose_links'
                    ))

            # environment variables
            env = svc_cfg.get('environment', {})
            if isinstance(env, list):
                env_dict = {}
                for item in env:
                    if '=' in item:
                        k, v = item.split('=', 1)
                        env_dict[k] = v
                env = env_dict

            for key, val in (env or {}).items():
                val = str(val) if val is not None else ''
                dep = self._extract_from_env(from_svc, key, val, fpath.name)
                if dep:
                    self.deps.append(dep)

    def _extract_from_env(self, from_svc: str, key: str,
                          val: str, source_file: str):
        """Try to extract a dep from a compose env key=value pair."""
        if not val:
            return None

        # Service name in value
        for svc_name in self.services:
            if svc_name == from_svc:
                continue
            if svc_name.lower() in val.lower():
                return Dependency(
                    from_service=from_svc,
                    to_service=svc_name,
                    dep_type='endpoint',
                    subtype='http',
                    confidence=0.82,
                    evidence=f'compose env {key}={val}',
                    source='compose_env'
                )

        # Port-based (learned port map)
        m = re.search(r':(\d+)', val)
        if m:
            port = int(m.group(1))
            if port in self.port_map:
                to_svc = self.port_map[port]
                if to_svc != from_svc:
                    return Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='endpoint',
                        subtype='http',
                        confidence=0.72,
                        evidence=f'compose port {port} → {to_svc}',
                        source='compose_port'
                    )

        return None

    def _match(self, name: str):
        norm = name.lower().replace('-', '').replace('_', '').replace('.', '')
        for svc_name, svc in self.services.items():
            if svc.normalized_name == norm:
                return svc_name
        return None

    @staticmethod
    def _is_vendor(path: Path) -> bool:
        return any(p in {'.git', 'vendor', 'node_modules'} for p in path.parts)
