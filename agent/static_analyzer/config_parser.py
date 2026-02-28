"""
Config Parser - detects dependencies from:
application.yml, application.yaml, .env, .env.example,
config.yml, .properties, appsettings.json, etc.
Zero hardcoded assumptions.
"""

from pathlib import Path
from typing import List, Dict
import re
import json
try:
    import yaml
except ImportError:
    yaml = None

from agent.models import Dependency, Service


CONFIG_FILENAMES = {
    'application.yml', 'application.yaml',
    'application-dev.yml', 'application-prod.yml',
    'application.properties',
    'config.yml', 'config.yaml', 'config.json',
    '.env', '.env.example', '.env.sample', '.env.local',
    'appsettings.json', 'appsettings.Development.json',
    'settings.py', 'settings.yml',
    'bootstrap.yml', 'bootstrap.yaml',
    'values.yaml',  # Helm values
}

INFRA_CONN_PATTERNS = [
    (r'postgres(?:ql)?://([^\s"\']+)', 'postgres'),
    (r'postgresql://([^\s"\']+)', 'postgres'),
    (r'mysql://([^\s"\']+)', 'mysql'),
    (r'mongodb(?:\+srv)?://([^\s"\']+)', 'mongodb'),
    (r'redis://([^\s"\']+)', 'redis'),
    (r'rediss://([^\s"\']+)', 'redis'),
    (r'amqp(?:s)?://([^\s"\']+)', 'rabbitmq'),
    (r'elasticsearch://([^\s"\']+)', 'elasticsearch'),
    (r'https?://([^\s"\']+):9200', 'elasticsearch'),
    (r'vault://([^\s"\']+)', 'vault'),
    (r'nats://([^\s"\']+)', 'nats'),
]


class ConfigParser:

    def __init__(self, services: Dict[str, Service]):
        self.services = services
        self.deps: List[Dependency] = []

    def parse(self, project_path: str) -> List[Dependency]:
        print("\n⚙️  Parsing config files...")
        root = Path(project_path)

        for svc_name, svc in self.services.items():
            svc_path = Path(svc.path)
            for fname in CONFIG_FILENAMES:
                fpath = svc_path / fname
                if fpath.exists():
                    self._parse_config_file(svc_name, fpath)
            # Also check one level deeper
            for fname in CONFIG_FILENAMES:
                for fpath in svc_path.glob(f'**/{fname}'):
                    if not self._is_vendor(fpath):
                        self._parse_config_file(svc_name, fpath)

        print(f"   ✅ {len(self.deps)} deps from config files")
        return self.deps

    def _parse_config_file(self, from_svc: str, fpath: Path):
        try:
            content = fpath.read_text(errors='ignore')
        except Exception:
            return

        # Connection strings
        for pat, infra_type in INFRA_CONN_PATTERNS:
            for m in re.finditer(pat, content, re.IGNORECASE):
                host = m.group(1).split('/')[0].split('@')[-1].split(':')[0]
                to_svc = self._match(host) or infra_type
                if to_svc != from_svc:
                    self._add(Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='infrastructure',
                        subtype=infra_type,
                        confidence=0.88,
                        evidence=f'conn string in {fpath.name}',
                        source='config_conn_string'
                    ))

        # Kafka bootstrap-servers
        for m in re.finditer(r'bootstrap[-._]?servers?\s*[=:]\s*([^\s"\']+)', content, re.IGNORECASE):
            self._add(Dependency(
                from_service=from_svc,
                to_service='kafka',
                dep_type='infrastructure',
                subtype='kafka',
                confidence=0.88,
                evidence=f'bootstrap-servers in {fpath.name}',
                source='config_kafka'
            ))

        # Service URL config entries (e.g., services.payment.url: ...)
        for m in re.finditer(r'(?:url|host|address|endpoint)\s*[=:]\s*["\']?(https?://[^\s"\']+)',
                              content, re.IGNORECASE):
            url = m.group(1)
            dep = self._url_to_dep(from_svc, url, 'http', fpath.name)
            if dep:
                self._add(dep)

        # Feature flag gated URLs
        is_feature_flagged = bool(re.search(r'feature\.', content, re.IGNORECASE))

        # YAML structured parsing (if available)
        if yaml and fpath.suffix in ('.yml', '.yaml'):
            try:
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    self._walk_yaml(from_svc, data, is_feature_flagged, fpath.name)
            except Exception:
                pass

        # JSON
        if fpath.suffix == '.json':
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    self._walk_yaml(from_svc, data, is_feature_flagged, fpath.name)
            except Exception:
                pass

        # .env format
        if fpath.name.startswith('.env') or fpath.suffix == '.env':
            for line in content.splitlines():
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, _, v = line.partition('=')
                    self._check_env_kv(from_svc, k.strip(), v.strip(), fpath.name)

        # .properties format
        if fpath.suffix == '.properties':
            for line in content.splitlines():
                if '=' in line and not line.startswith('#'):
                    k, _, v = line.partition('=')
                    self._check_env_kv(from_svc, k.strip(), v.strip(), fpath.name)

    def _walk_yaml(self, from_svc: str, data, feature_flagged: bool, src: str, depth: int = 0):
        """Recursively walk YAML/JSON structure looking for service references."""
        if depth > 8:
            return
        if isinstance(data, dict):
            for k, v in data.items():
                k_lower = str(k).lower()
                if k_lower in ('url', 'host', 'address', 'endpoint', 'uri',
                                'baseurl', 'base_url', 'service_url'):
                    if isinstance(v, str):
                        dep = self._url_to_dep(from_svc, v, 'http', src)
                        if dep:
                            dep.conditional = feature_flagged
                            self._add(dep)
                elif k_lower in ('bootstrap_servers', 'bootstrap-servers'):
                    self._add(Dependency(
                        from_service=from_svc,
                        to_service='kafka',
                        dep_type='infrastructure',
                        subtype='kafka',
                        confidence=0.88,
                        evidence=f'{k} in {src}',
                        source='config_yaml'
                    ))
                else:
                    self._walk_yaml(from_svc, v, feature_flagged, src, depth + 1)
        elif isinstance(data, list):
            for item in data:
                self._walk_yaml(from_svc, item, feature_flagged, src, depth + 1)

    def _check_env_kv(self, from_svc: str, key: str, val: str, src: str):
        """Check a KEY=VALUE pair for service references."""
        if not val:
            return
        # Service name in value (URL or plain hostname)
        for svc_name in self.services:
            if svc_name == from_svc:
                continue
            if svc_name.lower() in val.lower():
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=svc_name,
                    dep_type='endpoint',
                    subtype='http',
                    confidence=0.80,
                    evidence=f'config {key}={val[:60]} in {src}',
                    source='config_env_kv'
                ))
                return

    def _url_to_dep(self, from_svc: str, url: str, subtype: str, src: str):
        if not url:
            return None
        clean = re.sub(r'^https?://', '', url)
        clean = re.sub(r'^grpc://', '', clean)
        host = clean.split('/')[0].split('@')[-1].split(':')[0]
        to_svc = self._match(host)
        if not to_svc or to_svc == from_svc:
            return None
        return Dependency(
            from_service=from_svc,
            to_service=to_svc,
            dep_type='endpoint',
            subtype=subtype,
            confidence=0.80,
            evidence=f'config url={url[:60]} in {src}',
            source='config_url'
        )

    def _match(self, name: str):
        norm = name.lower().replace('-', '').replace('_', '').replace('.', '')
        for svc_name, svc in self.services.items():
            if svc.normalized_name == norm:
                return svc_name
        return None

    def _add(self, dep: Dependency):
        if dep and dep.from_service and dep.to_service and dep.from_service != dep.to_service:
            self.deps.append(dep)

    @staticmethod
    def _is_vendor(path: Path) -> bool:
        skip = {'node_modules', 'vendor', '.git', '__pycache__',
                '.venv', 'venv', 'generated'}
        return any(p in skip for p in path.parts)
