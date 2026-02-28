"""
Build Parser - detects build-time and external dependencies from:
go.mod, package.json, pom.xml, requirements.txt, Cargo.toml, Gemfile, composer.json
Zero hardcoded service names — learns from project.
"""

from pathlib import Path
from typing import List, Dict, Set
import re
import json
try:
    import yaml
except ImportError:
    yaml = None

from agent.models import Dependency, Service


# Known external SDK package name fragments → (provider, dep_type, subtype)
# NOTE: 'grpc' and 'protobuf' are intentionally excluded — every service imports
# these as language libraries. Importing grpc does NOT mean calling another service.
KNOWN_EXTERNAL = {
    'stripe': ('stripe', 'external', 'stripe'),
    'braintree': ('braintree', 'external', 'braintree'),
    'paypal': ('paypal', 'external', 'paypal'),
    'sendgrid': ('sendgrid', 'external', 'sendgrid'),
    'twilio': ('twilio', 'external', 'twilio'),
    'mailgun': ('mailgun', 'external', 'mailgun'),
    'mandrill': ('mandrill', 'external', 'mandrill'),
    'auth0': ('auth0', 'external', 'auth0'),
    'okta': ('okta', 'external', 'okta'),
    'keycloak': ('keycloak', 'external', 'keycloak'),
    'kafka': ('kafka', 'infrastructure', 'kafka'),
    'rabbitmq': ('rabbitmq', 'infrastructure', 'rabbitmq'),
    'amqp': ('rabbitmq', 'infrastructure', 'rabbitmq'),
    'nats': ('nats', 'infrastructure', 'nats'),
    'redis': ('redis', 'infrastructure', 'redis'),
    'mongodb': ('mongodb', 'infrastructure', 'mongodb'),
    'mongo-driver': ('mongodb', 'infrastructure', 'mongodb'),
    'elasticsearch': ('elasticsearch', 'infrastructure', 'elasticsearch'),
    'postgres': ('postgres', 'infrastructure', 'postgres'),
    'psycopg': ('postgres', 'infrastructure', 'postgres'),
    'asyncpg': ('postgres', 'infrastructure', 'postgres'),
    'mysql': ('mysql', 'infrastructure', 'mysql'),
    'boto3': ('aws', 'external', 'aws'),
    'botocore': ('aws', 'external', 'aws'),
    'aws-sdk': ('aws', 'external', 'aws'),
    '@aws-sdk': ('aws', 'external', 'aws'),
    'google-cloud': ('gcp', 'external', 'gcp'),
    '@google-cloud': ('gcp', 'external', 'gcp'),
    'azure': ('azure', 'external', 'azure'),
    '@azure': ('azure', 'external', 'azure'),
    'opentelemetry': ('opentelemetry', 'observability', 'jaeger'),
    'prometheus-client': ('prometheus', 'observability', 'prometheus'),
    'prom-client': ('prometheus', 'observability', 'prometheus'),
    'jaeger': ('jaeger', 'observability', 'jaeger'),
    'zipkin': ('zipkin', 'observability', 'jaeger'),
    'datadog': ('datadog', 'observability', 'prometheus'),
    'vault': ('vault', 'infrastructure', 'vault'),
}


class BuildParser:

    def __init__(self, services: Dict[str, Service]):
        self.services = services
        self.deps: List[Dependency] = []

    def parse(self, project_path: str) -> List[Dependency]:
        print("\n🏗️  Parsing build files...")
        root = Path(project_path)

        for svc_name, svc in self.services.items():
            svc_path = Path(svc.path)
            self._parse_go_mod(svc_name, svc_path)
            self._parse_package_json(svc_name, svc_path)
            self._parse_pom_xml(svc_name, svc_path)
            self._parse_requirements(svc_name, svc_path)
            self._parse_cargo_toml(svc_name, svc_path)
            self._parse_gemfile(svc_name, svc_path)
            self._parse_build_gradle(svc_name, svc_path)
            self._parse_dockerfile(svc_name, svc_path)

        print(f"   ✅ {len(self.deps)} deps from build files")
        return self.deps

    # -------------------------------------------------------------------------
    # GO
    # -------------------------------------------------------------------------

    def _parse_go_mod(self, from_svc: str, svc_path: Path):
        go_mod = svc_path / 'go.mod'
        if not go_mod.exists():
            return
        try:
            content = go_mod.read_text(errors='ignore')

            # Parse require block entries for external/known packages
            for m in re.finditer(r'^\s+([^\s]+)\s+v[^\s]+', content, re.MULTILINE):
                pkg = m.group(1)
                self._check_pkg(from_svc, pkg, 'go.mod')

            # Cross-service internal imports: only match if service name appears
            # as the LAST path segment of a require entry (e.g. .../cartservice),
            # NOT anywhere in the full file (avoids matching module-level comments,
            # go.sum sections, or the owning module's own name).
            in_require = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith('require ('):
                    in_require = True
                    continue
                if in_require and stripped == ')':
                    in_require = False
                    continue
                if stripped.startswith('require ') or in_require:
                    # Last path segment of the module path
                    pkg_path = stripped.split()[0] if stripped.split() else ''
                    last_segment = pkg_path.split('/')[-1].lower().replace('-', '').replace('_', '')
                    for svc_name in self.services:
                        if svc_name == from_svc:
                            continue
                        svc_norm = svc_name.lower().replace('-', '').replace('_', '')
                        if svc_norm == last_segment:   # exact match on last segment only
                            self._add(Dependency(
                                from_service=from_svc,
                                to_service=svc_name,
                                dep_type='build',
                                subtype='internal_lib',
                                confidence=0.75,
                                evidence=f'go.mod requires {pkg_path}',
                                source='build_go_mod'
                            ))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # JAVASCRIPT / NODE
    # -------------------------------------------------------------------------

    def _parse_package_json(self, from_svc: str, svc_path: Path):
        pkg_json = svc_path / 'package.json'
        if not pkg_json.exists():
            return
        try:
            data = json.loads(pkg_json.read_text(errors='ignore'))
            all_deps = {}
            all_deps.update(data.get('dependencies', {}))
            all_deps.update(data.get('devDependencies', {}))
            for pkg in all_deps:
                self._check_pkg(from_svc, pkg, 'package.json')
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # JAVA / MAVEN
    # -------------------------------------------------------------------------

    def _parse_pom_xml(self, from_svc: str, svc_path: Path):
        pom = svc_path / 'pom.xml'
        if not pom.exists():
            return
        try:
            content = pom.read_text(errors='ignore')
            # Extract artifactId from dependencies
            for m in re.finditer(r'<artifactId>([^<]+)</artifactId>', content):
                self._check_pkg(from_svc, m.group(1), 'pom.xml')
            # Internal deps (same groupId prefix)
            for svc_name in self.services:
                if svc_name == from_svc:
                    continue
                if svc_name.lower().replace('-', '') in content.lower().replace('-', ''):
                    self._add(Dependency(
                        from_service=from_svc,
                        to_service=svc_name,
                        dep_type='build',
                        subtype='internal_lib',
                        confidence=0.75,
                        evidence=f'pom.xml references {svc_name}',
                        source='build_pom'
                    ))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # PYTHON
    # -------------------------------------------------------------------------

    def _parse_requirements(self, from_svc: str, svc_path: Path):
        for fname in ('requirements.txt', 'requirements-base.txt',
                      'requirements.in', 'pyproject.toml', 'Pipfile'):
            fpath = svc_path / fname
            if not fpath.exists():
                continue
            try:
                content = fpath.read_text(errors='ignore')
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    pkg = re.split(r'[>=<!\[;]', line)[0].strip()
                    self._check_pkg(from_svc, pkg.lower(), fname)
                    # Internal git+ packages
                    if line.startswith('git+') or line.startswith('-e git+'):
                        for svc_name in self.services:
                            if svc_name != from_svc and svc_name.lower() in line.lower():
                                self._add(Dependency(
                                    from_service=from_svc,
                                    to_service=svc_name,
                                    dep_type='build',
                                    subtype='internal_lib',
                                    confidence=0.80,
                                    evidence=f'requirements.txt git+ {svc_name}',
                                    source='build_requirements'
                                ))
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # RUST
    # -------------------------------------------------------------------------

    def _parse_cargo_toml(self, from_svc: str, svc_path: Path):
        cargo = svc_path / 'Cargo.toml'
        if not cargo.exists():
            return
        try:
            content = cargo.read_text(errors='ignore')
            for m in re.finditer(r'^(\S+)\s*=', content, re.MULTILINE):
                self._check_pkg(from_svc, m.group(1).strip('"'), 'Cargo.toml')
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # RUBY
    # -------------------------------------------------------------------------

    def _parse_gemfile(self, from_svc: str, svc_path: Path):
        gemfile = svc_path / 'Gemfile'
        if not gemfile.exists():
            return
        try:
            content = gemfile.read_text(errors='ignore')
            for m in re.finditer(r"gem\s+['\"]([^'\"]+)['\"]", content):
                self._check_pkg(from_svc, m.group(1), 'Gemfile')
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # JAVA / GRADLE
    # -------------------------------------------------------------------------

    def _parse_build_gradle(self, from_svc: str, svc_path: Path):
        for fname in ('build.gradle', 'build.gradle.kts'):
            fpath = svc_path / fname
            if not fpath.exists():
                continue
            try:
                content = fpath.read_text(errors='ignore')
                for m in re.finditer(r"['\"]([^'\"]+:[^'\"]+)['\"]", content):
                    parts = m.group(1).split(':')
                    if len(parts) >= 2:
                        self._check_pkg(from_svc, parts[1], fname)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # DOCKERFILE base images
    # -------------------------------------------------------------------------

    def _parse_dockerfile(self, from_svc: str, svc_path: Path):
        dockerfile = svc_path / 'Dockerfile'
        if not dockerfile.exists():
            return
        try:
            content = dockerfile.read_text(errors='ignore')
            for m in re.finditer(r'^FROM\s+(\S+)', content, re.MULTILINE):
                image = m.group(1)
                if image.lower() == 'scratch':
                    continue
                img_name = image.split('/')[-1].split(':')[0]
                to_svc = self._match(img_name)
                if to_svc and to_svc != from_svc:
                    self._add(Dependency(
                        from_service=from_svc,
                        to_service=to_svc,
                        dep_type='build',
                        subtype='base_image',
                        confidence=0.82,
                        evidence=f'Dockerfile FROM {image}',
                        source='build_dockerfile'
                    ))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _check_pkg(self, from_svc: str, pkg_name: str, source_file: str):
        """Check if package name maps to an external provider or internal svc."""
        pkg_lower = pkg_name.lower()

        # External / known providers
        for keyword, (provider, dep_type, subtype) in KNOWN_EXTERNAL.items():
            if keyword in pkg_lower:
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=provider,
                    dep_type=dep_type,
                    subtype=subtype,
                    confidence=0.85,
                    evidence=f'{source_file}: {pkg_name}',
                    source='build_file'
                ))
                return

        # Internal service dep: service name must match the LAST segment of the
        # package path (e.g. github.com/org/cartservice → 'cartservice').
        # Substring-anywhere matching creates massive false positives in monorepos
        # where service names appear in module paths, import comments, etc.
        last_segment = pkg_lower.split('/')[-1].replace('-', '').replace('_', '')
        for svc_name in self.services:
            if svc_name == from_svc:
                continue
            svc_norm = svc_name.lower().replace('-', '').replace('_', '')
            if svc_norm == last_segment:   # exact match on last path segment
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=svc_name,
                    dep_type='build',
                    subtype='internal_lib',
                    confidence=0.75,
                    evidence=f'{source_file}: {pkg_name}',
                    source='build_file'
                ))
                return

    def _match(self, name: str) -> str:
        norm = name.lower().replace('-', '').replace('_', '').replace('.', '')
        for svc_name, svc in self.services.items():
            if svc.normalized_name == norm:
                return svc_name
        return None

    def _add(self, dep: Dependency):
        if dep and dep.from_service and dep.to_service and dep.from_service != dep.to_service:
            self.deps.append(dep)
