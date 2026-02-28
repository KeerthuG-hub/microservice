"""
LLM Analyzer - invoked ONLY under specific trigger conditions.
Per-service batched analysis, not per-file.
Validates output against known services.
"""

from pathlib import Path
from typing import List, Dict, Optional, Set
import json
import re
from agent.models import Dependency, Service, ProjectContext
from agent.llm.llm_client import LLMClient


# Files that give LLM the most context per token
PRIORITY_FILES = [
    'main.go', 'main.py', 'main.js', 'main.ts',
    'server.go', 'server.py', 'server.js', 'server.ts',
    'app.py', 'app.go', 'app.js', 'app.ts',
    'index.js', 'index.ts',
    'client.go', 'client.py', 'client.js',
    'handler.go', 'routes.js', 'routes.ts',
    'application.yml', 'application.yaml', '.env',
]

MAX_FILES = 6
MAX_CHARS_PER_FILE = 1500

SKIP_DIRS = {'node_modules', 'vendor', '.git', 'build', 'dist',
             'target', '__pycache__', '.venv', 'venv', 'generated'}
SKIP_SUFFIXES = {'.pb.go', '_pb2.py', '_pb2_grpc.py', '.pb.js',
                 '_test.go', '.test.js', '.spec.ts'}


class LLMAnalyzer:

    CONF_MAP = {'high': 0.88, 'medium': 0.70, 'low': 0.52}

    def __init__(self, services: Dict[str, Service],
                 context: ProjectContext,
                 existing_deps: List[Dependency]):
        self.services = services
        self.context = context
        self.existing_deps = existing_deps
        self.deps: List[Dependency] = []
        self._client: Optional[LLMClient] = None

    # -------------------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------------------

    def analyze(self) -> List[Dependency]:
        print("\n🤖 Running LLM analysis (targeted)...")

        services_to_analyze = self._select_services()
        if not services_to_analyze:
            print("   ✅ All services have sufficient static coverage — LLM skipped")
            return []

        print(f"   📋 LLM needed for {len(services_to_analyze)} services")

        try:
            self._client = LLMClient()
        except Exception as e:
            print(f"   ⚠️  LLM unavailable: {e}. Skipping.")
            return []

        for svc_name in services_to_analyze:
            svc = self.services[svc_name]
            self._analyze_service(svc_name, svc)

        print(f"   ✅ {len(self.deps)} deps from LLM")
        return self.deps

    # -------------------------------------------------------------------------
    # TRIGGER CONDITIONS
    # -------------------------------------------------------------------------

    def _select_services(self) -> List[str]:
        """
        Trigger LLM for a service if:
        1. Zero deps found AND has source code files
        2. Low average confidence (< 0.65) AND has source code
        3. User noted custom framework in context
        """
        # Count existing deps per service
        svc_dep_count: Dict[str, int] = {s: 0 for s in self.services}
        svc_confidence_sum: Dict[str, float] = {s: 0.0 for s in self.services}

        for dep in self.existing_deps:
            if dep.dep_type in ('observability', 'external'):
                continue
            svc_dep_count[dep.from_service] = svc_dep_count.get(dep.from_service, 0) + 1
            svc_confidence_sum[dep.from_service] = (
                svc_confidence_sum.get(dep.from_service, 0.0) + dep.confidence
            )

        to_analyze = []
        for svc_name, svc in self.services.items():
            count = svc_dep_count.get(svc_name, 0)
            if count == 0 or svc.language == 'unknown':
                # Trigger 1: zero deps
                if self._has_source_code(svc):
                    to_analyze.append(svc_name)
            elif count > 0:
                avg_conf = svc_confidence_sum.get(svc_name, 0) / count
                if avg_conf < 0.65:
                    # Trigger 2: low confidence
                    if self._has_source_code(svc):
                        to_analyze.append(svc_name)

        # Trigger 3: custom framework
        if self.context.custom_notes:
            for svc_name in self.services:
                if svc_name not in to_analyze:
                    to_analyze.append(svc_name)

        return to_analyze

    def _has_source_code(self, svc: Service) -> bool:
        svc_path = Path(svc.path)
        code_exts = {'.go', '.py', '.java', '.js', '.ts', '.rs', '.rb', '.cs', '.php'}
        for ext in code_exts:
            if list(svc_path.rglob(f'*{ext}')):
                return True
        return False

    # -------------------------------------------------------------------------
    # PER-SERVICE ANALYSIS
    # -------------------------------------------------------------------------

    def _analyze_service(self, svc_name: str, svc: Service):
        files = self._select_files(svc)
        if not files:
            return

        file_contents = []
        for fpath in files[:MAX_FILES]:
            try:
                content = fpath.read_text(errors='ignore')[:MAX_CHARS_PER_FILE]
                file_contents.append(f"### {fpath.name}\n{content}")
            except Exception:
                pass

        if not file_contents:
            return

        known_services_list = list(self.services.keys())
        prompt = self._build_prompt(svc_name, svc.language,
                                     known_services_list, file_contents)

        raw = self._client.complete(prompt)
        if not raw:
            return

        deps = self._parse_llm_response(svc_name, raw)
        self.deps.extend(deps)

    def _select_files(self, svc: Service) -> List[Path]:
        svc_path = Path(svc.path)
        selected = []

        # Priority files first
        for name in PRIORITY_FILES:
            p = svc_path / name
            if p.exists() and not self._should_skip(p):
                selected.append(p)

        # Add any other source files up to MAX_FILES
        code_exts = {'.go', '.py', '.java', '.js', '.ts', '.rs', '.rb', '.cs'}
        for ext in code_exts:
            for p in svc_path.rglob(f'*{ext}'):
                if len(selected) >= MAX_FILES:
                    break
                if not self._should_skip(p) and p not in selected:
                    selected.append(p)

        return selected[:MAX_FILES]

    # -------------------------------------------------------------------------
    # PROMPT
    # -------------------------------------------------------------------------

    def _build_prompt(self, svc_name: str, language: str,
                       known_services: List[str],
                       file_contents: List[str]) -> str:
        custom_note = (f"\n\nProject note: {self.context.custom_notes}"
                       if self.context.custom_notes else "")

        services_str = ', '.join(known_services)
        files_str = '\n\n'.join(file_contents)

        return f"""You are analyzing a microservice project.

KNOWN SERVICES: {services_str}

SERVICE TO ANALYZE: {svc_name} (language: {language}){custom_note}

Analyze the following source files for ALL dependency types:
- endpoint: HTTP, gRPC, GraphQL, WebSocket calls to other services
- async: Kafka, RabbitMQ, SQS, NATS topic/queue usage
- data: shared database, shared table, shared Redis namespace
- build: proto imports, internal library imports
- deployment: startup ordering, init dependencies
- infrastructure: postgres, mysql, mongodb, redis, elasticsearch, s3, vault
- observability: prometheus, jaeger, zipkin, opentelemetry (label separately)
- external: stripe, sendgrid, twilio, auth0, aws, gcp, azure

SOURCE FILES:
{files_str}

INSTRUCTIONS:
1. Only output dependencies where "to" is in the KNOWN SERVICES list OR is a well-known infrastructure/external provider
2. Do NOT invent service names
3. Return ONLY valid JSON, no markdown, no explanation

OUTPUT FORMAT (JSON array):
[
  {{
    "from": "service-name",
    "to": "target-service-or-provider",
    "type": "endpoint",
    "subtype": "http",
    "topic_or_table": null,
    "confidence": "high",
    "evidence": "short description of what you found"
  }}
]"""

    # -------------------------------------------------------------------------
    # RESPONSE PARSING
    # -------------------------------------------------------------------------

    def _parse_llm_response(self, from_svc: str, raw: str) -> List[Dependency]:
        deps = []

        # Strip markdown fences if present
        text = re.sub(r'```(?:json)?', '', raw).strip()

        # Find JSON array
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if not m:
            return []

        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

        known_providers = {
            'kafka', 'rabbitmq', 'redis', 'mongodb', 'postgres', 'mysql',
            'elasticsearch', 'vault', 'nats', 'prometheus', 'jaeger', 'zipkin',
            'opentelemetry', 'stripe', 'paypal', 'braintree', 'sendgrid',
            'twilio', 'mailgun', 'auth0', 'okta', 'keycloak',
            'aws', 'gcp', 'azure', 'datadog', 'grafana',
        }

        for item in items:
            if not isinstance(item, dict):
                continue

            to = item.get('to', '').strip()
            if not to:
                continue

            # Validate: to must be a known service or known provider
            to_norm = to.lower().replace('-', '').replace('_', '')
            is_known_svc = any(
                svc.normalized_name == to_norm
                for svc in self.services.values()
            )
            is_known_provider = any(p in to.lower() for p in known_providers)

            if not is_known_svc and not is_known_provider:
                continue   # Reject hallucinated names

            # Resolve to canonical service name
            canonical_to = to
            if is_known_svc:
                for svc_name, svc in self.services.items():
                    if svc.normalized_name == to_norm:
                        canonical_to = svc_name
                        break

            conf_str = str(item.get('confidence', 'medium')).lower()
            confidence = self.CONF_MAP.get(conf_str, 0.70)

            deps.append(Dependency(
                from_service=from_svc,
                to_service=canonical_to,
                dep_type=item.get('type', 'endpoint'),
                subtype=item.get('subtype'),
                confidence=confidence,
                evidence=item.get('evidence', 'llm analysis'),
                source='llm_analysis',
                topic_or_table=item.get('topic_or_table')
            ))

        return deps

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _should_skip(self, path: Path) -> bool:
        if any(p in SKIP_DIRS for p in path.parts):
            return True
        return any(path.name.endswith(s) for s in SKIP_SUFFIXES)
