"""
Code Parser - detects dependencies from source code.
Supports Go, Python, Java, JavaScript/TypeScript, Ruby, Rust, C#, PHP.
Language patterns discovered dynamically - NO hardcoded service names.
"""

from pathlib import Path
from typing import List, Dict, Optional, Set
import re
from agent.models import Dependency, Service


# ---------------------------------------------------------------------------
# Language-specific pattern packs
# All patterns produce named groups: url, service, topic, table, queue
# ---------------------------------------------------------------------------

LANG_PATTERNS = {
    'go': {
        # HTTP clients
        'http_client': [
            r'http\.(?:Get|Post|Put|Delete|Head|Patch)\s*\(\s*"([^"]+)"',
            r'http\.NewRequest\s*\([^,]+,\s*"([^"]+)"',
            r'req\s*:=\s*&http\.Request\{[^}]*URL\s*:[^"]*"([^"]+)"',
            r'url\s*(?::=|=)\s*fmt\.Sprintf\s*\("([^"]+)"',
            r'url\s*(?::=|=)\s*"(https?://[^"]+)"',
        ],
        # gRPC
        'grpc': [
            r'grpc\.Dial\s*\(\s*"?([a-zA-Z0-9_\-.:]+)"?',
            r'grpc\.DialContext\s*\([^,]+,\s*"([^"]+)"',
        ],
        # Kafka
        'kafka_produce': [r'\.WriteMessages\(', r'\.Produce\s*\(', r'NewWriter\s*\(\s*kafka\.WriterConfig'],
        'kafka_topic_str': [r'(?:Topic|topic)\s*:\s*"([^"]+)"'],
        'kafka_consume': [r'kafka\.NewReader\s*\(', r'\.ReadMessage\s*\('],
        # RabbitMQ
        'amqp': [r'amqp\.Dial\s*\(\s*"([^"]+)"'],
        'amqp_publish': [r'\.Publish\s*\(\s*"([^"]*)"'],
        'amqp_consume': [r'\.Consume\s*\(\s*"([^"]*)"'],
        # Redis
        'redis': [r'redis\.NewClient\s*\(\s*&redis\.Options\{[^}]*Addr\s*:\s*"([^"]+)"'],
        # Postgres/MySQL
        'pg': [r'sql\.Open\s*\([^,]+,\s*"([^"]+)"'],
        # Env reads
        'env_read': [r'os\.Getenv\s*\(\s*"([^"]+)"', r'os\.LookupEnv\s*\(\s*"([^"]+)"'],
    },

    'python': {
        'http_client': [
            r'requests\.(?:get|post|put|delete|patch|head)\s*\(\s*"([^"]+)"',
            r'requests\.(?:get|post|put|delete|patch|head)\s*\(\s*f?"([^"]+)"',
            r'httpx\.(?:get|post|put|delete|patch)\s*\(\s*"([^"]+)"',
            r'aiohttp\.ClientSession\s*\(\s*\)\s*\.(?:get|post)\s*\(\s*"([^"]+)"',
            r'urllib\.request\.urlopen\s*\(\s*"([^"]+)"',
            r'requests\.Session\s*\(\s*\)',  # session usage
            r'base_url\s*=\s*["\']([^"\']+)["\']',
        ],
        'grpc': [
            r'grpc\.insecure_channel\s*\(\s*"([^"]+)"',
            r'grpc\.secure_channel\s*\(\s*"([^"]+)"',
        ],
        'kafka_producer': [
            r'KafkaProducer\s*\(',
            r'producer\.send\s*\(\s*["\']([^"\']+)["\']',
        ],
        'kafka_consumer': [
            r'KafkaConsumer\s*\(',
            r'KafkaConsumer\s*\(["\']([^"\']+)["\']',
        ],
        'kafka_topic_str': [r'topic\s*=\s*["\']([^"\']+)["\']'],
        'celery': [
            r'@(?:app|celery)\.task',
            r'\.apply_async\s*\(',
        ],
        'redis': [r'redis\.Redis\s*\(|redis\.StrictRedis\s*\(|aioredis\.create_redis'],
        'pg': [r'psycopg2\.connect\s*\(|asyncpg\.connect|create_engine\s*\('],
        'mongo': [r'MongoClient\s*\('],
        'env_read': [r'os\.(?:environ\.get|getenv)\s*\(\s*["\']([^"\']+)["\']'],
        # Feature flags
        'feature_flag': [r'if\s+.*feature.*enabled|if\s+.*flag.*|if\s+os\.getenv\s*\([^)]+\)\s*=='],
    },

    'java': {
        'http_client': [
            r'(?:new\s+)?RestTemplate\s*\(',
            r'WebClient\.create\s*\(\s*"([^"]+)"',
            r'HttpClient\.(?:newHttpClient|newBuilder)',
            r'OkHttpClient',
            r'Feign\.builder',
            r'@FeignClient\s*\([^)]*value\s*=\s*"([^"]+)"',
            r'@FeignClient\s*\([^)]*name\s*=\s*"([^"]+)"',
        ],
        'grpc': [
            r'ManagedChannelBuilder\.forAddress\s*\(\s*"([^"]+)"',
            r'ManagedChannelBuilder\.forTarget\s*\(\s*"([^"]+)"',
        ],
        'kafka_producer': [
            r'KafkaTemplate',
            r'kafkaTemplate\.send\s*\(\s*"([^"]+)"',
        ],
        'kafka_consumer': [
            r'@KafkaListener\s*\([^)]*topics\s*=\s*["{]([^}"]+)',
        ],
        'amqp': [r'@RabbitListener\s*\([^)]*queues\s*=\s*"([^"]+)"',
                 r'rabbitTemplate\.convertAndSend'],
        'pg': [r'DataSource|JdbcTemplate|@Repository|HikariDataSource'],
        'env_read': [
            r'System\.getenv\s*\(\s*"([^"]+)"',
            r'@Value\s*\(\s*"\$\{([^}]+)\}"',
            r'environment\.getProperty\s*\(\s*"([^"]+)"',
        ],
    },

    'javascript': {
        'http_client': [
            r'axios\.(?:get|post|put|delete|patch)\s*\(\s*["\`]([^"\`]+)["\`]',
            r'fetch\s*\(\s*["\`]([^"\`]+)["\`]',
            r'(?:got|superagent)\.(?:get|post)\s*\(\s*["\`]([^"\`]+)["\`]',
            r'new\s+(?:XMLHttpRequest|HttpClient)',
            r'(?:axios|http)\s*\.\s*defaults\s*\.\s*baseURL\s*=\s*["\`]([^"\`]+)["\`]',
            r'const\s+\w+\s*=\s*new\s+ApolloClient\s*\(\s*\{[^}]*uri\s*:\s*["\`]([^"\`]+)["\`]',
        ],
        'websocket': [
            r'new\s+WebSocket\s*\(\s*["\`]([^"\`]+)["\`]',
            r'io\.connect\s*\(\s*["\`]([^"\`]+)["\`]',
        ],
        'kafka': [
            r'new\s+Kafka\s*\(',
            r'producer\.send\s*\(\{[^}]*topic\s*:\s*["\`]([^"\`]+)["\`]',
            r'consumer\.subscribe\s*\(\{[^}]*topics?\s*:\s*\[?["\`]([^"\`]+)["\`]',
        ],
        'amqp': [r'amqplib\.connect\s*\(\s*["\`]([^"\`]+)["\`]'],
        'redis': [r'createClient\s*\(\s*\{?[^}]*url\s*:\s*["\`]([^"\`]+)["\`]',
                  r'Redis\s*\(\s*["\`]([^"\`]+)["\`]'],
        'env_read': [r'process\.env\s*\.\s*([A-Z_][A-Z0-9_]+)',
                     r'process\.env\s*\[\s*["\']([^"\']+)["\']'],
        'graphql': [
            r'new\s+ApolloClient',
            r'graphql-request',
            r'createClient\s*\(\s*\{[^}]*url\s*:\s*["\`]([^"\`]+)["\`]',
        ],
        'feature_flag': [r'process\.env\s*\.\s*\w+\s*===?\s*["\`]true["\`]',
                         r'if\s*\(\s*feature\w*\s*\)'],
    },

    'typescript': None,   # Same as javascript — handled below

    'ruby': {
        'http_client': [
            r'Faraday\.new\s*\(\s*["\']([^"\']+)["\']',
            r'Net::HTTP\.(?:get|post)\s*\(',
            r'HTTParty\.',
            r'RestClient\.',
        ],
        'kafka': [r'Rdkafka|ruby-kafka|WaterDrop'],
        'amqp': [r'Bunny\.new\s*\(\s*["\']([^"\']+)["\']'],
        'redis': [r'Redis\.new\s*\('],
        'pg': [r'PG\.connect\s*\(|ActiveRecord::Base\.establish_connection'],
        'env_read': [r'ENV\s*\[\s*["\']([^"\']+)["\']', r'ENV\.fetch\s*\(\s*["\']([^"\']+)["\']'],
    },

    'rust': {
        'http_client': [
            r'reqwest::(?:get|Client::new)',
            r'hyper::Client::new',
        ],
        'kafka': [r'rdkafka'],
        'amqp': [r'lapin'],
        'redis': [r'redis::Client::open\s*\(\s*"([^"]+)"'],
        'pg': [r'tokio_postgres|sqlx::postgres'],
        'env_read': [r'env::var\s*\(\s*"([^"]+)"', r'std::env::var\s*\(\s*"([^"]+)"'],
    },

    'csharp': {
        'http_client': [
            r'new\s+HttpClient\s*\(',
            r'_httpClient\.(?:GetAsync|PostAsync|PutAsync|DeleteAsync)\s*\(\s*"([^"]+)"',
            r'services\.AddHttpClient\s*\(\s*"([^"]+)"',
        ],
        'grpc': [r'new\s+Channel\s*\(\s*"([^"]+)"'],
        'kafka': [r'ProducerBuilder|ConsumerBuilder|IProducer|IConsumer'],
        'amqp': [r'new\s+ConnectionFactory\s*\{[^}]*HostName\s*=\s*"([^"]+)"'],
        'redis': [r'ConnectionMultiplexer\.Connect\s*\(\s*"([^"]+)"'],
        'pg': [r'new\s+NpgsqlConnection\s*\(\s*"([^"]+)"', r'UseSqlServer|UseNpgsql|UseMySql'],
        'env_read': [r'Environment\.GetEnvironmentVariable\s*\(\s*"([^"]+)"',
                     r'Configuration\s*\[\s*"([^"]+)"\s*\]'],
    },
}

# Alias typescript to javascript
LANG_PATTERNS['typescript'] = LANG_PATTERNS['javascript']


# ---------------------------------------------------------------------------
# External SDK detection — these map to (provider, dep_type, subtype)
# ---------------------------------------------------------------------------

EXTERNAL_SDKS = {
    # Payment
    r'stripe': ('stripe', 'external', 'stripe'),
    r'braintree': ('braintree', 'external', 'braintree'),
    r'paypal': ('paypal', 'external', 'paypal'),
    r'square': ('square', 'external', 'square'),
    # Email/SMS
    r'sendgrid': ('sendgrid', 'external', 'sendgrid'),
    r'twilio': ('twilio', 'external', 'twilio'),
    r'mailgun': ('mailgun', 'external', 'mailgun'),
    r'mandrill': ('mandrill', 'external', 'mandrill'),
    # Identity
    r'auth0': ('auth0', 'external', 'auth0'),
    r'okta': ('okta', 'external', 'okta'),
    r'keycloak': ('keycloak', 'external', 'keycloak'),
    # Cloud
    r'boto3|aws-sdk|@aws-sdk': ('aws', 'external', 'aws'),
    r'google\.cloud|@google-cloud': ('gcp', 'external', 'gcp'),
    r'azure-sdk|@azure': ('azure', 'external', 'azure'),
    # Observability
    r'opentelemetry|otel': ('opentelemetry', 'observability', 'jaeger'),
    r'jaeger': ('jaeger', 'observability', 'jaeger'),
    r'zipkin': ('zipkin', 'observability', 'jaeger'),
    r'prometheus_client|prom-client|prometheus/client_golang': ('prometheus', 'observability', 'prometheus'),
    r'datadog': ('datadog', 'observability', 'prometheus'),
}

# Connection string patterns → infrastructure dep
CONN_PATTERNS = [
    (r'postgres(?:ql)?://([^/\s"\']+)', 'postgres'),
    (r'postgresql://([^/\s"\']+)', 'postgres'),
    (r'mysql://([^/\s"\']+)', 'mysql'),
    (r'mongodb(?:\+srv)?://([^/\s"\']+)', 'mongodb'),
    (r'redis://([^/\s"\']+)', 'redis'),
    (r'amqp(?:s)?://([^/\s"\']+)', 'rabbitmq'),
    (r'elasticsearch://([^/\s"\']+)', 'elasticsearch'),
    (r'https?://([^/\s"\']+):9200', 'elasticsearch'),
    (r'vault://([^/\s"\']+)', 'vault'),
    (r'VAULT_ADDR\s*=\s*["\']([^"\']+)["\']', 'vault'),
]

FEATURE_FLAG_PATTERNS = [
    r'if\s+.*feature.*enabled',
    r'if\s+.*flag\s*[=!]=',
    r'FeatureFlag\.',
    r'LaunchDarkly',
    r'Unleash',
    r'ff\.',
]


class CodeParser:
    """Multi-language source code parser."""

    # Files to prioritize (entry points and client files)
    PRIORITY_2 = {
        'main.go', 'main.py', 'main.js', 'main.ts',
        'index.js', 'index.ts', 'app.py', 'app.go',
        'server.go', 'server.py', 'server.js', 'server.ts',
        'app.js', 'app.ts',
    }
    PRIORITY_3_PATTERNS = [
        '*client*.go', '*client*.py', '*client*.js', '*client*.ts',
        '*handler*.go', '*handler*.py', '*routes*.js', '*routes*.ts',
        '*service*.go', '*service*.py', '*service*.js',
        '*repository*.go', '*repository*.java',
    ]
    SKIP_PATTERNS = {
        '_test.go', '_test.py', '.test.js', '.test.ts',
        '.spec.js', '.spec.ts', '_spec.rb',
        '.pb.go', '_pb2.py', '_pb2_grpc.py', '.pb.js',
    }
    SKIP_DIRS = {
        'node_modules', 'vendor', '.git', 'build', 'dist',
        'target', '__pycache__', '.venv', 'venv', 'env',
        '.idea', '.vscode', 'bin', 'obj', 'generated', '.tox',
        '__generated__', 'proto', 'pb', 'stubs',
    }

    def __init__(self, services: Dict[str, Service],
                 port_map: Dict[int, str],
                 env_patterns: Set[str]):
        self.services = services
        self.port_map = port_map
        self.env_patterns = env_patterns
        self.deps: List[Dependency] = []
        self._seen_strings: Dict[str, Set[str]] = {}   # string → {service names}

    # -------------------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------------------

    def parse(self, project_path: str) -> List[Dependency]:
        print("\n💻 Parsing source code...")
        root = Path(project_path)

        for svc_name, svc in self.services.items():
            svc_path = Path(svc.path)
            if not svc_path.exists():
                continue

            lang = svc.language
            patterns = LANG_PATTERNS.get(lang, {})

            files = self._select_files(svc_path, lang)
            for fpath in files:
                self._parse_file(svc_name, fpath, lang, patterns)

        print(f"   ✅ {len(self.deps)} deps from code")
        return self.deps

    def get_string_map(self) -> Dict[str, Set[str]]:
        """Return shared string map for semantic analysis."""
        return self._seen_strings

    # -------------------------------------------------------------------------
    # FILE SELECTION
    # -------------------------------------------------------------------------

    def _select_files(self, svc_path: Path, lang: str) -> List[Path]:
        """Select files in priority order — avoid reading everything."""
        selected: List[Path] = []
        ext_map = {
            'go': ['.go'], 'python': ['.py'], 'java': ['.java'],
            'javascript': ['.js'], 'typescript': ['.ts'],
            'ruby': ['.rb'], 'rust': ['.rs'], 'csharp': ['.cs'],
            'php': ['.php'], 'kotlin': ['.kt'], 'scala': ['.scala'],
        }
        exts = ext_map.get(lang, [])
        if not exts:
            return []

        # Priority 2: known entry point names
        for name in self.PRIORITY_2:
            p = svc_path / name
            if p.exists() and not self._should_skip(p):
                selected.append(p)

        # Priority 3: *client*, *handler*, *routes* etc.
        for pattern in self.PRIORITY_3_PATTERNS:
            for p in svc_path.rglob(pattern):
                if not self._should_skip(p) and p not in selected:
                    selected.append(p)

        # Fill up to 20 files with remaining source files
        remaining_budget = max(0, 20 - len(selected))
        for ext in exts:
            for p in svc_path.rglob(f'*{ext}'):
                if remaining_budget <= 0:
                    break
                if not self._should_skip(p) and p not in selected:
                    selected.append(p)
                    remaining_budget -= 1

        return selected

    def _should_skip(self, path: Path) -> bool:
        if any(part in self.SKIP_DIRS for part in path.parts):
            return True
        name = path.name
        if any(name.endswith(sk) for sk in self.SKIP_PATTERNS):
            return True
        # Skip generated proto code
        if name.endswith(('.pb.go', '_pb2.py', '_pb2_grpc.py',
                          '.pb.js', '.pb.ts', '_grpc.pb.go')):
            return True
        return False

    # -------------------------------------------------------------------------
    # FILE PARSING
    # -------------------------------------------------------------------------

    def _parse_file(self, from_svc: str, fpath: Path,
                    lang: str, patterns: dict):
        try:
            content = fpath.read_text(errors='ignore')
        except Exception:
            return

        # Track whether we're inside a feature flag block (heuristic)
        is_conditional = any(re.search(p, content, re.IGNORECASE)
                             for p in FEATURE_FLAG_PATTERNS)

        # --- HTTP endpoint calls ---
        for pat in patterns.get('http_client', []):
            for m in re.finditer(pat, content, re.IGNORECASE):
                url = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                dep = self._url_to_dep(from_svc, url, 'http', str(fpath))
                if dep:
                    dep.conditional = is_conditional
                    self._add(dep)

        # --- WebSocket ---
        for pat in patterns.get('websocket', []):
            for m in re.finditer(pat, content, re.IGNORECASE):
                url = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                dep = self._url_to_dep(from_svc, url, 'websocket', str(fpath))
                if dep:
                    dep.conditional = is_conditional
                    self._add(dep)

        # --- gRPC ---
        grpc_found = any(re.search(p, content) for p in patterns.get('grpc', []))
        if grpc_found:
            for pat in patterns.get('grpc', []):
                for m in re.finditer(pat, content):
                    addr = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                    dep = self._url_to_dep(from_svc, addr, 'grpc', str(fpath))
                    if dep:
                        dep.conditional = is_conditional
                        self._add(dep)

        # --- GraphQL ---
        for pat in patterns.get('graphql', []):
            for m in re.finditer(pat, content, re.IGNORECASE):
                url = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                dep = self._url_to_dep(from_svc, url, 'graphql', str(fpath))
                if dep:
                    dep.conditional = is_conditional
                    self._add(dep)

        # --- Kafka ---
        kafka_topic_patterns = patterns.get('kafka_topic_str', [])
        for pat in kafka_topic_patterns:
            for m in re.finditer(pat, content, re.IGNORECASE):
                topic = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                if topic and len(topic) < 80:
                    self._record_string(topic, from_svc)
                    self._add(Dependency(
                        from_service=from_svc,
                        to_service='kafka',
                        dep_type='async',
                        subtype='kafka',
                        topic_or_table=topic,
                        confidence=0.80,
                        evidence=f'kafka topic: {topic} in {fpath.name}',
                        source='code_kafka'
                    ))

        # RabbitMQ queue/exchange strings
        for pat in patterns.get('amqp_publish', []) + patterns.get('amqp_consume', []):
            for m in re.finditer(pat, content, re.IGNORECASE):
                queue = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                if queue:
                    self._record_string(queue, from_svc)
                    self._add(Dependency(
                        from_service=from_svc,
                        to_service='rabbitmq',
                        dep_type='async',
                        subtype='rabbitmq',
                        topic_or_table=queue,
                        confidence=0.80,
                        evidence=f'rabbitmq queue: {queue} in {fpath.name}',
                        source='code_amqp'
                    ))

        # --- Connection strings ---
        for pat, infra_type in CONN_PATTERNS:
            for m in re.finditer(pat, content, re.IGNORECASE):
                host = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                # Try to match to a known service
                host_clean = host.split('/')[0].split('@')[-1].split(':')[0]
                to_svc = self._match(host_clean) or infra_type
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=to_svc,
                    dep_type='infrastructure',
                    subtype=infra_type,
                    confidence=0.85,
                    evidence=f'conn string: {pat[:40]} in {fpath.name}',
                    source='code_conn_string'
                ))

        # --- External SDKs ---
        for sdk_pat, (provider, dep_type, subtype) in EXTERNAL_SDKS.items():
            if re.search(sdk_pat, content, re.IGNORECASE):
                self._add(Dependency(
                    from_service=from_svc,
                    to_service=provider,
                    dep_type=dep_type,
                    subtype=subtype,
                    confidence=0.85,
                    evidence=f'SDK import: {sdk_pat} in {fpath.name}',
                    source='code_sdk'
                ))

        # --- SQL table names (for shared data detection) ---
        for m in re.finditer(r'(?:FROM|INTO|UPDATE|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]+)',
                             content, re.IGNORECASE):
            table = m.group(1).lower()
            if len(table) > 2:
                self._record_string(table, from_svc)

        # --- Env var reads (for shared env pattern analysis) ---
        for pat in patterns.get('env_read', []):
            for m in re.finditer(pat, content, re.IGNORECASE):
                var_name = m.group(1) if m.lastindex and m.lastindex >= 1 else ''
                if var_name:
                    # Check if this env var references a service
                    for svc_name in self.services:
                        svc_norm = svc_name.lower().replace('-', '_')
                        if svc_norm in var_name.lower():
                            dep = self._env_var_to_dep(from_svc, var_name, str(fpath))
                            if dep:
                                dep.conditional = is_conditional
                                self._add(dep)
                            break

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _url_to_dep(self, from_svc: str, url: str,
                    subtype: str, source_file: str) -> Optional[Dependency]:
        """Convert a URL string to a dependency if target matches a known service."""
        if not url:
            return None

        # Normalize URL — remove scheme, credentials, port
        clean = re.sub(r'^https?://', '', url)
        clean = re.sub(r'^grpc://', '', clean)
        clean = clean.split('/')[0]    # remove path
        clean = clean.split('@')[-1]   # remove creds
        host = clean.split(':')[0]     # remove port

        # Try to resolve to known service
        to_svc = self._match(host)
        if not to_svc:
            # Also try with env var substitution removed
            host2 = re.sub(r'\$\{?[A-Z_]+\}?', '', host).strip('_-. ')
            to_svc = self._match(host2) if host2 else None

        if not to_svc:
            return None

        if to_svc == from_svc:
            return None

        return Dependency(
            from_service=from_svc,
            to_service=to_svc,
            dep_type='endpoint',
            subtype=subtype,
            confidence=0.85,
            evidence=f'{subtype} call to {url[:80]} in {Path(source_file).name}',
            source=f'code_{subtype}'
        )

    def _env_var_to_dep(self, from_svc: str, var_name: str,
                        source_file: str) -> Optional[Dependency]:
        """Try to resolve env var to a service dependency.

        Confidence is elevated when the env var key ends with a service-address
        suffix, because these are explicitly declared service endpoints rather
        than ambiguous config reads.  No service names are hardcoded — only the
        well-known suffix conventions are matched.
        """
        # Suffixes that strongly suggest the value is a peer service address.
        # Matched against the upper-cased key so casing doesn't matter.
        SERVICE_ADDR_SUFFIXES = (
            '_SERVICE_ADDR', '_ADDR', '_SERVICE_HOST', '_HOST',
            '_ENDPOINT', '_URL',
        )
        var_upper = var_name.upper()
        is_service_addr_var = any(var_upper.endswith(sfx)
                                  for sfx in SERVICE_ADDR_SUFFIXES)
        # Service-address vars are high-confidence (developer explicitly declared
        # the target). Generic env reads stay at the base 0.68.
        confidence = 0.85 if is_service_addr_var else 0.68
        subtype = 'grpc' if var_upper.endswith('_ADDR') else 'http'

        for svc_name in self.services:
            svc_norm = svc_name.upper().replace('-', '_').replace('.', '_')
            if svc_norm in var_upper and svc_name != from_svc:
                return Dependency(
                    from_service=from_svc,
                    to_service=svc_name,
                    dep_type='endpoint',
                    subtype=subtype,
                    confidence=confidence,
                    evidence=f'env var {var_name} in {Path(source_file).name}',
                    source='code_env_var'
                )
        return None

    def _match(self, name: str) -> Optional[str]:
        norm = name.lower().replace('-', '').replace('_', '').replace('.', '')
        for svc_name, svc in self.services.items():
            if svc.normalized_name == norm:
                return svc_name
        return None

    def _add(self, dep: Dependency):
        if dep and dep.from_service and dep.to_service and dep.from_service != dep.to_service:
            self.deps.append(dep)

    def _record_string(self, s: str, svc_name: str):
        """Record string for cross-service co-occurrence analysis."""
        key = s.strip().lower()
        if key not in self._seen_strings:
            self._seen_strings[key] = set()
        self._seen_strings[key].add(svc_name)
