"""
Context Collector - asks ONLY 3 questions that cannot be auto-detected.
Auto-detects everything possible from the filesystem.
Zero hardcoded assumptions.
"""

from pathlib import Path
from typing import Optional
from agent.models import ProjectContext


class ContextCollector:
    """Collect minimal user input + auto-detect everything else"""

    def __init__(self, project_path: str, non_interactive: bool = False):
        self.project_path = Path(project_path)
        self.non_interactive = non_interactive   # For testing / CI

    def collect(self) -> ProjectContext:
        context = self._auto_detect()
        if not self.non_interactive:
            context = self._ask_user(context)
        return context

    # -------------------------------------------------------------------------
    # AUTO-DETECTION (zero hardcoding)
    # -------------------------------------------------------------------------

    def _auto_detect(self) -> ProjectContext:
        context = ProjectContext(
            service_discovery="unknown",
            async_messaging="unknown",
            project_root=str(self.project_path)
        )

        # --- Languages ---
        ext_lang = {
            '.go': 'go', '.py': 'python', '.java': 'java',
            '.js': 'javascript', '.ts': 'typescript', '.rs': 'rust',
            '.rb': 'ruby', '.php': 'php', '.cs': 'csharp',
            '.cpp': 'cpp', '.c': 'c', '.scala': 'scala', '.kt': 'kotlin'
        }
        for ext, lang in ext_lang.items():
            hits = list(self.project_path.rglob(f'*{ext}'))
            hits = [h for h in hits if not self._is_vendor(h)]
            if hits:
                context.languages.add(lang)

        # --- Kubernetes ---
        yaml_files = list(self.project_path.rglob('*.yaml')) + \
                     list(self.project_path.rglob('*.yml'))
        for f in yaml_files:
            if self._is_vendor(f):
                continue
            try:
                content = f.read_text(errors='ignore')
                if 'kind:' in content and 'apiVersion:' in content:
                    context.has_kubernetes = True
                if 'kind: VirtualService' in content or 'istio-injection' in content:
                    context.has_service_mesh = True
            except Exception:
                pass

        # --- Docker Compose ---
        context.has_docker_compose = any([
            (self.project_path / 'docker-compose.yml').exists(),
            (self.project_path / 'docker-compose.yaml').exists(),
            (self.project_path / 'docker-compose.override.yml').exists(),
        ])

        # --- gRPC ---
        proto_files = [p for p in self.project_path.rglob('*.proto')
                       if not self._is_vendor(p)]
        context.has_grpc = len(proto_files) > 0

        # --- Git ---
        context.has_git = (self.project_path / '.git').exists()

        # --- Layout detection ---
        context.layout = self._detect_layout()

        return context

    def _detect_layout(self) -> str:
        """
        flat:     services sit directly at project root
        nested:   services inside a src/ or services/ dir
        monorepo: multiple top-level dirs each with their own anchor files
        """
        root = self.project_path
        top_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        anchor_names = {
            'Dockerfile', 'go.mod', 'package.json', 'pom.xml',
            'requirements.txt', 'Cargo.toml', 'Gemfile', 'composer.json'
        }
        # count top-level dirs that look like services
        service_dirs = 0
        for d in top_dirs:
            for anchor in anchor_names:
                if (d / anchor).exists():
                    service_dirs += 1
                    break

        if service_dirs == 0:
            return 'nested'
        elif service_dirs / max(len(top_dirs), 1) >= 0.3:
            return 'flat'
        return 'monorepo'

    # -------------------------------------------------------------------------
    # USER QUESTIONS
    # -------------------------------------------------------------------------

    def _ask_user(self, context: ProjectContext) -> ProjectContext:
        print("\n" + "=" * 70)
        print("📋  MICROSERVICE DEPENDENCY ANALYZER")
        print("=" * 70)

        print(f"\n✅ Auto-detected:")
        print(f"   📂 Project : {context.project_root}")
        print(f"   💻 Languages: {', '.join(sorted(context.languages)) or 'none'}")
        print(f"   ☸️  K8s     : {'Yes' if context.has_kubernetes else 'No'}")
        print(f"   🐳 Compose  : {'Yes' if context.has_docker_compose else 'No'}")
        print(f"   🔌 gRPC     : {'Yes' if context.has_grpc else 'No'}")
        print(f"   🕸️  Mesh     : {'Yes' if context.has_service_mesh else 'No'}")
        print(f"   📊 Git      : {'Yes' if context.has_git else 'No'}")
        print(f"   📁 Layout   : {context.layout}")

        print("\n" + "-" * 70)
        print("📝  3 questions (press Enter to skip = 'unknown'):")
        print("-" * 70)

        # Q1 — service discovery
        print("\n1️⃣  How do services find each other's addresses?")
        print("   a) Environment variables  b) Config files")
        print("   c) Service-discovery platform (Consul/Eureka)")
        print("   d) Hardcoded  e) Unknown")
        c = input("   Choice [a/b/c/d/e]: ").strip().lower()
        context.service_discovery = {
            'a': 'env_vars', 'b': 'config_files',
            'c': 'service_discovery', 'd': 'hardcoded'
        }.get(c, 'unknown')

        # Q2 — async messaging
        print("\n2️⃣  Async message queues used?")
        print("   a) None  b) Kafka  c) RabbitMQ  d) AWS SQS")
        print("   e) NATS  f) Other  g) Unknown")
        c = input("   Choice [a-g]: ").strip().lower()
        context.async_messaging = {
            'a': 'no', 'b': 'kafka', 'c': 'rabbitmq',
            'd': 'sqs', 'e': 'nats', 'f': 'other'
        }.get(c, 'unknown')

        # Q3 — custom notes
        print("\n3️⃣  Custom frameworks / non-standard setup? (Enter to skip)")
        notes = input("   Notes: ").strip()
        context.custom_notes = notes if notes else None

        print("\n✅ Context collection complete!\n")
        return context

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_vendor(path: Path) -> bool:
        skip = {'node_modules', 'vendor', '.git', 'build', 'dist',
                'target', '__pycache__', '.venv', 'venv', 'env',
                '.idea', '.vscode', 'bin', 'obj', 'generated', '.tox'}
        return any(p in skip for p in path.parts)
