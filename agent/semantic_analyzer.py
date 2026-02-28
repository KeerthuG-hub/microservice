"""
Semantic Analyzer - finds implicit coupling through:
1. Shared string co-occurrence (Kafka topics, table names, constants)
2. Git change coupling (services that always change together)
Zero LLM calls. Zero hardcoded patterns.
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple
import re
from collections import defaultdict
from agent.models import Dependency, Service

# Noise strings to exclude from co-occurrence
NOISE_STRINGS = {
    'true', 'false', 'null', 'none', 'nil', 'undefined', 'error', 'success',
    'ok', 'failed', 'pending', 'active', 'inactive', 'enabled', 'disabled',
    'start', 'stop', 'get', 'set', 'post', 'put', 'delete', 'http', 'https',
    'grpc', 'json', 'xml', 'utf8', 'utf-8', 'content-type', 'application/json',
    'localhost', '127.0.0.1', '0.0.0.0', 'port', 'host', 'url', 'path',
    'name', 'type', 'value', 'data', 'id', 'key', 'message', 'status',
    'created', 'updated', 'deleted', 'timestamp', 'version', 'env', 'app',
    'service', 'server', 'client', 'request', 'response', 'handler', 'manager',
    'controller', 'repository', 'interface', 'class', 'struct', 'object',
}

# Code extensions for string extraction
CODE_EXTS = {'.go', '.py', '.java', '.js', '.ts', '.rb', '.rs', '.cs', '.php'}

# Skip dirs
SKIP_DIRS = {'node_modules', 'vendor', '.git', 'build', 'dist', 'target',
             '__pycache__', '.venv', 'venv', 'generated', 'pb', 'stubs'}


class SemanticAnalyzer:

    def __init__(self, services: Dict[str, Service],
                 project_path: str,
                 existing_string_map: Dict[str, Set[str]] = None):
        self.services = services
        self.project_path = Path(project_path)
        self.deps: List[Dependency] = []
        # Can be pre-populated from CodeParser
        self.string_map: Dict[str, Set[str]] = existing_string_map or {}

    # -------------------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------------------

    def analyze(self) -> List[Dependency]:
        print("\n🔗 Running semantic analysis...")
        self._extract_strings()
        self._find_string_cooccurrences()
        self._find_git_coupling()
        print(f"   ✅ {len(self.deps)} implicit/semantic deps")
        return self.deps

    # -------------------------------------------------------------------------
    # STRING CO-OCCURRENCE
    # -------------------------------------------------------------------------

    def _extract_strings(self):
        """Extract all interesting string literals from every service."""
        for svc_name, svc in self.services.items():
            svc_path = Path(svc.path)
            for ext in CODE_EXTS:
                for fpath in svc_path.rglob(f'*{ext}'):
                    if self._is_vendor(fpath):
                        continue
                    if self._should_skip_file(fpath):
                        continue
                    self._extract_from_file(fpath, svc_name)

    def _extract_from_file(self, fpath: Path, svc_name: str):
        try:
            content = fpath.read_text(errors='ignore')
        except Exception:
            return

        # Extract quoted strings between 3 and 60 chars, no spaces
        patterns = [
            r'"([A-Za-z0-9_\-\.]{3,60})"',
            r"'([A-Za-z0-9_\-\.]{3,60})'",
            r'`([A-Za-z0-9_\-\.]{3,60})`',
        ]
        for pat in patterns:
            for m in re.finditer(pat, content):
                s = m.group(1).lower()
                if self._is_interesting(s):
                    if s not in self.string_map:
                        self.string_map[s] = set()
                    self.string_map[s].add(svc_name)

    def _is_interesting(self, s: str) -> bool:
        """Filter out noise — keep domain-specific strings."""
        if s in NOISE_STRINGS:
            return False
        if len(s) < 4 or len(s) > 60:
            return False
        if re.match(r'^[\d.]+$', s):  # pure numbers
            return False
        if re.match(r'^[a-f0-9]{8,}$', s):  # hex hashes
            return False
        # Must have at least one letter
        if not re.search(r'[a-zA-Z]', s):
            return False
        # Skip k8s namespace prefixes etc.
        if s in {svc.normalized_name for svc in self.services.values()}:
            return False
        return True

    def _find_string_cooccurrences(self):
        """Any string appearing in 2+ services = implicit coupling."""
        for s, svcs in self.string_map.items():
            if len(svcs) < 2 or (len(svcs) == 2 and len(s) < 15):
                continue
            svcs_list = sorted(svcs)
            # Create bidirectional dependencies between all pairs
            for i, svc_a in enumerate(svcs_list):
                for svc_b in svcs_list[i + 1:]:
                    self.deps.append(Dependency(
                        from_service=svc_a,
                        to_service=svc_b,
                        dep_type='semantic',
                        subtype='shared_domain',
                        confidence=0.50,
                        evidence=f'shared string: "{s}"',
                        source='string_cooccurrence',
                        topic_or_table=s
                    ))

    # -------------------------------------------------------------------------
    # GIT CHANGE COUPLING
    # -------------------------------------------------------------------------

    def _find_git_coupling(self):
        """Find services that change together frequently in git history."""
        if not (self.project_path / '.git').exists():
            return

        try:
            import git
            repo = git.Repo(str(self.project_path))
        except Exception:
            print("   ⚠️  git not available, skipping change coupling")
            return

        # Map service name → set of service paths (relative to project root)
        svc_paths: Dict[str, Set[str]] = {}
        for svc_name, svc in self.services.items():
            try:
                rel = str(Path(svc.path).relative_to(self.project_path))
                svc_paths[svc_name] = rel
            except ValueError:
                pass

        if len(svc_paths) < 2:
            return

        # Count co-changes
        change_counts: Dict[str, int] = defaultdict(int)   # service → total changes
        co_changes: Dict[Tuple[str, str], int] = defaultdict(int)

        commits = list(repo.iter_commits('HEAD', max_count=200))

        for commit in commits:
            if not commit.parents:
                continue
            try:
                changed_files = {item.a_path for item in commit.diff(commit.parents[0])}
                changed_files |= {item.b_path for item in commit.diff(commit.parents[0])}
            except Exception:
                continue

            # Determine which services changed in this commit
            changed_svcs = set()
            for svc_name, svc_rel in svc_paths.items():
                if any(f.startswith(svc_rel) for f in changed_files):
                    changed_svcs.add(svc_name)

            for svc in changed_svcs:
                change_counts[svc] += 1

            changed_list = sorted(changed_svcs)
            for i, svc_a in enumerate(changed_list):
                for svc_b in changed_list[i + 1:]:
                    key = (min(svc_a, svc_b), max(svc_a, svc_b))
                    co_changes[key] += 1

        # Compute coupling ratio
        for (svc_a, svc_b), co_count in co_changes.items():
            if co_count < 5:   # was 3 — require more evidence
                continue
            min_individual = min(change_counts.get(svc_a, 1),
                                  change_counts.get(svc_b, 1))
            if min_individual < 5:  # too sparse — fewer than 5 changes each
                continue
            ratio = co_count / min_individual
            if ratio >= 0.50:  # was 0.30 — must co-change in ≥50% of either service's changes
                conf = min(0.80, 0.50 + ratio * 0.30)  # cap at 0.80 (was 0.85)
                self.deps.append(Dependency(
                    from_service=svc_a,
                    to_service=svc_b,
                    dep_type='semantic',
                    subtype='change_coupling',
                    confidence=conf,
                    evidence=f'co-changed {co_count}x / ratio={ratio:.2f}',
                    source='git_change_coupling'
                ))

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _should_skip_file(self, path: Path) -> bool:
        name = path.name
        skip_suffixes = {'_test.go', '_test.py', '.test.js', '.test.ts',
                         '.spec.js', '.spec.ts', '.pb.go', '_pb2.py'}
        return any(name.endswith(s) for s in skip_suffixes)

    @staticmethod
    def _is_vendor(path: Path) -> bool:
        return any(p in SKIP_DIRS for p in path.parts)
