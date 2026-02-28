"""
Core Orchestrator - runs all analysis phases in order.
Phase 0: Context collection
Phase 1: Service discovery
Phase 2: Static analysis (all parsers)
Phase 3: LLM analysis (targeted)
Phase 4: Semantic analysis
Phase 5: Merge + output
"""

import time
from typing import List, Optional
from pathlib import Path

from agent.models import ProjectContext, AnalysisResult, Dependency
from agent.context_collector import ContextCollector
from agent.explorer import ServiceExplorer
from agent.merger import Merger
from agent.semantic_analyzer import SemanticAnalyzer

from agent.static_analyzer.k8s_parser import KubernetesParser
from agent.static_analyzer.code_parser import CodeParser
from agent.static_analyzer.build_parser import BuildParser
from agent.static_analyzer.proto_parser import ProtoParser
from agent.static_analyzer.compose_parser import ComposeParser
from agent.static_analyzer.config_parser import ConfigParser


class DependencyAnalyzer:
    """Main entry point for the microservice dependency analyzer."""

    def __init__(self, project_path: str, non_interactive: bool = False,
                 skip_llm: bool = False):
        self.project_path = str(Path(project_path).resolve())
        self.non_interactive = non_interactive
        self.skip_llm = skip_llm

    def analyze(self) -> AnalysisResult:
        start = time.time()
        print("\n" + "=" * 70)
        print("🚀  MICROSERVICE DEPENDENCY ANALYZER")
        print("=" * 70)
        print(f"📂  Project: {self.project_path}")

        # ---- Phase 0: Context Collection ----
        collector = ContextCollector(self.project_path, self.non_interactive)
        context = collector.collect()

        # ---- Phase 1: Service Discovery ----
        explorer = ServiceExplorer(context)
        services = explorer.discover()

        if not services:
            print("⚠️  No services found. Check project path.")
            return AnalysisResult(services=[], dependencies=[])

        # ---- Phase 2: Static Analysis ----
        all_deps: List[Dependency] = []

        # K8s
        k8s_parser = KubernetesParser(services, explorer.port_map, explorer.env_patterns)
        all_deps.extend(k8s_parser.parse(self.project_path))

        # Docker Compose
        compose_parser = ComposeParser(services, explorer.port_map)
        all_deps.extend(compose_parser.parse(self.project_path))

        # Proto files
        if context.has_grpc:
            proto_parser = ProtoParser(services)
            all_deps.extend(proto_parser.parse(self.project_path))

        # Build files
        build_parser = BuildParser(services)
        all_deps.extend(build_parser.parse(self.project_path))

        # Config files
        config_parser = ConfigParser(services)
        all_deps.extend(config_parser.parse(self.project_path))

        # Source code (returns string map for semantic analysis)
        code_parser = CodeParser(services, explorer.port_map, explorer.env_patterns)
        all_deps.extend(code_parser.parse(self.project_path))
        string_map = code_parser.get_string_map()

        # ---- Phase 3: LLM Analysis ----
        if not self.skip_llm:
            try:
                from agent.llm.llm_analyzer import LLMAnalyzer
                llm_analyzer = LLMAnalyzer(services, context, all_deps)
                all_deps.extend(llm_analyzer.analyze())
            except Exception as e:
                print(f"   ⚠️  LLM analysis skipped: {e}")

        # ---- Phase 4: Semantic Analysis ----
        semantic = SemanticAnalyzer(services, self.project_path, string_map)
        all_deps.extend(semantic.analyze())

        # ---- Phase 5: Merge + Finalize ----
        merger = Merger(services)
        merged_deps = merger.merge(all_deps)

        result = self._build_result(services, merged_deps, explorer)

        elapsed = time.time() - start
        self._print_summary(result, elapsed)

        return result

    # -------------------------------------------------------------------------
    # RESULT BUILDING
    # -------------------------------------------------------------------------

    def _build_result(self, services, merged_deps, explorer) -> AnalysisResult:
        from collections import defaultdict, deque

        infra_svcs = {n for n, s in services.items() if s.is_infrastructure}

        confirmed = [d for d in merged_deps
                     if d.confidence >= 0.85
                     and d.dep_type not in ('observability',)
                     and d.source not in ('string_cooccurrence', 'git_change_coupling')]
        probable = [d for d in merged_deps
                    if 0.65 <= d.confidence < 0.85
                    and d.dep_type not in ('observability',)
                    and d.source not in ('string_cooccurrence', 'git_change_coupling')]
        uncertain = [d for d in merged_deps
                     if 0.40 <= d.confidence < 0.65
                     and d.dep_type not in ('observability',)
                     and d.source not in ('string_cooccurrence', 'git_change_coupling')]
        implicit = [d for d in merged_deps
                    if d.source in ('string_cooccurrence', 'git_change_coupling')]
        observability = [d for d in merged_deps if d.dep_type == 'observability']

        # Env-only deps count (capped at 0.68 in merger)
        env_only = [d for d in merged_deps
                    if '[env-only: no code call confirmed]' in (d.evidence or '')]

        # --- Per-service stats ---
        svc_stats: dict = {}
        for svc_name in services:
            svc_stats[svc_name] = {
                'outgoing': 0, 'incoming': 0,
                'outgoing_types': defaultdict(int),
                'incoming_types': defaultdict(int),
                'is_infrastructure': services[svc_name].is_infrastructure,
            }
        for dep in merged_deps:
            if dep.from_service in svc_stats:
                svc_stats[dep.from_service]['outgoing'] += 1
                svc_stats[dep.from_service]['outgoing_types'][dep.dep_type] += 1
            if dep.to_service in svc_stats:
                svc_stats[dep.to_service]['incoming'] += 1
                svc_stats[dep.to_service]['incoming_types'][dep.dep_type] += 1

        # --- Depth analysis (BFS on confirmed + probable deps) ---
        adj: dict = defaultdict(set)
        for dep in confirmed + probable:
            adj[dep.from_service].add(dep.to_service)

        depth_stats: dict = {}
        for svc_name in services:
            visited: set = set()
            queue = deque([(svc_name, 0)])
            max_depth = 0
            while queue:
                node, depth = queue.popleft()
                for neighbour in adj.get(node, set()):
                    if neighbour not in visited and neighbour != svc_name:
                        visited.add(neighbour)
                        max_depth = max(max_depth, depth + 1)
                        queue.append((neighbour, depth + 1))
            depth_stats[svc_name] = {
                'max_depth': max_depth,
                'transitive_deps': len(visited),
            }
            svc_stats[svc_name]['max_depth'] = max_depth
            svc_stats[svc_name]['transitive_deps'] = len(visited)

        # Convert defaultdicts to plain dicts for serialisation
        for s in svc_stats.values():
            s['outgoing_types'] = dict(s['outgoing_types'])
            s['incoming_types'] = dict(s['incoming_types'])

        return AnalysisResult(
            services=list(services.values()),
            dependencies=merged_deps,
            total_services=len(services) - len(infra_svcs),
            infrastructure_services=len(infra_svcs),
            total_dependencies=len(merged_deps),
            confirmed_deps=len(confirmed),
            probable_deps=len(probable),
            uncertain_deps=len(uncertain),
            implicit_deps=len(implicit),
            observability_deps=len(observability),
            env_only_deps=len(env_only),
            learned_port_map=explorer.port_map,
            learned_env_patterns=explorer.env_patterns,
            service_stats=svc_stats,
            depth_stats=depth_stats,
        )

    # -------------------------------------------------------------------------
    # CONSOLE OUTPUT
    # -------------------------------------------------------------------------

    def _print_summary(self, result: AnalysisResult, elapsed: float):
        print("\n" + "=" * 70)
        print("📊  ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"⏱️   Time: {elapsed:.1f}s")
        print(f"🔧  Business Services : {result.total_services}")
        print(f"🏗️  Infrastructure    : {result.infrastructure_services}")
        print(f"🔗  Total deps        : {result.total_dependencies}")
        print(f"   ✅ Confirmed  (≥0.85): {result.confirmed_deps}")
        print(f"   🟡 Probable   (0.65-0.85): {result.probable_deps}")
        print(f"   🔴 Uncertain  (<0.65): {result.uncertain_deps}")
        print(f"   👁️  Implicit   (cooccurrence/git): {result.implicit_deps}")
        print(f"   📡 Observability: {result.observability_deps}")
        print(f"   ⚠️  Env-only (no code confirmation): {result.env_only_deps}")

        # All confirmed dependencies — split into service-to-service vs infrastructure
        infra_svc_names = {
            s.name for s in result.services if s.is_infrastructure
        }
        # Dep types that represent peer-service runtime calls
        SVC_TO_SVC_TYPES = {'endpoint', 'async', 'graphql', 'websocket'}

        confirmed = [d for d in result.dependencies
                     if d.confidence >= 0.85
                     and d.dep_type != 'observability'
                     and d.source not in ('string_cooccurrence', 'git_change_coupling')]
        confirmed.sort(key=lambda d: d.confidence, reverse=True)

        # Partition into service-to-service vs infrastructure
        svc_to_svc = [
            d for d in confirmed
            if d.dep_type in SVC_TO_SVC_TYPES and d.to_service not in infra_svc_names
        ]
        infra_confirmed = [d for d in confirmed if d not in svc_to_svc]

        if svc_to_svc:
            print(f"\n🔝  Service-to-Service ({len(svc_to_svc)} confirmed):")
            for dep in svc_to_svc:
                cond = " [conditional]" if dep.conditional else ""
                print(f"   {dep.from_service} → {dep.to_service}"
                      f"  [{dep.dep_type}/{dep.subtype or '?'}]"
                      f"  conf={dep.confidence:.2f}{cond}")

        if infra_confirmed:
            print(f"\n🏗️  Infrastructure/External ({len(infra_confirmed)} confirmed):")
            for dep in infra_confirmed:
                cond = " [conditional]" if dep.conditional else ""
                print(f"   {dep.from_service} → {dep.to_service}"
                      f"  [{dep.dep_type}/{dep.subtype or '?'}]"
                      f"  conf={dep.confidence:.2f}{cond}")

        # Per-service stats table
        if result.service_stats:
            print("\n📊  Per-service dependency breakdown:")
            print(f"   {'Service':<35} {'Out':>5} {'In':>5} {'MaxDepth':>9} {'Transitive':>10}  InfraFlag")
            print("   " + "-" * 75)
            for svc_name, stats in sorted(result.service_stats.items()):
                flag = " [INFRA]" if stats.get('is_infrastructure') else ""
                print(f"   {svc_name:<35} "
                      f"{stats['outgoing']:>5} "
                      f"{stats['incoming']:>5} "
                      f"{stats.get('max_depth', 0):>9} "
                      f"{stats.get('transitive_deps', 0):>10}"
                      f"{flag}")

        # Infrastructure nodes (not counted as services)
        infra = [s for s in result.services if s.is_infrastructure]
        if infra:
            print(f"\n🏗️  Infrastructure nodes (not counted as services):")
            for s in sorted(infra, key=lambda x: x.name):
                print(f"   • {s.name}  [{s.language}]  anchor={s.anchor}")

        print("=" * 70)
