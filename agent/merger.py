"""
Merger - deduplicates dependencies and fuses confidence from multiple sources.
Boosts confidence when multiple independent sources agree.
"""

from typing import List, Dict, Tuple
from agent.models import Dependency, Service

# Base confidence by source (used if dep.confidence is 0)
SOURCE_BASE_CONFIDENCE = {
    'proto_parser': 0.95,
    'k8s_env': 0.90,
    'k8s_env_svcname': 0.82,
    'k8s_port': 0.72,
    'k8s_envfrom': 0.72,
    'k8s_init_container': 0.85,
    'k8s_ingress': 0.90,
    'k8s_image': 0.75,
    'k8s_job': 0.80,
    'k8s_servicemonitor': 0.85,
    'k8s_virtualservice': 0.78,
    'k8s_env_broker': 0.85,
    'compose_depends_on': 0.88,
    'compose_links': 0.80,
    'compose_env': 0.82,
    'compose_port': 0.72,
    'build_file': 0.85,
    'build_go_mod': 0.75,
    'build_pom': 0.75,
    'build_requirements': 0.80,
    'build_dockerfile': 0.82,
    'config_conn_string': 0.88,
    'config_kafka': 0.88,
    'config_yaml': 0.80,
    'config_url': 0.80,
    'config_env_kv': 0.80,
    'code_http': 0.85,
    'code_grpc': 0.85,
    'code_graphql': 0.80,
    'code_websocket': 0.80,
    'code_kafka': 0.80,
    'code_amqp': 0.80,
    'code_conn_string': 0.85,
    'code_sdk': 0.85,
    'code_env_var': 0.68,
    'llm_analysis': 0.70,
    'string_cooccurrence': 0.50,
    'git_change_coupling': 0.55,
    'proto_service_def': 0.88,
}

# Sources that come from env var / config declarations only (our parser labels).
# NOTE: k8s_env and k8s_env_svcname are intentionally excluded here — K8s
# manifests are authoritative runtime configuration and should never be
# treated the same as ambiguous in-code env reads.
ENV_ONLY_SOURCES = {
    'k8s_port', 'k8s_envfrom',
    'compose_env', 'config_env_kv', 'code_env_var'
}

# Sources that confirm an actual runtime code call exists (our parser labels)
CODE_SOURCES = {
    'code_http', 'code_grpc', 'code_graphql', 'code_websocket',
    'code_kafka', 'code_amqp', 'code_conn_string', 'code_sdk',
    'proto_parser', 'proto_service_def'
}


class Merger:

    def __init__(self, services: Dict[str, Service]):
        self.services = services

    def merge(self, all_deps: List[Dependency]) -> List[Dependency]:
        print("\n🔀 Merging and deduplicating dependencies...")

        # Group by (from, to, type, subtype)
        groups: Dict[Tuple, List[Dependency]] = {}

        for dep in all_deps:
            if not dep.from_service or not dep.to_service:
                continue
            if dep.from_service == dep.to_service:
                continue

            key = (dep.from_service, dep.to_service, dep.dep_type,
                   dep.subtype or '')
            if key not in groups:
                groups[key] = []
            groups[key].append(dep)

        # Merge each group
        merged: List[Dependency] = []
        for key, deps in groups.items():
            merged_dep = self._merge_group(deps)
            merged.append(merged_dep)

        # Penalise deps backed only by env-var evidence (no code call confirmed)
        env_only_count = self._penalize_env_only(merged)

        print(f"   Raw: {len(all_deps)} → Merged: {len(merged)} "
              f"({env_only_count} env-only capped at 0.68)")
        return merged

    def _merge_group(self, deps: List[Dependency]) -> Dependency:
        """Merge a group of identical (from, to, type, subtype) deps."""
        if len(deps) == 1:
            d = deps[0]
            if d.confidence == 0.0:
                d.confidence = SOURCE_BASE_CONFIDENCE.get(d.source, 0.65)
            d.sources = [d.source]
            return d

        # Use highest confidence as base
        best = max(deps, key=lambda d: d.confidence or
                   SOURCE_BASE_CONFIDENCE.get(d.source, 0.65))

        # Collect unique sources
        sources = list({d.source for d in deps if d.source})

        # Boost for multi-source agreement
        base_conf = best.confidence or SOURCE_BASE_CONFIDENCE.get(best.source, 0.65)
        extra = (len(sources) - 1) * 0.05
        final_conf = min(0.98, base_conf + extra)

        # Combine evidence
        evidences = [d.evidence for d in deps if d.evidence]
        combined_evidence = ' | '.join(evidences[:3])

        # topic_or_table from first non-None
        topic = next((d.topic_or_table for d in deps if d.topic_or_table), None)

        # conditional: True only if ALL sources say conditional
        conditional = all(d.conditional for d in deps)

        return Dependency(
            from_service=best.from_service,
            to_service=best.to_service,
            dep_type=best.dep_type,
            subtype=best.subtype,
            confidence=final_conf,
            evidence=combined_evidence,
            source=sources[0],
            sources=sources,
            topic_or_table=topic,
            conditional=conditional,
        )

    def _penalize_env_only(self, merged: List[Dependency]) -> int:
        """
        Cap confidence at 0.68 for env-var-referenced deps with no code confirmation,
        BUT only where the cap is meaningful — i.e. where code confirmation is
        actually possible and the env var source is ambiguous.

        EXEMPT from penalty (dep_type-aware, no hardcoding):
          • dep_type in ('infrastructure', 'external', 'observability', 'build',
                         'deployment', 'semantic')
            → These dep types are NEVER confirmed by code calls. An 'infrastructure'
              dep to postgres is always an env-var connection string — there is no
              "gRPC call to postgres" to confirm. Penalising these is wrong.

          • dep_type == 'endpoint' AND source in K8S_ENDPOINT_SOURCES
            → K8s env vars for endpoint deps have been parsed from actual service
              address values (host:port or URL pattern that resolved to a known
              service). That resolution IS the confirmation — K8s deployment config
              is authoritative runtime configuration. The developer explicitly wrote
              it. No code confirmation needed.

        PENALISED:
          • dep_type == 'endpoint' from compose/config env sources with no code
            call seen — these are less authoritative than K8s manifests.

        Uses internal dep_type values and parser source labels — zero hardcoding.
        """
        # K8s sources where the parser validated the value → known service address
        K8S_ENDPOINT_SOURCES = {
            'k8s_env', 'k8s_env_svcname', 'k8s_port', 'k8s_envfrom',
            'k8s_init_container', 'k8s_ingress', 'k8s_env_broker'
        }
        # Dep types where env-var-only is perfectly normal and expected
        ALWAYS_ENV_BASED_TYPES = {
            'infrastructure', 'external', 'observability', 'build', 'deployment', 'semantic'
        }

        # Env var key suffixes that indicate an explicitly declared service address.
        # Matched case-insensitively against the dep evidence string.
        # No service names hardcoded — suffix conventions only.
        SERVICE_ADDR_SUFFIXES = (
            '_service_addr', '_addr', '_service_host', '_host',
            '_endpoint', '_url',
        )

        ENV_ONLY_CAP = 0.68
        penalised = 0
        for dep in merged:
            sources = set(dep.sources or [dep.source])
            has_code_source = bool(sources & CODE_SOURCES)
            is_env_only = bool(sources) and sources.issubset(ENV_ONLY_SOURCES)

            if not (is_env_only and not has_code_source):
                continue

            # Exempt: dep types where env-based evidence IS the confirmation
            if dep.dep_type in ALWAYS_ENV_BASED_TYPES:
                continue

            # Exempt: K8s-sourced endpoint deps — K8s manifests are authoritative
            # runtime config. The k8s_parser validated the value resolves to a
            # known service, which IS the confirmation.
            if dep.dep_type == 'endpoint' and bool(sources & K8S_ENDPOINT_SOURCES):
                continue

            # Exempt: code_env_var deps where the env key names a service address.
            # These were already scored at 0.85 by code_parser; capping is wrong.
            if dep.dep_type == 'endpoint' and sources == {'code_env_var'}:
                evidence_lower = (dep.evidence or '').lower()
                if any(sfx in evidence_lower for sfx in SERVICE_ADDR_SUFFIXES):
                    continue

            # At this point: endpoint dep from non-K8s env source, no code confirmation
            if dep.confidence > ENV_ONLY_CAP:
                dep.confidence = ENV_ONLY_CAP
                dep.evidence = (dep.evidence or '') + ' [env-only: no code call confirmed]'
            penalised += 1
        return penalised
