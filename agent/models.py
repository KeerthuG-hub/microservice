"""
Data models - NO hardcoded service names, ports, or patterns.
Everything is discovered dynamically from the project.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any


@dataclass
class ProjectContext:
    """User-provided + auto-detected context"""

    # USER PROVIDED (3 questions)
    service_discovery: str   # env_vars | config_files | service_discovery | hardcoded | unknown
    async_messaging: str     # no | kafka | rabbitmq | sqs | nats | other | unknown
    custom_notes: Optional[str] = None

    # AUTO-DETECTED
    languages: Set[str] = field(default_factory=set)
    has_kubernetes: bool = False
    has_docker_compose: bool = False
    has_grpc: bool = False
    has_git: bool = False
    has_service_mesh: bool = False   # Istio/Linkerd detected
    project_root: str = ""
    layout: str = "unknown"          # flat | nested | monorepo


@dataclass
class Service:
    """Discovered service - name discovered FROM project, never hardcoded"""
    name: str                        # Discovered name
    path: str                        # Absolute path to service directory
    language: str                    # Detected from files
    anchor: str                      # What proved this is a service
    ports: List[int] = field(default_factory=list)
    normalized_name: str = ""        # For fuzzy matching
    is_infrastructure: bool = False  # True if no owned code + no Deployment + no code calls


@dataclass
class Dependency:
    """Detected dependency edge - all nine types"""
    from_service: str
    to_service: str
    dep_type: str     # endpoint | async | data | semantic | build | deployment | infrastructure | observability | external
    subtype: Optional[str] = None    # http | grpc | graphql | websocket | kafka | rabbitmq | sqs | nats |
                                     # shared_db | shared_table | shared_redis | shared_dto | logic_clone |
                                     # shared_domain | change_coupling | proto_import | internal_lib |
                                     # base_image | generated | startup_order | helm_hook | db_migration |
                                     # iac | postgres | mysql | mongodb | redis | elasticsearch | s3 |
                                     # vault | jaeger | prometheus | fluentd | service_mesh |
                                     # stripe | sendgrid | auth0 | aws | gcp | azure | other_external
    confidence: float = 0.0
    evidence: str = ""
    source: str = ""                 # Which parser found it
    sources: List[str] = field(default_factory=list)   # All sources (after merge)
    topic_or_table: Optional[str] = None  # For async / data deps
    conditional: bool = False        # Inside feature flag?

    def __hash__(self):
        return hash((self.from_service, self.to_service, self.dep_type, self.subtype))

    def __eq__(self, other):
        return (self.from_service == other.from_service and
                self.to_service == other.to_service and
                self.dep_type == other.dep_type and
                self.subtype == other.subtype)


@dataclass
class AnalysisResult:
    """Final output"""
    services: List[Service]
    dependencies: List[Dependency]

    total_services: int = 0
    total_dependencies: int = 0
    infrastructure_services: int = 0  # Services classified as pure infrastructure
    confirmed_deps: int = 0        # confidence >= 0.85
    probable_deps: int = 0         # 0.65 <= confidence < 0.85
    uncertain_deps: int = 0        # 0.40 <= confidence < 0.65
    implicit_deps: int = 0         # string_cooccurrence / git_change_coupling
    observability_deps: int = 0
    env_only_deps: int = 0         # Deps backed only by env var sources, no code confirmation

    # Learned conventions (NOT hardcoded)
    learned_port_map: Dict[int, str] = field(default_factory=dict)
    learned_env_patterns: Set[str] = field(default_factory=set)

    # Per-service breakdown: {svc_name: {out_count, in_count, types, max_depth, transitive_count}}
    service_stats: Dict[str, Any] = field(default_factory=dict)
    # Depth analysis: {svc_name: {max_depth, transitive_deps}}
    depth_stats: Dict[str, Any] = field(default_factory=dict)

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
