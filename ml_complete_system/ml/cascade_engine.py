"""
cascade_engine.py  —  Graph-Based Cascade Failure Simulator
============================================================
NEW FILE — place in ml/cascade_engine.py

Replaces ML-predicted affected_count with deterministic BFS graph traversal.

Noise filtering decisions:
- Edges with confidence < 0.4 are ignored (too weak/inferred to trust)
- Nodes with circuit_breaker=True AND health_score > 0.85 absorb failures
- Async/queue edges propagate at 40% (not 90% like sync HTTP)
- Already-visited nodes are never re-processed (handles cycles)
"""

from __future__ import annotations

import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_EDGE_CONFIDENCE   = 0.4   # edges below this are noise — skip
MIN_PROPAGATION_PROB  = 0.2   # propagation below this — stop cascade here
DEGRADED_ERROR_RATE   = 0.05  # node already degraded — cannot absorb
HEALTHY_ABSORB_THRESHOLD = 0.85  # health_score above this + circuit_breaker = absorbs
MAX_CASCADE_DEPTH     = 20    # safety cap for circular graphs


# ── Data structures ───────────────────────────────────────────────────────────

class FailureMode(str, Enum):
    FULL_OUTAGE = "full_outage"
    DEGRADED    = "degraded"
    TIMEOUT     = "timeout"
    ABSORBED    = "absorbed"


@dataclass
class CascadeNode:
    service_name:      str
    failure_mode:      FailureMode
    depth:             int
    propagated_from:   Optional[str]
    propagation_prob:  float
    reason:            str


@dataclass
class CascadeResult:
    origin_service: str
    cascade_chain:  List[CascadeNode] = field(default_factory=list)
    absorbed_services: List[str]      = field(default_factory=list)

    @property
    def affected_services(self) -> List[str]:
        return [n.service_name for n in self.cascade_chain
                if n.failure_mode != FailureMode.ABSORBED]

    @property
    def affected_count(self) -> int:
        return len(self.affected_services)

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.cascade_chain), default=0)

    def cascade_by_depth(self) -> Dict[int, List[CascadeNode]]:
        result: Dict[int, List[CascadeNode]] = {}
        for node in self.cascade_chain:
            result.setdefault(node.depth, []).append(node)
        return dict(sorted(result.items()))

    def to_dict(self) -> dict:
        return {
            "origin_service":    self.origin_service,
            "affected_count":    self.affected_count,
            "affected_services": self.affected_services,
            "absorbed_services": self.absorbed_services,
            "max_cascade_depth": self.max_depth,
            "cascade_waves": {
                depth: [
                    {
                        "service":                 n.service_name,
                        "failure_mode":            n.failure_mode.value,
                        "propagated_from":         n.propagated_from,
                        "propagation_probability": round(n.propagation_prob, 2),
                        "reason":                  n.reason,
                    }
                    for n in nodes
                ]
                for depth, nodes in self.cascade_by_depth().items()
            },
        }


# ── Core engine ───────────────────────────────────────────────────────────────

class CascadeEngine:
    """
    Deterministic BFS cascade failure simulator.

    Usage:
        engine = CascadeEngine(G)
        result = engine.simulate("payment-service", incident_probability=0.8)
        print(result.to_dict())
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G

    def simulate(
        self,
        failed_service: str,
        failure_mode: FailureMode = FailureMode.FULL_OUTAGE,
        incident_probability: float = 1.0,
    ) -> CascadeResult:
        """
        Simulate cascade from a single origin service.

        Args:
            failed_service:       Service that failed.
            failure_mode:         How the origin failed.
            incident_probability: ML P(incident) — scales propagation probability.
                                  Low ML confidence → more conservative cascade.
        """
        if failed_service not in self.G:
            raise ValueError(f"Service '{failed_service}' not found in graph.")

        result  = CascadeResult(origin_service=failed_service)
        visited: Set[str] = {failed_service}

        # BFS queue: (service, depth, parent, incoming_prob)
        queue: List[tuple] = [(failed_service, 0, None, 1.0)]

        while queue:
            current, depth, parent, parent_prob = queue.pop(0)
            if depth > MAX_CASCADE_DEPTH:
                continue

            current_mode = (
                failure_mode if current == failed_service
                else self._failure_mode_from_prob(parent_prob)
            )

            # Check absorption (circuit breaker catches the failure)
            if current != failed_service and self._absorbs(current, parent_prob):
                result.absorbed_services.append(current)
                result.cascade_chain.append(CascadeNode(
                    service_name=current,
                    failure_mode=FailureMode.ABSORBED,
                    depth=depth,
                    propagated_from=parent,
                    propagation_prob=parent_prob,
                    reason=f"Circuit breaker / high health absorbed failure from '{parent}'",
                ))
                # Do NOT propagate further through this node
                continue

            # Record as failed
            if current != failed_service:
                result.cascade_chain.append(CascadeNode(
                    service_name=current,
                    failure_mode=current_mode,
                    depth=depth,
                    propagated_from=parent,
                    propagation_prob=parent_prob,
                    reason=self._reason(current, parent, parent_prob, current_mode),
                ))

            # Propagate downstream
            for downstream in self.G.successors(current):
                if downstream in visited:
                    continue
                edge = self.G.edges[current, downstream]

                # Skip low-confidence edges (noise)
                if edge.get("confidence", 1.0) < MIN_EDGE_CONFIDENCE:
                    continue

                prob = self._propagation_prob(edge, current_mode, incident_probability)
                if prob < MIN_PROPAGATION_PROB:
                    continue

                visited.add(downstream)
                queue.append((downstream, depth + 1, current, prob))

        return result

    def compute_blast_radius(self, service: str) -> dict:
        """Quick summary without full simulation — useful for ranking."""
        if service not in self.G:
            return {"error": f"{service} not in graph"}
        direct     = [s for s in self.G.successors(service)
                      if self.G.edges[service, s].get("confidence", 1.0) >= MIN_EDGE_CONFIDENCE]
        transitive = list(nx.descendants(self.G, service))
        return {
            "service":               service,
            "direct_dependents":     direct,
            "direct_count":          len(direct),
            "transitive_dependents": [s for s in transitive if s not in direct],
            "transitive_count":      len(transitive),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _propagation_prob(self, edge: dict, source_mode: FailureMode,
                          incident_prob: float) -> float:
        base = edge.get("propagation_weight", None)
        if base is None:
            t = edge.get("type", "http").lower()
            if t in ("http", "grpc", "rpc", "sync"):
                base = 0.9
            elif t in ("kafka", "rabbitmq", "pubsub", "async", "queue"):
                base = 0.4
            elif t in ("db", "database", "postgres", "mysql", "redis"):
                base = 0.95
            else:
                base = 0.7

        if source_mode == FailureMode.DEGRADED:
            base *= 0.6
        elif source_mode == FailureMode.TIMEOUT:
            base *= 0.4

        base *= max(0.3, incident_prob)
        return round(min(1.0, base), 3)

    def _absorbs(self, service: str, incoming_prob: float) -> bool:
        node       = self.G.nodes.get(service, {})
        health     = node.get("health_score", 0.5)
        has_cb     = node.get("circuit_breaker", False)
        error_rate = node.get("error_rate", 0.01)
        if error_rate > DEGRADED_ERROR_RATE:
            return False
        return health >= HEALTHY_ABSORB_THRESHOLD and has_cb

    def _failure_mode_from_prob(self, prob: float) -> FailureMode:
        if prob >= 0.75:
            return FailureMode.FULL_OUTAGE
        elif prob >= 0.45:
            return FailureMode.DEGRADED
        return FailureMode.TIMEOUT

    def _reason(self, service: str, parent: Optional[str],
                prob: float, mode: FailureMode) -> str:
        if parent is None:
            return "Origin of failure"
        edge_type = self.G.edges.get((parent, service), {}).get("type", "unknown")
        if mode == FailureMode.FULL_OUTAGE:
            return (f"Full outage from '{parent}' via {edge_type} "
                    f"(propagation: {prob:.0%})")
        elif mode == FailureMode.DEGRADED:
            return (f"Partial failure from '{parent}' via {edge_type} "
                    f"(propagation: {prob:.0%})")
        return (f"Timeout cascade from '{parent}' via {edge_type} "
                f"(propagation: {prob:.0%})")


# ── Convenience function ──────────────────────────────────────────────────────

def compute_blast_radius(
    G: nx.DiGraph,
    failed_service: str,
    incident_probability: float = 1.0,
) -> CascadeResult:
    """
    Top-level function — drop-in for predictor.py.

    Args:
        G:                    Full dependency graph.
        failed_service:       Service that failed.
        incident_probability: ML P(incident) — scales propagation.

    Returns:
        CascadeResult with exact affected services and cascade waves.
    """
    return CascadeEngine(G).simulate(
        failed_service=failed_service,
        incident_probability=incident_probability,
    )