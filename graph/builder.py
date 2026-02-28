"""
Graph Builder - constructs NetworkX graph from analysis results.
"""

from typing import List, Dict
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from agent.models import AnalysisResult, Dependency


# Colors per dependency type
DEP_COLORS = {
    'endpoint': '#3498DB',       # Blue
    'async': '#E67E22',          # Orange
    'data': '#E74C3C',           # Red
    'semantic': '#9B59B6',       # Purple
    'build': '#F39C12',          # Yellow
    'deployment': '#1ABC9C',     # Teal
    'infrastructure': '#95A5A6', # Gray
    'observability': '#BDC3C7',  # Light gray
    'external': '#27AE60',       # Green
}

# Edge styles per confidence tier
EDGE_STYLES = {
    'confirmed': 'solid',
    'probable': 'dashed',
    'uncertain': 'dotted',
    'implicit': 'dashdot',
}


class GraphBuilder:

    def __init__(self, result: AnalysisResult):
        self.result = result

    def build(self):
        if not HAS_NX:
            print("⚠️  networkx not installed. Skipping graph build.")
            return None

        G = nx.DiGraph()

        # Add service nodes
        for svc in self.result.services:
            G.add_node(svc.name, language=svc.language, anchor=svc.anchor)

        # Add dependency edges
        for dep in self.result.dependencies:
            if dep.dep_type == 'observability':
                continue   # Don't mix with functional graph

            tier = self._tier(dep)
            G.add_edge(
                dep.from_service,
                dep.to_service,
                dep_type=dep.dep_type,
                subtype=dep.subtype or '',
                confidence=dep.confidence,
                tier=tier,
                color=DEP_COLORS.get(dep.dep_type, '#888888'),
                style=EDGE_STYLES.get(tier, 'solid'),
                evidence=dep.evidence[:100] if dep.evidence else '',
                conditional=dep.conditional,
            )

        return G

    def _tier(self, dep: Dependency) -> str:
        if dep.source in ('string_cooccurrence', 'git_change_coupling'):
            return 'implicit'
        if dep.confidence >= 0.85:
            return 'confirmed'
        if dep.confidence >= 0.65:
            return 'probable'
        return 'uncertain'
