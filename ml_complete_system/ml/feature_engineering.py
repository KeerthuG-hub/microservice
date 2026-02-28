"""
Feature Engineering - Extract ML features from dependency graphs.

Features extracted:
- Graph metrics (centrality, coupling, depth)
- Dependency counts (upstream, downstream)
- Service characteristics (language, complexity)
- Synthetic metrics (error rates, latency estimates)
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# Add project root to path (go up: ml/ → ml_complete_system/ → project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.models import AnalysisResult, Dependency, Service


class FeatureEngineer:
    """Extract ML features from dependency analysis results."""
    
    def __init__(self):
        self.features_cache = {}
    
    def extract_features(self, result: AnalysisResult) -> pd.DataFrame:
        """
        Extract features for all services in the analysis result.
        
        Returns:
            DataFrame with one row per service, columns are features
        """
        print("\n🔧 Extracting ML features...")
        
        # Filter to REAL dependencies only (no implicit noise)
        real_deps = self._filter_real_dependencies(result.dependencies)
        print(f"   📊 Using {len(real_deps)}/{len(result.dependencies)} real deps (filtered implicit)")
        
        # Build directed graph from real dependencies
        G = self._build_graph(result.services, real_deps)
        
        # Extract features for each service
        features_list = []
        for service in result.services:
            if service.name not in G:
                continue  # Skip infrastructure nodes
            
            features = self._extract_service_features(service, G, result)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        print(f"   ✅ Extracted {len(df)} service feature vectors with {len(df.columns)} features")
        
        return df
    
    def _filter_real_dependencies(self, dependencies: List[Dependency]) -> List[Dependency]:
        """
        Filter out implicit/noisy dependencies.
        Keep only high-confidence, confirmed dependencies.
        """
        real_deps = []
        for dep in dependencies:
            # Exclude implicit dependencies
            if dep.source in ('string_cooccurrence', 'git_change_coupling'):
                continue
            
            # Exclude observability (not business impact)
            if dep.dep_type == 'observability':
                continue
            
            # Keep only confident dependencies
            if dep.confidence >= 0.65:
                real_deps.append(dep)
        
        return real_deps
    
    def _build_graph(self, services: List[Service], 
                     dependencies: List[Dependency]) -> nx.DiGraph:
        """Build directed graph from dependencies."""
        G = nx.DiGraph()
        
        # Add service nodes
        for service in services:
            G.add_node(service.name, **{
                'language': service.language,
                'path': service.path,
                'ports': len(service.ports)
            })
        
        # Add dependency edges
        for dep in dependencies:
            if dep.from_service in G and dep.to_service in G:
                G.add_edge(
                    dep.from_service,
                    dep.to_service,
                    dep_type=dep.dep_type,
                    subtype=dep.subtype,
                    confidence=dep.confidence
                )
        
        return G
    
    def _extract_service_features(self, service: Service, 
                                   G: nx.DiGraph,
                                   result: AnalysisResult) -> Dict:
        """Extract all features for a single service."""
        
        features = {
            'service_name': service.name,
            'language': service.language,
        }
        
        # === GRAPH TOPOLOGY FEATURES ===
        features.update(self._graph_features(service.name, G))
        
        # === DEPENDENCY COUNT FEATURES ===
        features.update(self._dependency_counts(service.name, G))
        
        # === CENTRALITY FEATURES ===
        features.update(self._centrality_features(service.name, G))
        
        # === COUPLING FEATURES ===
        features.update(self._coupling_features(service.name, G))
        
        # === SYNTHETIC METRICS (estimates for training) ===
        features.update(self._synthetic_metrics(service.name, G, features))
        
        return features
    
    def _graph_features(self, service_name: str, G: nx.DiGraph) -> Dict:
        """Basic graph topology features."""
        return {
            'has_upstream': 1 if G.in_degree(service_name) > 0 else 0,
            'has_downstream': 1 if G.out_degree(service_name) > 0 else 0,
            'is_leaf': 1 if G.out_degree(service_name) == 0 else 0,
            'is_root': 1 if G.in_degree(service_name) == 0 else 0,
        }
    
    def _dependency_counts(self, service_name: str, G: nx.DiGraph) -> Dict:
        """Count various types of dependencies."""
        # Direct dependencies
        upstream = G.in_degree(service_name)
        downstream = G.out_degree(service_name)
        
        # Transitive dependencies (all descendants)
        try:
            transitive_downstream = len(nx.descendants(G, service_name))
        except nx.NetworkXError:
            transitive_downstream = 0
        
        # Transitive upstream (all ancestors)
        try:
            transitive_upstream = len(nx.ancestors(G, service_name))
        except nx.NetworkXError:
            transitive_upstream = 0
        
        return {
            'upstream_count': upstream,
            'downstream_count': downstream,
            'transitive_downstream': transitive_downstream,
            'transitive_upstream': transitive_upstream,
            'total_coupling': upstream + downstream,
        }
    
    def _centrality_features(self, service_name: str, G: nx.DiGraph) -> Dict:
        """Centrality metrics - how 'important' is this service?"""
        
        # Betweenness: how often service is on path between others
        betweenness = nx.betweenness_centrality(G)
        
        # PageRank: importance based on incoming edges
        pagerank = nx.pagerank(G)
        
        # In-degree centrality: normalized incoming edges
        in_centrality = nx.in_degree_centrality(G)
        
        # Out-degree centrality: normalized outgoing edges
        out_centrality = nx.out_degree_centrality(G)
        
        return {
            'betweenness_centrality': betweenness.get(service_name, 0.0),
            'pagerank': pagerank.get(service_name, 0.0),
            'in_degree_centrality': in_centrality.get(service_name, 0.0),
            'out_degree_centrality': out_centrality.get(service_name, 0.0),
        }
    
    def _coupling_features(self, service_name: str, G: nx.DiGraph) -> Dict:
        """How tightly coupled is this service?"""
        
        # Fan-in: number of services that depend on this one
        fan_in = G.in_degree(service_name)
        
        # Fan-out: number of services this one depends on
        fan_out = G.out_degree(service_name)
        
        # Coupling ratio
        coupling_ratio = fan_out / max(fan_in, 1)
        
        # Critical path depth (longest path through this node)
        try:
            # Max depth to any leaf
            paths = nx.single_source_shortest_path_length(G, service_name)
            max_depth = max(paths.values()) if paths else 0
        except:
            max_depth = 0
        
        return {
            'fan_in': fan_in,
            'fan_out': fan_out,
            'coupling_ratio': coupling_ratio,
            'max_path_depth': max_depth,
        }
    
    def _synthetic_metrics(self, service_name: str, G: nx.DiGraph, 
                          features: Dict) -> Dict:
        """
        Generate synthetic metrics for training.
        In production, these would come from real monitoring data.
        """
        
        # Estimate based on service characteristics
        downstream = features['downstream_count']
        betweenness = features['betweenness_centrality']
        pagerank = features['pagerank']
        
        # Services with more dependencies tend to have more issues
        complexity_score = (
            downstream * 0.3 +
            betweenness * 0.4 +
            pagerank * 0.3
        )
        
        # Synthetic error rate (0.001 to 0.05)
        # More complex = higher error rate
        error_rate = 0.001 + (complexity_score * 0.049)
        error_rate = min(error_rate, 0.05)
        
        # Synthetic latency p99 (100ms to 2000ms)
        # More downstream deps = higher latency
        latency_p99 = 100 + (downstream * 150) + (complexity_score * 500)
        latency_p99 = min(latency_p99, 2000)
        
        # Synthetic traffic (1000 to 100000 requests/day)
        # High centrality = high traffic
        requests_per_day = 1000 + (pagerank * 99000)
        
        # Health score (0-100)
        # Lower complexity = higher health
        health_score = 100 - (complexity_score * 50)
        health_score = max(health_score, 20)
        
        return {
            'error_rate': round(error_rate, 4),
            'latency_p99_ms': round(latency_p99, 1),
            'requests_per_day': round(requests_per_day, 0),
            'health_score': round(health_score, 1),
            'complexity_score': round(complexity_score, 3),
        }


def load_analysis_result_from_json(json_path: str) -> AnalysisResult:
    """
    Reconstruct an AnalysisResult from the report JSON produced by run.py.

    Args:
        json_path: Path to <project>_latest.json written by run.py

    Returns:
        AnalysisResult with services and dependencies populated
    """
    import json as _json
    print(f"\n📂 Loading analysis result from: {json_path}")
    data = _json.loads(Path(json_path).read_text())

    # Reconstruct Service objects
    services = [
        Service(
            name=s['name'],
            path=s.get('path', ''),
            language=s.get('language', 'unknown'),
            anchor=s.get('anchor', ''),
            ports=s.get('ports', []),
            is_infrastructure=s.get('is_infrastructure', False),
        )
        for s in data.get('services', [])
    ]

    # Reconstruct Dependency objects — merge all tiers
    dependencies = []
    for tier in ('confirmed', 'probable', 'uncertain', 'implicit', 'observability'):
        for d in data.get(tier, []):
            dependencies.append(Dependency(
                from_service=d['from'],
                to_service=d['to'],
                dep_type=d.get('type', 'endpoint'),
                subtype=d.get('subtype'),
                confidence=d.get('confidence', 0.0),
                evidence=d.get('evidence', ''),
                source=(d.get('sources') or [''])[0],
                sources=d.get('sources', []),
                topic_or_table=d.get('topic_or_table'),
                conditional=d.get('conditional', False),
            ))

    result = AnalysisResult(
        services=services,
        dependencies=dependencies,
        total_services=data.get('metadata', {}).get('total_services', len(services)),
        total_dependencies=data.get('metadata', {}).get('total_dependencies', len(dependencies)),
    )
    print(f"   ✅ Loaded {len(services)} services, {len(dependencies)} dependencies")
    return result


def extract_features_from_json(json_path: str) -> pd.DataFrame:
    """
    Load run.py JSON output and extract ML features.

    Args:
        json_path: Path to <project>_latest.json written by run.py

    Returns:
        DataFrame of features for all services
    """
    result = load_analysis_result_from_json(json_path)
    engineer = FeatureEngineer()
    features_df = engineer.extract_features(result)
    return features_df




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python feature_engineering.py <project_name>")
        print("Example: python feature_engineering.py bank-of-anthos")
        print("         python feature_engineering.py TeaStore")
        print("         python feature_engineering.py online-boutique")
        print()
        print("NOTE: run.py must be executed first — it saves data/outputs/<project_name>_latest.json")
        sys.exit(1)

    project_name = sys.argv[1]   # e.g. bank-of-anthos, TeaStore, online-boutique

    # Auto-locate the latest JSON written by run.py for this project
    json_path = Path(f'data/outputs/{project_name}_latest.json')
    if not json_path.exists():
        print(f"❌ Could not find: {json_path}")
        print(f"   → Run first: python3 run.py data/test_projects/{project_name} --non-interactive --no-llm")
        sys.exit(1)

    features = extract_features_from_json(str(json_path))
    features['project'] = project_name   # ensure correct project label

    print("\n📊 Sample features:")
    print(features.head())

    output_path = f'data/features/{project_name}_features.csv'
    Path('data/features').mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")

