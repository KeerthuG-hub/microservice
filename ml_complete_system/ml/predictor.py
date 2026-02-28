"""
ML Predictor

CHANGES from original:
1. _load_models() no longer loads impact_regressor — it's been removed.
   If an old impact_regressor.pkl exists on disk it is silently skipped.

2. predict() now requires G: nx.DiGraph as a parameter.
   Blast radius / affected services come from cascade_engine.py (BFS),
   NOT from ML model or fan_out approximation.

3. Severity is derived from the actual cascade result (fraction of system
   affected + cascade depth), not from the old formula-based blast_score.

4. Recovery time formula updated — no longer uses affected_count from ML,
   uses cascade result count instead.

5. predict_batch() updated to pass G through.

6. Demo updated to use contract column names (latency_p99 not latency_p99_ms,
   health_score as 0-1, traffic_requests_per_day not requests_per_day).

Call-site change required in app.py / anywhere predict() is called:
    # OLD
    predictor.predict(service_features, change_type='deployment')
    # NEW
    predictor.predict(service_features, G=graph, change_type='deployment')
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.cascade_engine import compute_blast_radius, CascadeResult


class IncidentPredictor:
    """Predict incident risk using trained ML models + graph cascade engine."""

    def __init__(self, models_dir: str = 'data/models'):
        self.models_dir    = Path(models_dir)
        self.models        = {}
        self.feature_names = None
        self._load_models()

    def _load_models(self):
        """Load trained models. impact_regressor is no longer loaded."""
        print(f"📦 Loading models from: {self.models_dir}")

        model_files = {
            'incident_classifier': 'incident_classifier.pkl',
            'severity_regressor':  'severity_regressor.pkl',
            # impact_regressor intentionally removed
        }

        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                self.models[name] = joblib.load(path)
                print(f"   ✅ Loaded: {name}")
            else:
                print(f"   ⚠️  Not found (will use fallback): {path}")

        feature_path = self.models_dir / 'feature_names.pkl'
        if feature_path.exists():
            self.feature_names = joblib.load(feature_path)
            print(f"   ✅ Loaded feature names: {len(self.feature_names)} features")
        else:
            print(f"   ⚠️  feature_names.pkl not found — features will not be aligned")

    # ── Primary prediction ────────────────────────────────────────────────────

    def predict(
        self,
        service_features: Dict,
        G: nx.DiGraph,                        # REQUIRED — actual dependency graph
        change_type: str = 'deployment',
        change_magnitude: float = 0.5,
    ) -> Dict:
        """
        Predict incident risk for a service change.

        Args:
            service_features: Feature dict from feature_engineering.py
            G:                The full NetworkX dependency graph (required for cascade)
            change_type:      'deployment' | 'config_change' | 'code_change' | 'scaling'
            change_magnitude: 0.0-1.0

        Returns:
            Dict with ML risk scores + deterministic cascade results.
        """
        features = service_features.copy()
        features['change_type']      = change_type
        features['change_magnitude'] = change_magnitude
        features.setdefault('has_tests',            1)
        features.setdefault('deployment_frequency', 10)
        features.setdefault('last_incident_days',   30)

        X           = self._prepare_features(features)
        service_name= features.get('service_name', 'unknown')
        predictions : Dict = {'service_name': service_name,
                               'change_type': change_type,
                               'change_magnitude': change_magnitude}

        # ── 1. ML: Incident probability ───────────────────────────────────
        if 'incident_classifier' in self.models:
            proba = float(self.models['incident_classifier'].predict_proba(X)[0][1])
        else:
            proba = self._fallback_incident_prob(features)

        predictions['incident_probability']         = round(proba, 3)
        predictions['incident_probability_percent'] = f"{proba*100:.1f}%"
        predictions['will_incident_occur']          = proba > 0.5

        # ── 2. Graph: Cascade simulation (replaces impact_regressor) ──────
        cascade: Optional[CascadeResult] = None
        if service_name in G:
            cascade = compute_blast_radius(
                G=G,
                failed_service=service_name,
                incident_probability=proba,
            )
            predictions['cascade']           = cascade.to_dict()
            predictions['affected_services'] = cascade.affected_services
            predictions['affected_count']    = cascade.affected_count
            predictions['cascade_depth']     = cascade.max_depth
            predictions['absorbed_count']    = len(cascade.absorbed_services)
        else:
            predictions['cascade']           = None
            predictions['affected_services'] = []
            predictions['affected_count']    = 0
            predictions['cascade_depth']     = 0
            predictions['absorbed_count']    = 0

        # ── 3. Severity from actual cascade result ────────────────────────
        affected_count   = predictions['affected_count']
        total_services   = len(G.nodes) if G else 1
        fraction_affected= affected_count / max(total_services, 1)
        cascade_depth    = predictions['cascade_depth']

        if fraction_affected >= 0.5 or cascade_depth >= 4:
            severity       = 'CRITICAL'
            severity_score = min(1.0, 0.7 + fraction_affected * 0.3)
        elif fraction_affected >= 0.25 or cascade_depth >= 2:
            severity       = 'HIGH'
            severity_score = 0.5 + fraction_affected * 0.4
        elif fraction_affected >= 0.1 or affected_count >= 2:
            severity       = 'MEDIUM'
            severity_score = 0.2 + fraction_affected * 0.5
        else:
            severity       = 'LOW'
            severity_score = fraction_affected * 0.4

        predictions['severity']       = severity
        predictions['severity_score'] = round(severity_score, 3)

        # ── 4. Recovery time ──────────────────────────────────────────────
        base_recovery    = 30
        severity_factor  = severity_score * 180
        affected_factor  = affected_count * 20
        recovery_minutes = int(base_recovery + severity_factor + affected_factor)
        predictions['estimated_recovery_minutes'] = recovery_minutes

        # ── 5. Risk level + recommendation ───────────────────────────────
        if proba >= 0.7 or severity == 'CRITICAL':
            predictions['risk_level']     = 'HIGH'
            wave1_names = []
            if cascade and cascade.to_dict().get('cascade_waves', {}).get(1):
                wave1_names = [w['service'] for w in cascade.to_dict()['cascade_waves'][1][:3]]
            wave_str = f" Immediate impact: {', '.join(wave1_names)}." if wave1_names else ""
            predictions['recommendation'] = (
                f"⚠️  HIGH RISK: Do NOT deploy without staged rollout + rollback plan. "
                f"{affected_count} services in blast radius.{wave_str}"
            )
        elif proba >= 0.5 or severity == 'HIGH':
            predictions['risk_level']     = 'MEDIUM'
            predictions['recommendation'] = (
                f"⚡ MEDIUM RISK: Deploy during low traffic window. "
                f"Monitor {affected_count} downstream services closely."
            )
        elif proba >= 0.3:
            predictions['risk_level']     = 'LOW'
            predictions['recommendation'] = (
                f"✓ LOW RISK: Proceed with caution. "
                f"Have rollback ready. {affected_count} services may be affected."
            )
        else:
            predictions['risk_level']     = 'MINIMAL'
            predictions['recommendation'] = "✓ MINIMAL RISK: Deploy normally. Standard monitoring sufficient."

        return predictions

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(
        self,
        services_features: List[Dict],
        G: nx.DiGraph,
        change_type: str = 'deployment',
        change_magnitude: float = 0.5,
    ) -> pd.DataFrame:
        """Predict for multiple services at once."""
        return pd.DataFrame([
            self.predict(sf, G=G, change_type=change_type, change_magnitude=change_magnitude)
            for sf in services_features
        ])

    # ── Feature preparation ───────────────────────────────────────────────────

    def _prepare_features(self, features: Dict):
        """Convert feature dict to DataFrame aligned to training feature space."""
        exclude = {
            'service_name', 'project',
            'label_incident_occurred', 'label_incident_probability',
            'label_severity_score', 'label_affected_services', 'label_recovery_minutes',
        }
        clean = {k: v for k, v in features.items() if k not in exclude}

        df = pd.DataFrame([clean])

        for col in ['language', 'change_type']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        if self.feature_names is not None:
            df = df.reindex(columns=self.feature_names, fill_value=0)

        return df

    # ── Fallbacks ─────────────────────────────────────────────────────────────

    def _fallback_incident_prob(self, features: Dict) -> float:
        """Rule-based fallback when incident_classifier model not available."""
        error_rate = features.get('error_rate', 0.01)
        centrality = features.get('betweenness_centrality', 0)
        magnitude  = features.get('change_magnitude', 0.5)
        return float(min(0.95, error_rate * 5 + centrality * 0.3 + magnitude * 0.4))


# ── Real pipeline (replaces demo) ────────────────────────────────────────────

def build_graph_from_json(json_path: str) -> nx.DiGraph:
    """Build real graph from run.py output JSON. Filters infra + low-confidence edges."""
    import json as _json
    with open(json_path) as f:
        data = _json.load(f)

    # Catch infra nodes not flagged in JSON
    INFRA_PATTERNS = {
        'gcp', 'aws', 'azure', 'postgres', 'postgresql', 'mysql', 'redis',
        'mongo', 'mongodb', 'kafka', 'rabbitmq', 'elasticsearch', 'loadgenerator',
        'adservice', 'shoppingassistantservice', 'jaeger', 'zipkin',
        'prometheus', 'grafana', 'istio', 'envoy', 'nginx', 'locust',
    }
    def is_infra(name: str) -> bool:
        n = name.lower().replace('-', '').replace('_', '')
        return any(p.replace('-','').replace('_','') in n for p in INFRA_PATTERNS)

    infra    = set(data.get('infrastructure', []))
    services = {s['name'] for s in data.get('services', [])
                if not s.get('is_infrastructure', False)
                and s['name'] not in infra
                and not is_infra(s['name'])}

    G = nx.DiGraph()
    for svc in data.get('services', []):
        if svc['name'] in services:
            G.add_node(svc['name'],
                language=svc.get('language', 'unknown'),
                circuit_breaker=False,
                health_score=0.8,
                error_rate=0.01,
            )

    for tier in ('confirmed', 'probable'):
        for dep in data.get(tier, []):
            src, tgt, conf = dep.get('from',''), dep.get('to',''), dep.get('confidence', 0)
            if not src or not tgt: continue
            if src.startswith('__') or tgt.startswith('__'): continue
            if src in infra or tgt in infra: continue
            if is_infra(src) or is_infra(tgt): continue
            if src not in services or tgt not in services: continue
            if conf < 0.4: continue
            G.add_edge(src, tgt, type=dep.get('type','http'), confidence=conf)

    print(f"✅ Graph: {G.number_of_nodes()} services, {G.number_of_edges()} edges")
    return G


def run_predictions(
    project_name: str,
    change_type: str = 'deployment',
    change_magnitude: float = 0.5,
    outputs_dir: str = 'data/outputs',
    features_dir: str = 'data/features',
    models_dir:   str = 'data/models',
):
    import json as _json

    print(f"\n{'='*60}\n  PREDICTION PIPELINE — {project_name}\n{'='*60}\n")

    # Step 1: Load graph from run.py JSON
    json_path = Path(outputs_dir) / f'{project_name}_latest.json'
    if not json_path.exists():
        print(f"❌ Not found: {json_path}")
        print(f"   Run: python run.py data/test_projects/{project_name} --non-interactive --no-llm")
        sys.exit(1)
    G = build_graph_from_json(str(json_path))

    # Step 2: Load features CSV from feature_engineering.py
    feat_path = Path(features_dir) / f'{project_name}_features.csv'
    if not feat_path.exists():
        print(f"❌ Not found: {feat_path}")
        print(f"   Run: python ml/feature_engineering.py {project_name}")
        sys.exit(1)
    features_df = pd.read_csv(feat_path)

    # Push real health/error values into graph nodes for cascade absorption
    for _, row in features_df.iterrows():
        svc = row['service_name']
        if svc in G:
            hs = row.get('health_score', 80)
            G.nodes[svc]['health_score'] = hs / 100.0 if hs > 1.5 else hs
            G.nodes[svc]['error_rate']   = row.get('error_rate', 0.01)

    # Step 3: Predict for every service in CSV
    predictor   = IncidentPredictor(models_dir=models_dir)
    predictions = []

    for _, row in features_df.iterrows():
        svc = row['service_name']
        if svc not in G:
            continue
        result = predictor.predict(
            service_features=row.to_dict(),
            G=G,
            change_type=change_type,
            change_magnitude=change_magnitude,
        )
        result['estimated_affected_services'] = result.get('affected_count', 0)
        # Strip cascade object — too large for app.py
        clean = {k: v for k, v in result.items() if k != 'cascade'}
        if 'affected_services' in clean:
            clean['affected_services'] = list(clean['affected_services'])
        predictions.append(clean)
        print(f"   {svc}: {result['incident_probability_percent']} [{result['risk_level']}] "
              f"→ {result['affected_count']} affected")

    # Step 4: Save predictions.json
    out_path = Path(outputs_dir) / 'predictions.json'
    with open(out_path, 'w') as f:
        _json.dump(predictions, f, indent=2)

    high = sum(1 for p in predictions if p['risk_level'] == 'HIGH')
    print(f"\n✅ Saved {len(predictions)} predictions → {out_path}")
    print(f"   HIGH: {high}  MEDIUM: {sum(1 for p in predictions if p['risk_level']=='MEDIUM')}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--project',    required=True,           help='e.g. online-boutique')
    ap.add_argument('--change-type',default='deployment',
                    choices=['deployment','config_change','code_change','scaling'])
    ap.add_argument('--magnitude',  type=float, default=0.5, help='0.0-1.0')
    ap.add_argument('--outputs',    default='data/outputs')
    ap.add_argument('--features',   default='data/features')
    ap.add_argument('--models',     default='data/models')
    args = ap.parse_args()

    run_predictions(
        project_name   = args.project,
        change_type    = args.change_type,
        change_magnitude=args.magnitude,
        outputs_dir    = args.outputs,
        features_dir   = args.features,
        models_dir     = args.models,
    )