"""
Training Data Generator

CHANGES from original:
1. REMOVED label_affected_services and _calculate_affected_services()
   Reason: affected service count is deterministic from graph topology.
   BFS in cascade_engine.py computes it exactly. Training ML to approximate
   what graph traversal can answer exactly is wrong architecture.

2. REMOVED affected_services parameter from _calculate_recovery_time()
   Recovery time now uses structural complexity as proxy instead.

3. Column names use ML contract names throughout:
     latency_p99              (was latency_p99_ms)
     traffic_requests_per_day (was requests_per_day)
     health_score  0.0-1.0    (was 0-100)
   So synthetic CSVs match real feature_engineering.py output from day one.

LABELS KEPT (all still use formulas — appropriate for ML):
  label_incident_occurred   — random draw against conditional P(incident)
  label_incident_probability — the raw probability value
  label_severity_score      — beta-distributed risk proxy per service profile
  label_recovery_minutes    — detection + isolation + remediation components
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ── Risk thresholds ───────────────────────────────────────────────────────────

CRITICAL_CENTRALITY  = 0.4
HIGH_DOWNSTREAM      = 5
DEGRADED_ERROR_RATE  = 0.02
RECENT_INCIDENT_DAYS = 7
LARGE_CHANGE         = 0.6


class TrainingDataGenerator:
    """Generate synthetic training data with principled label generation."""

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_from_features(self, features_df: pd.DataFrame,
                               samples_per_service: int = 10) -> pd.DataFrame:
        """
        Generate training rows from a service feature DataFrame.
        Each service gets N scenario variations (different change types/magnitudes).
        """
        print(f"\n🎲 Generating synthetic training data...")
        print(f"   Services: {len(features_df)} | Samples per service: {samples_per_service}")

        rows = []
        for _, svc in features_df.iterrows():
            for _ in range(samples_per_service):
                rows.append(self._generate_single_example(svc))

        df = pd.DataFrame(rows)
        label_cols = [c for c in df.columns if c.startswith('label_')]
        print(f"   ✅ Generated {len(df)} examples")
        print(f"   Labels: {label_cols}")
        print(f"   Incident rate: {df['label_incident_occurred'].mean()*100:.1f}%")
        return df

    def generate_synthetic_project_data(self, num_services: int = 20) -> pd.DataFrame:
        """
        Generate completely synthetic service features (no real project needed).
        Mix of profiles gives model diversity during training.
        """
        print(f"\n🏗️  Generating synthetic project ({num_services} services)...")
        rows = []
        for i in range(num_services):
            profile = np.random.choice(
                ['critical_hub', 'high_fan_out', 'leaf', 'standard'],
                p=[0.1, 0.2, 0.3, 0.4],
            )
            rows.append(self._synthetic_service_features(f'service_{i}', profile))

        df = pd.DataFrame(rows)
        print(f"   ✅ Generated {len(df)} synthetic services")
        return df

    # ── Single example ────────────────────────────────────────────────────────

    def _generate_single_example(self, service_features) -> Dict:
        """One training row: perturbed features + labels."""
        example = self._perturb_features(dict(service_features))
        example = self._add_change_scenario(example)
        profile  = self._classify_service(example)

        incident_prob = self._calculate_incident_probability(example, profile)
        occurred      = int(np.random.random() < incident_prob)

        example['label_incident_occurred']    = occurred
        example['label_incident_probability'] = round(float(incident_prob), 4)

        if occurred:
            example['label_severity_score']   = self._calculate_severity(example, profile)
            example['label_recovery_minutes'] = self._calculate_recovery_time(
                example, profile, example['label_severity_score']
            )
        else:
            example['label_severity_score']   = 0.0
            example['label_recovery_minutes'] = 0

        # label_affected_services intentionally NOT generated.
        # cascade_engine.py computes the exact list via BFS on the real graph.

        return example

    def _perturb_features(self, example: Dict) -> Dict:
        """Add realistic variance so each scenario of the same service differs."""
        for key in ['downstream_count', 'upstream_count', 'transitive_downstream',
                    'transitive_upstream', 'total_coupling', 'fan_in', 'fan_out']:
            if key in example:
                example[key] = max(0, int(example[key] * np.random.uniform(0.8, 1.2)))

        for key in ['betweenness_centrality', 'pagerank', 'in_degree_centrality',
                    'out_degree_centrality', 'complexity_score']:
            if key in example:
                example[key] = float(np.clip(example[key] * np.random.uniform(0.75, 1.25), 0.0, 1.0))

        if 'error_rate' in example:
            example['error_rate'] = float(np.clip(
                example['error_rate'] * np.random.uniform(0.5, 2.0), 0.001, 0.05))

        # Contract column name: latency_p99 (not latency_p99_ms)
        if 'latency_p99' in example:
            example['latency_p99'] = float(np.clip(
                example['latency_p99'] * np.random.uniform(0.7, 1.5), 100, 5000))

        # health_score is 0-1 (contract scale)
        if 'health_score' in example:
            example['health_score'] = float(np.clip(
                example['health_score'] + np.random.normal(0, 0.05), 0.0, 1.0))

        # Contract column name: traffic_requests_per_day (not requests_per_day)
        if 'traffic_requests_per_day' in example:
            example['traffic_requests_per_day'] = float(max(
                0, example['traffic_requests_per_day'] * np.random.uniform(0.8, 1.2)))

        return example

    def _add_change_scenario(self, example: Dict) -> Dict:
        """Add a change event context — varies per training sample."""
        change_type = np.random.choice(
            ['deployment', 'config_change', 'code_change', 'scaling'],
            p=[0.40, 0.30, 0.20, 0.10],
        )
        example['change_type'] = change_type

        if change_type == 'deployment':
            example['change_magnitude'] = float(np.random.beta(2, 2))
        elif change_type == 'config_change':
            example['change_magnitude'] = float(np.random.beta(2, 5))
        elif change_type == 'code_change':
            example['change_magnitude'] = float(np.random.beta(3, 2))
        else:
            example['change_magnitude'] = float(np.random.beta(2, 3))

        example['has_tests']            = int(np.random.random() > 0.3)
        example['deployment_frequency'] = int(np.random.uniform(1, 30))
        example['last_incident_days']   = int(np.random.exponential(30))
        return example

    # ── Service classification ────────────────────────────────────────────────

    def _classify_service(self, features: Dict) -> str:
        """Classify into profile — drives label generation logic."""
        centrality = features.get('betweenness_centrality', 0)
        downstream = features.get('downstream_count', 0)
        transitive = features.get('transitive_downstream', 0)

        if centrality >= CRITICAL_CENTRALITY and downstream >= HIGH_DOWNSTREAM:
            return 'critical_hub'
        elif downstream >= HIGH_DOWNSTREAM or transitive >= 10:
            return 'high_fan_out'
        elif downstream == 0 and features.get('upstream_count', 0) <= 1:
            return 'leaf'
        return 'standard'

    # ── Label formulas ────────────────────────────────────────────────────────

    def _calculate_incident_probability(self, features: Dict, profile: str) -> float:
        """
        Conditional probability of an incident — non-linear interaction logic.
        Same service + same change doesn't always produce the same outcome,
        so ML learning this pattern is appropriate.
        """
        error_rate       = features.get('error_rate', 0.01)
        centrality       = features.get('betweenness_centrality', 0)
        change_magnitude = features.get('change_magnitude', 0.3)
        last_incident    = features.get('last_incident_days', 30)
        has_tests        = features.get('has_tests', 1)

        base = 0.05

        if error_rate > DEGRADED_ERROR_RATE * 2:
            base += 0.35
        elif error_rate > DEGRADED_ERROR_RATE:
            base += 0.20

        if profile == 'critical_hub' and change_magnitude > LARGE_CHANGE:
            base += 0.30
        elif profile == 'critical_hub':
            base += 0.15
        elif profile == 'high_fan_out' and change_magnitude > LARGE_CHANGE:
            base += 0.20
        elif profile == 'leaf':
            base -= 0.03

        if last_incident < RECENT_INCIDENT_DAYS:
            base += 0.20

        if not has_tests:
            base += 0.10

        base += centrality * change_magnitude * 0.5

        return float(np.clip(base + np.random.normal(0, 0.05), 0.02, 0.97))

    def _calculate_severity(self, features: Dict, profile: str) -> float:
        """
        Severity score 0.0-1.0 — a risk proxy, NOT a count of affected services.
        Uses beta distribution shaped by service profile.
        Appropriate for ML: "how bad will this be?" is genuinely uncertain.
        """
        centrality = features.get('betweenness_centrality', 0)
        downstream = features.get('downstream_count', 0)
        complexity = features.get('complexity_score', 0.3)

        if profile == 'critical_hub':
            alpha, beta = 5, 2
        elif profile == 'high_fan_out':
            alpha, beta = 4, 3
        elif profile == 'leaf':
            alpha, beta = 2, 5
        else:
            alpha, beta = 3, 3

        base = np.random.beta(alpha, beta)
        adjustment = (
            centrality * 0.15 +
            min(downstream / 10.0, 1.0) * 0.10 +
            complexity * 0.05
        )
        return round(float(np.clip(base + adjustment, 0.0, 1.0)), 4)

    def _calculate_recovery_time(self, features: Dict, profile: str,
                                  severity: float) -> int:
        """
        Recovery time in minutes.
        Three drawn components: detection + isolation + remediation.
        Appropriate for ML: recovery varies even for identical incidents.
        NOTE: no longer takes affected_services count (removed label).
        """
        error_rate = features.get('error_rate', 0.01)
        complexity = features.get('complexity_score', 0.3)
        coupling   = features.get('total_coupling', 3)

        detection = (
            int(np.random.uniform(2, 10))
            if error_rate > DEGRADED_ERROR_RATE
            else int(np.random.uniform(5, 30))
        )

        iso_base  = 10 + (complexity * 40) + (coupling * 3)
        isolation = max(5, int(np.random.normal(iso_base, iso_base * 0.2)))

        rem_base    = 15 + (severity * 90)
        remediation = max(5, int(np.random.normal(rem_base, rem_base * 0.25)))

        return max(10, detection + isolation + remediation)

    # ── Synthetic feature generation ──────────────────────────────────────────

    def _synthetic_service_features(self, name: str, profile: str) -> Dict:
        """Generate realistic feature dict for a given profile."""
        if profile == 'critical_hub':
            downstream = int(np.random.uniform(6, 15))
            upstream   = int(np.random.uniform(1, 4))
            transitive = int(np.random.uniform(10, 30))
            centrality = float(np.random.uniform(0.4, 0.9))
            error_rate = float(np.random.beta(2, 8) * 0.1)
            health     = float(np.random.uniform(0.4, 0.8))
        elif profile == 'high_fan_out':
            downstream = int(np.random.uniform(5, 12))
            upstream   = int(np.random.uniform(1, 6))
            transitive = int(np.random.uniform(8, 20))
            centrality = float(np.random.uniform(0.1, 0.4))
            error_rate = float(np.random.beta(2, 10) * 0.08)
            health     = float(np.random.uniform(0.5, 0.9))
        elif profile == 'leaf':
            downstream = 0
            upstream   = int(np.random.uniform(1, 3))
            transitive = 0
            centrality = float(np.random.uniform(0.0, 0.1))
            error_rate = float(np.random.beta(1, 15) * 0.05)
            health     = float(np.random.uniform(0.7, 1.0))
        else:  # standard
            downstream = int(np.random.uniform(1, 5))
            upstream   = int(np.random.uniform(1, 5))
            transitive = int(np.random.uniform(2, 10))
            centrality = float(np.random.uniform(0.05, 0.3))
            error_rate = float(np.random.beta(1, 20) * 0.06)
            health     = float(np.random.uniform(0.6, 0.95))

        pagerank   = float(np.random.beta(2, 5))
        complexity = float(np.clip(
            centrality * 0.6 + downstream / 20 + np.random.normal(0, 0.1), 0, 1
        ))

        return {
            'service_name':            name,
            'language':                np.random.choice(
                ['python', 'go', 'java', 'javascript'], p=[0.35, 0.25, 0.25, 0.15]),
            'upstream_count':          upstream,
            'downstream_count':        downstream,
            'transitive_downstream':   transitive,
            'transitive_upstream':     int(np.random.uniform(0, upstream * 3)),
            'total_coupling':          upstream + downstream,
            'betweenness_centrality':  centrality,
            'pagerank':                pagerank,
            'in_degree_centrality':    float(np.random.uniform(0, 0.5)),
            'out_degree_centrality':   float(np.random.uniform(0, 0.5)),
            'fan_in':                  upstream,
            'fan_out':                 downstream,
            'coupling_ratio':          downstream / max(upstream, 1),
            'max_path_depth':          int(np.random.poisson(3)),
            'error_rate':              error_rate,
            'latency_p99':             float(np.random.lognormal(4, 1)),    # contract name
            'traffic_requests_per_day':float(np.random.lognormal(10, 2)),   # contract name
            'health_score':            health,                               # 0-1 already
            'complexity_score':        complexity,
            'project':                 'synthetic',
        }


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_training_dataset(
    output_dir: str = 'data/training_data',
    num_synthetic_services: int = 20,
    samples_per_service: int = 15,
) -> pd.DataFrame:
    gen = TrainingDataGenerator()

    features_df = gen.generate_synthetic_project_data(num_synthetic_services)
    training_df = gen.generate_from_features(features_df, samples_per_service)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_df.to_csv(out / 'training_data.csv', index=False)
    features_df.to_csv(out / 'service_features.csv', index=False)

    print(f"\n📊 Stats:")
    print(f"   Total rows:         {len(training_df)}")
    print(f"   Incident rate:      {training_df['label_incident_occurred'].mean():.1%}")
    print(f"   Avg severity:       {training_df['label_severity_score'].mean():.3f}")
    print(f"   Avg recovery (min): {training_df['label_recovery_minutes'].mean():.0f}")
    print(f"   label_affected_services: NOT generated — cascade_engine.py handles this")

    return training_df


if __name__ == "__main__":
    # Real project training: bank-of-anthos + TeaStore
    # Online Boutique held out for validation — NOT loaded here
    bank_path     = Path("data/features/bank-of-anthos_features.csv")
    teastore_path = Path("data/features/TeaStore_features.csv")

    if bank_path.exists() and teastore_path.exists():
        print("📂 Loading real project features (training set)...")
        bank_features     = pd.read_csv(bank_path)
        teastore_features = pd.read_csv(teastore_path)
        print(f"   bank-of-anthos: {len(bank_features)} services")
        print(f"   TeaStore:       {len(teastore_features)} services")
        print(f"   (online-boutique is held out for validation)")

        combined = pd.concat([bank_features, teastore_features], ignore_index=True)
        print(f"   Combined:       {len(combined)} services")

        generator   = TrainingDataGenerator()
        training_df = generator.generate_from_features(combined, samples_per_service=50)

        Path("data/training_data").mkdir(parents=True, exist_ok=True)
        training_df.to_csv("data/training_data/training_data.csv", index=False)
        print("✅ Saved: data/training_data/training_data.csv")
        print("📋 Sample:")
        print(training_df[[
            "service_name", "downstream_count", "betweenness_centrality",
            "error_rate", "change_type", "change_magnitude",
            "label_incident_occurred", "label_severity_score",
            "label_recovery_minutes",
        ]].head(5).to_string())
    else:
        print("⚠️  Real project CSVs not found — using synthetic generation.")
        print("    Run feature_engineering.py on bank-of-anthos and TeaStore first.")
        generate_training_dataset()