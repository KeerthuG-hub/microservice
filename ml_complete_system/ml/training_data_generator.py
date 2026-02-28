"""
Training Data Generator - Generate synthetic training data for ML models.

Label generation approach:
- Uses NON-LINEAR conditional logic instead of a simple weighted sum
- Each risk factor interacts with others (compounding effects)
- Uses beta distributions to model realistic uncertainty/variance
- Change type affects risk differently depending on service criticality
- Failure modes are categorized (cascade vs isolated vs degradation)
- Recovery time modeled separately per failure mode

Why this is better than a weighted formula:
- Real incidents aren't linear sums — they're threshold + interaction effects
- A service with 10 downstream deps AND high error rate is disproportionately risky
- Change type matters differently for critical vs leaf services
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ─────────────────────────────────────────────
# Risk thresholds — based on structural reasoning
# (not arbitrary weights)
# ─────────────────────────────────────────────

# A service is "critical" if it sits on many shortest paths
CRITICAL_CENTRALITY = 0.4        # top ~20% of services in a typical graph

# High fan-out means many services depend on you directly
HIGH_DOWNSTREAM = 5              # more than 5 direct dependents = high blast radius

# Error rate above 2% is generally considered degraded (common SRE threshold)
DEGRADED_ERROR_RATE = 0.02

# Recent incident: within a week = elevated repeat risk
RECENT_INCIDENT_DAYS = 7

# Change magnitude above 0.6 = significant change (major refactor / large deploy)
LARGE_CHANGE = 0.6


class TrainingDataGenerator:
    """Generate synthetic training data with principled label generation."""

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.random_seed = random_seed

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def generate_from_features(self, features_df: pd.DataFrame,
                               samples_per_service: int = 10) -> pd.DataFrame:
        """
        Generate training data from extracted service features.

        Args:
            features_df: DataFrame with one row per service (from feature_engineering.py)
            samples_per_service: How many scenario variations to create per service

        Returns:
            DataFrame with features + labels
        """
        print(f"\n🎲 Generating synthetic training data...")
        print(f"   Services: {len(features_df)}")
        print(f"   Samples per service: {samples_per_service}")

        training_examples = []
        for _, service_features in features_df.iterrows():
            for _ in range(samples_per_service):
                example = self._generate_single_example(service_features)
                training_examples.append(example)

        training_df = pd.DataFrame(training_examples)

        print(f"   ✅ Generated {len(training_df)} training examples")
        print(f"   Features: {len([c for c in training_df.columns if not c.startswith('label_')])}")
        print(f"   Labels: {len([c for c in training_df.columns if c.startswith('label_')])}")
        print(f"   Incident rate: {training_df['label_incident_occurred'].mean()*100:.1f}%")

        return training_df

    def generate_synthetic_project_data(self, num_services: int = 20) -> pd.DataFrame:
        """
        Generate completely synthetic service features.
        Used when no real project is available to extract from.
        """
        print(f"\n🏗️  Generating synthetic project with {num_services} services...")

        services = []
        for i in range(num_services):
            service = {
                'service_name': f'service_{i}',
                'language': np.random.choice(['python', 'go', 'java', 'javascript'],
                                             p=[0.35, 0.25, 0.25, 0.15]),

                # Graph structure — Poisson is realistic for microservice graphs
                'upstream_count': int(np.random.poisson(2)),
                'downstream_count': int(np.random.poisson(3)),
                'transitive_downstream': int(np.random.poisson(5)),
                'transitive_upstream': int(np.random.poisson(4)),
                'total_coupling': 0,

                # Centrality — beta(2,5) skews toward low values (most services aren't critical hubs)
                'betweenness_centrality': float(np.random.beta(2, 5)),
                'pagerank': float(np.random.beta(2, 5)),
                'in_degree_centrality': float(np.random.uniform(0, 0.5)),
                'out_degree_centrality': float(np.random.uniform(0, 0.5)),

                # Coupling
                'fan_in': int(np.random.poisson(2)),
                'fan_out': int(np.random.poisson(2)),
                'coupling_ratio': float(np.random.uniform(0.5, 2.0)),
                'max_path_depth': int(np.random.poisson(3)),

                # Operational metrics
                'error_rate': float(np.random.uniform(0.001, 0.05)),
                'latency_p99_ms': float(np.random.uniform(100, 2000)),
                'requests_per_day': float(np.random.uniform(1000, 100000)),
                'health_score': float(np.random.uniform(50, 100)),
                'complexity_score': float(np.random.beta(2, 5)),

                'project': 'synthetic'
            }
            service['total_coupling'] = service['upstream_count'] + service['downstream_count']
            services.append(service)

        features_df = pd.DataFrame(services)
        print(f"   ✅ Generated {len(features_df)} synthetic services")
        return features_df

    # ──────────────────────────────────────────
    # Core generation logic
    # ──────────────────────────────────────────

    def _generate_single_example(self, service_features: pd.Series) -> Dict:
        """Generate one training example: perturbed features + realistic labels."""

        example = self._perturb_features(service_features.to_dict())
        example = self._add_change_scenario(example)

        # Classify what kind of service this is — drives label logic
        service_profile = self._classify_service(example)

        # Generate labels using conditional interaction logic
        incident_prob = self._calculate_incident_probability(example, service_profile)

        example['label_incident_occurred'] = int(np.random.random() < incident_prob)
        example['label_incident_probability'] = round(float(incident_prob), 4)
        example['label_severity_score'] = self._calculate_severity(example, service_profile, incident_prob)
        example['label_affected_services'] = self._calculate_affected_services(example, service_profile)
        example['label_recovery_minutes'] = self._calculate_recovery_time(
            example, service_profile,
            example['label_severity_score'],
            example['label_affected_services']
        )

        return example

    def _perturb_features(self, example: Dict) -> Dict:
        """
        Add realistic variance to features.
        Each service generates multiple scenarios — variance models
        real-world fluctuation in metrics over time.
        """
        # Graph structure: ±20% integer variance
        for key in ['downstream_count', 'upstream_count', 'transitive_downstream',
                    'transitive_upstream', 'total_coupling', 'fan_in', 'fan_out']:
            if key in example:
                factor = np.random.uniform(0.8, 1.2)
                example[key] = max(0, int(example[key] * factor))

        # Centrality: ±25% float variance, clipped to [0,1]
        for key in ['betweenness_centrality', 'pagerank', 'in_degree_centrality',
                    'out_degree_centrality', 'complexity_score']:
            if key in example:
                factor = np.random.uniform(0.75, 1.25)
                example[key] = float(np.clip(example[key] * factor, 0.0, 1.0))

        # Operational metrics: independent variance per metric
        if 'error_rate' in example:
            example['error_rate'] = float(np.clip(
                example['error_rate'] * np.random.uniform(0.5, 2.0), 0.001, 0.05))

        if 'latency_p99_ms' in example:
            example['latency_p99_ms'] = float(np.clip(
                example['latency_p99_ms'] * np.random.uniform(0.7, 1.5), 100, 5000))

        if 'health_score' in example:
            example['health_score'] = float(np.clip(
                example['health_score'] + np.random.normal(0, 10), 0, 100))

        if 'requests_per_day' in example:
            example['requests_per_day'] = float(max(0,
                example['requests_per_day'] * np.random.uniform(0.8, 1.2)))

        return example

    def _add_change_scenario(self, example: Dict) -> Dict:
        """
        Add a change event to the scenario.
        Change type distribution is realistic:
        - Deployments are most common
        - Config changes are frequent but lower magnitude
        - Scaling events happen but are less frequent
        """
        change_type = np.random.choice(
            ['deployment', 'config_change', 'code_change', 'scaling'],
            p=[0.40, 0.30, 0.20, 0.10]
        )
        example['change_type'] = change_type

        # Change magnitude varies by type
        if change_type == 'deployment':
            # Deployments vary widely — small hotfixes to large releases
            example['change_magnitude'] = float(np.random.beta(2, 2))
        elif change_type == 'config_change':
            # Config changes tend to be smaller
            example['change_magnitude'] = float(np.random.beta(2, 5))
        elif change_type == 'code_change':
            # Code changes moderate to large
            example['change_magnitude'] = float(np.random.beta(3, 2))
        else:  # scaling
            # Scaling events are moderate
            example['change_magnitude'] = float(np.random.beta(2, 3))

        example['has_tests'] = int(np.random.random() > 0.3)
        example['deployment_frequency'] = int(np.random.uniform(1, 30))

        # Last incident: exponential distribution — most services haven't had recent incidents
        example['last_incident_days'] = int(np.random.exponential(30))

        return example

    # ──────────────────────────────────────────
    # Service classification
    # ──────────────────────────────────────────

    def _classify_service(self, features: Dict) -> str:
        """
        Classify service into one of 4 profiles based on structural position.
        This drives the label generation — different profiles have different failure modes.

        Profiles:
        - critical_hub: High centrality + high downstream. Failures cascade widely.
        - high_fan_out: Many downstream but not a shortest-path hub. Blast radius is wide.
        - leaf: Low coupling, edge of graph. Failures are isolated.
        - standard: Everything else.
        """
        centrality = features.get('betweenness_centrality', 0)
        downstream = features.get('downstream_count', 0)
        transitive = features.get('transitive_downstream', 0)

        if centrality >= CRITICAL_CENTRALITY and downstream >= HIGH_DOWNSTREAM:
            return 'critical_hub'
        elif downstream >= HIGH_DOWNSTREAM or transitive >= 10:
            return 'high_fan_out'
        elif downstream == 0 and features.get('upstream_count', 0) <= 1:
            return 'leaf'
        else:
            return 'standard'

    # ──────────────────────────────────────────
    # Label generation
    # ──────────────────────────────────────────

    def _calculate_incident_probability(self, features: Dict, profile: str) -> float:
        """
        Calculate incident probability using conditional interaction logic.

        Key insight: risk factors COMPOUND each other, they don't add linearly.
        A service that is BOTH high centrality AND has high error rate AND just had a
        large deployment is far more than 3x risky — it's the interaction that matters.

        Base probability comes from profile, then multiplied by risk factors.
        """

        # ── Base probability by service profile ──
        base_probs = {
            'critical_hub': 0.55,   # Critical hubs fail less often but when they do it's bad
            'high_fan_out': 0.40,
            'standard': 0.25,
            'leaf': 0.10,
        }
        prob = base_probs[profile]

        # ── Risk factor 1: Current health ──
        # Degraded error rate is a strong leading indicator
        error_rate = features.get('error_rate', 0.01)
        if error_rate > DEGRADED_ERROR_RATE:
            # Compound: already showing signs of trouble
            prob *= 1.8
        elif error_rate > 0.01:
            prob *= 1.3

        # ── Risk factor 2: Change event ──
        change_mag = features.get('change_magnitude', 0.0)
        change_type = features.get('change_type', 'deployment')
        has_tests = features.get('has_tests', 1)

        change_risk = 1.0
        if change_type == 'deployment' and change_mag > LARGE_CHANGE:
            change_risk = 1.6
        elif change_type == 'code_change' and change_mag > LARGE_CHANGE:
            change_risk = 1.5
        elif change_type == 'config_change' and change_mag > LARGE_CHANGE:
            change_risk = 1.3
        elif change_mag > 0.3:
            change_risk = 1.2

        # Tests reduce change risk — but don't eliminate it
        if not has_tests:
            change_risk *= 1.4

        prob *= change_risk

        # ── Risk factor 3: Recent incident history ──
        # Repeat incidents within a week are common (underlying cause often not fixed)
        last_incident = features.get('last_incident_days', 999)
        if last_incident <= RECENT_INCIDENT_DAYS:
            prob *= 1.7
        elif last_incident <= 14:
            prob *= 1.3

        # ── Risk factor 4: Deployment frequency ──
        # Very high frequency = more exposure; very low = less practiced
        freq = features.get('deployment_frequency', 10)
        if freq > 20:
            prob *= 1.2  # High velocity = more chances for something to go wrong
        elif freq < 3:
            prob *= 1.1  # Rare deploys = less practiced, risky when they happen

        # ── Risk factor 5: Latency degradation ──
        latency = features.get('latency_p99_ms', 300)
        if latency > 1500:
            prob *= 1.3  # Already slow — likely under stress

        # ── Clip and add realistic noise ──
        prob = np.clip(prob, 0.02, 0.97)
        noise = np.random.normal(0, 0.04)
        return float(np.clip(prob + noise, 0.0, 1.0))

    def _calculate_severity(self, features: Dict, profile: str,
                            incident_prob: float) -> float:
        """
        Severity = how bad the incident is IF it occurs.
        Driven by: structural position, blast radius, and current health.

        Uses beta distribution to add realistic spread:
        - Critical hubs: severity skewed high (beta(5,2))
        - Leaf services: severity skewed low (beta(2,5))
        """
        centrality = features.get('betweenness_centrality', 0)
        downstream = features.get('downstream_count', 0)
        complexity = features.get('complexity_score', 0)

        if profile == 'critical_hub':
            # High centrality services cause large, hard-to-isolate failures
            alpha, beta = 5, 2
            base = np.random.beta(alpha, beta)
        elif profile == 'high_fan_out':
            alpha, beta = 4, 3
            base = np.random.beta(alpha, beta)
        elif profile == 'leaf':
            alpha, beta = 2, 5
            base = np.random.beta(alpha, beta)
        else:
            alpha, beta = 3, 3
            base = np.random.beta(alpha, beta)

        # Adjust for actual feature values
        adjustment = (
            centrality * 0.15 +
            min(downstream / 10.0, 1.0) * 0.10 +
            complexity * 0.05
        )

        severity = np.clip(base + adjustment, 0.0, 1.0)
        return round(float(severity), 4)

    def _calculate_affected_services(self, features: Dict, profile: str) -> int:
        """
        Number of services affected by a failure.

        Modeled using the actual graph structure:
        - Direct downstream always affected
        - Transitive impact is partial (not all downstream services fully fail)
        - Critical hubs have higher transitive propagation rate
        """
        direct = features.get('downstream_count', 0)
        transitive = features.get('transitive_downstream', 0)

        if profile == 'critical_hub':
            # Most transitive services affected due to central position
            propagation_rate = np.random.uniform(0.6, 0.9)
        elif profile == 'high_fan_out':
            propagation_rate = np.random.uniform(0.3, 0.6)
        elif profile == 'leaf':
            propagation_rate = 0.0  # Leaf failures don't propagate
        else:
            propagation_rate = np.random.uniform(0.1, 0.4)

        affected = direct + int(transitive * propagation_rate)
        noise = int(np.random.normal(0, 1))
        return max(0, affected + noise)

    def _calculate_recovery_time(self, features: Dict, profile: str,
                                 severity: float, affected_services: int) -> int:
        """
        Recovery time in minutes.

        Three components:
        1. Detection time: how long to notice (depends on monitoring/error rate visibility)
        2. Isolation time: how long to find root cause (depends on complexity + coupling)
        3. Remediation time: how long to fix (depends on severity + affected count)

        Critical hubs take longer because the blast radius makes root cause harder to isolate.
        """
        # Detection: high error rate = faster detection
        error_rate = features.get('error_rate', 0.01)
        if error_rate > DEGRADED_ERROR_RATE:
            detection_minutes = int(np.random.uniform(2, 10))
        else:
            detection_minutes = int(np.random.uniform(5, 30))

        # Isolation: complexity + coupling makes root cause harder
        complexity = features.get('complexity_score', 0.3)
        coupling = features.get('total_coupling', 3)
        isolation_base = 10 + (complexity * 40) + (coupling * 3)
        isolation_minutes = int(np.random.normal(isolation_base, isolation_base * 0.2))
        isolation_minutes = max(5, isolation_minutes)

        # Remediation: scales with severity and affected count
        remediation_base = 15 + (severity * 90) + (affected_services * 10)
        remediation_minutes = int(np.random.normal(remediation_base, remediation_base * 0.25))
        remediation_minutes = max(5, remediation_minutes)

        total = detection_minutes + isolation_minutes + remediation_minutes
        return max(10, total)


# ──────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────

def generate_training_dataset(output_dir: str = 'data/training_data',
                              num_synthetic_services: int = 20,
                              samples_per_service: int = 15) -> pd.DataFrame:
    """
    Generate complete training dataset and save to CSV.

    Args:
        output_dir: Where to save the CSVs
        num_synthetic_services: How many synthetic services to create as base
        samples_per_service: How many scenario variations per service

    Returns:
        Complete training DataFrame
    """
    generator = TrainingDataGenerator()

    features_df = generator.generate_synthetic_project_data(num_synthetic_services)
    training_df = generator.generate_from_features(features_df, samples_per_service)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / 'training_data.csv'
    training_df.to_csv(train_file, index=False)
    print(f"\n💾 Saved training data: {train_file}")

    features_file = output_path / 'service_features.csv'
    features_df.to_csv(features_file, index=False)
    print(f"💾 Saved service features: {features_file}")

    print(f"\n📊 Training Data Statistics:")
    print(f"   Total examples:        {len(training_df)}")
    print(f"   Incident rate:         {training_df['label_incident_occurred'].mean()*100:.1f}%")
    print(f"   Avg severity:          {training_df['label_severity_score'].mean():.3f}")
    print(f"   Avg affected services: {training_df['label_affected_services'].mean():.1f}")
    print(f"   Avg recovery time:     {training_df['label_recovery_minutes'].mean():.1f} min")
    print(f"\n   Profile breakdown:")
    for profile in ['critical_hub', 'high_fan_out', 'standard', 'leaf']:
        count = sum(1 for _, row in features_df.iterrows()
                    if generator._classify_service(row.to_dict()) == profile)
        print(f"     {profile}: {count} services")

    return training_df


if __name__ == '__main__':
    # Load ONLY training projects (Bank of Anthos + TeaStore)
    # NOT Online Boutique — that's the hold-out test set
    bank_path = 'data/features/bank-of-anthos_features.csv'
    teastore_path = 'data/features/TeaStore_features.csv'

    print("📂 Loading training project features...")
    bank_features = pd.read_csv(bank_path)
    teastore_features = pd.read_csv(teastore_path)
    print(f"   bank-of-anthos: {len(bank_features)} services")
    print(f"   TeaStore:       {len(teastore_features)} services")

    combined = pd.concat([bank_features, teastore_features], ignore_index=True)
    print(f"   Combined:       {len(combined)} services")

    generator = TrainingDataGenerator()
    training_df = generator.generate_from_features(combined, samples_per_service=50)

    Path('data/training_data').mkdir(parents=True, exist_ok=True)
    training_df.to_csv('data/training_data/training_data.csv', index=False)
    print("\n✅ Training data saved to: data/training_data/training_data.csv")
    print("\n📋 Sample:")
    print(training_df[['service_name', 'downstream_count', 'betweenness_centrality',
                        'error_rate', 'change_type', 'change_magnitude',
                        'label_incident_occurred', 'label_severity_score',
                        'label_recovery_minutes']].head(5).to_string())