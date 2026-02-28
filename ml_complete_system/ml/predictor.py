"""
ML Predictor - Use trained models to predict incident impact.

Loads trained models and makes predictions on new service changes.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class IncidentPredictor:
    """Predict incident impact using trained ML models."""
    
    def __init__(self, models_dir: str = 'data/models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = None
        self._load_models()
    
    def _load_models(self):
        """Load all trained models from disk."""
        print(f"📦 Loading models from: {self.models_dir}")
        
        model_files = {
            'incident_classifier': 'incident_classifier.pkl',
            'severity_regressor': 'severity_regressor.pkl',
            'impact_regressor': 'impact_regressor.pkl',
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"   ✅ Loaded: {model_name}")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load feature names
        feature_path = self.models_dir / 'feature_names.pkl'
        if feature_path.exists():
            self.feature_names = joblib.load(feature_path)
            print(f"   ✅ Loaded feature names: {len(self.feature_names)} features")
        else:
            raise FileNotFoundError(f"Feature names not found: {feature_path}")
    
    def predict(self, service_features: Dict, change_type: str = 'deployment',
                change_magnitude: float = 0.5) -> Dict:
        """
        Predict incident impact for a service change.
        
        Args:
            service_features: Dictionary of service features from feature engineering
            change_type: Type of change ('deployment', 'config_change', 'code_change', 'scaling')
            change_magnitude: Magnitude of change (0.0 to 1.0)
            
        Returns:
            Dictionary with predictions and recommendations
        """
        # Add change features
        features = service_features.copy()
        features['change_type'] = change_type
        features['change_magnitude'] = change_magnitude
        
        # Add default deployment features if not present
        if 'has_tests' not in features:
            features['has_tests'] = 1
        if 'deployment_frequency' not in features:
            features['deployment_frequency'] = 10
        if 'last_incident_days' not in features:
            features['last_incident_days'] = 30
        
        # Prepare feature vector
        X = self._prepare_features(features)
        
        # Make predictions
        predictions = {}
        
        # 1. Incident probability
        incident_proba = self.models['incident_classifier'].predict_proba(X)[0][1]
        predictions['incident_probability'] = float(incident_proba)
        predictions['incident_probability_percent'] = f"{incident_proba*100:.1f}%"
        predictions['will_incident_occur'] = incident_proba > 0.5
        
        # 2. Severity — graph-based blast radius (not synthetic ML labels)
        # fan_out + transitive_downstream give real impact if this service fails.
        fan_out         = features.get('fan_out', 0)
        transitive_down = features.get('transitive_downstream', 0)
        betweenness     = features.get('betweenness_centrality', 0)

        blast_score = min(1.0, (fan_out * 0.15) + (transitive_down * 0.05) + (betweenness * 2.0))

        if blast_score >= 0.6:
            predictions['severity'] = 'CRITICAL'
        elif blast_score >= 0.35:
            predictions['severity'] = 'HIGH'
        elif blast_score >= 0.15:
            predictions['severity'] = 'MEDIUM'
        else:
            predictions['severity'] = 'LOW'

        predictions['severity_score'] = round(blast_score, 3)

        # 3. Affected services
        try:
            affected_count = max(0, int(round(self.models['impact_regressor'].predict(X)[0])))
        except Exception:
            affected_count = int(fan_out + transitive_down * 0.5)
        predictions['estimated_affected_services'] = affected_count
        
        # 4. Recovery time — based on blast_score + affected services
        base_recovery   = 30
        severity_factor = blast_score * 180   # 0–180 min based on blast radius
        affected_factor = affected_count * 20  # 20 min per affected service
        recovery_minutes = int(base_recovery + severity_factor + affected_factor)
        predictions['estimated_recovery_minutes'] = recovery_minutes
        
        # 5. Risk level
        if incident_proba >= 0.7:
            predictions['risk_level'] = 'HIGH'
            predictions['recommendation'] = "⚠️  HIGH RISK: Deploy with extreme caution. Have rollback plan ready."
        elif incident_proba >= 0.5:
            predictions['risk_level'] = 'MEDIUM'
            predictions['recommendation'] = "⚡ MEDIUM RISK: Deploy during low traffic window. Monitor closely."
        elif incident_proba >= 0.3:
            predictions['risk_level'] = 'LOW'
            predictions['recommendation'] = "✓ LOW RISK: Safe to deploy. Standard monitoring sufficient."
        else:
            predictions['risk_level'] = 'MINIMAL'
            predictions['recommendation'] = "✓ MINIMAL RISK: Deploy normally."
        

        original_severity = predictions['severity']

        if incident_proba >= 0.85 and predictions['severity'] in ('LOW', 'MEDIUM'):
            predictions['severity'] = 'HIGH'
            predictions['severity_boosted'] = True
        elif incident_proba >= 0.7 and predictions['severity'] == 'LOW':
            predictions['severity'] = 'MEDIUM'
            predictions['severity_boosted'] = True

        
        # Add input features for reference
        predictions['service_name'] = features.get('service_name', 'unknown')
        predictions['change_type'] = change_type
        predictions['change_magnitude'] = change_magnitude
        
        return predictions
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """
        Convert feature dict to numpy array matching training feature order.

        Critical: at inference time, we must:
        1. One-hot encode categoricals the same way training did
        2. Add any missing columns as 0 (unseen categories)
        3. Reorder columns to exactly match saved feature_names
        Without step 2+3 this will crash on real projects.
        """
        # Drop metadata columns
        exclude = {'service_name', 'project', 'label_incident_occurred',
                   'label_incident_probability', 'label_severity_score',
                   'label_affected_services', 'label_recovery_minutes'}
        clean = {k: v for k, v in features.items() if k not in exclude}

        # Build single-row DataFrame
        df = pd.DataFrame([clean])

        # One-hot encode categoricals — same as training
        categorical_cols = ['language', 'change_type']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # Align to training feature space:
        # - Add missing columns with 0 (unseen category or missing feature)
        # - Drop extra columns (features that weren't in training)
        # - Reorder to exact training column order
        # Return DataFrame (not .values) so sklearn gets named features → no warnings
        return df.reindex(columns=self.feature_names, fill_value=0)
    
    def predict_batch(self, services_features: List[Dict],
                     change_type: str = 'deployment',
                     change_magnitude: float = 0.5) -> pd.DataFrame:
        """
        Predict for multiple services at once.
        
        Args:
            services_features: List of service feature dictionaries
            change_type: Type of change
            change_magnitude: Magnitude of change
            
        Returns:
            DataFrame with predictions for all services
        """
        predictions = []
        
        for service_feat in services_features:
            pred = self.predict(service_feat, change_type, change_magnitude)
            predictions.append(pred)
        
        return pd.DataFrame(predictions)


def demo_prediction():
    """Demonstrate prediction on sample service."""
    print("\n" + "=" * 70)
    print("🔮 INCIDENT PREDICTION DEMO")
    print("=" * 70)
    
    predictor = IncidentPredictor()
    
    # Sample service features
    sample_service = {
        'service_name': 'payment-service',
        'language': 'python',
        'upstream_count': 2,
        'downstream_count': 5,
        'transitive_downstream': 8,
        'transitive_upstream': 3,
        'total_coupling': 7,
        'betweenness_centrality': 0.35,
        'pagerank': 0.15,
        'in_degree_centrality': 0.2,
        'out_degree_centrality': 0.5,
        'fan_in': 2,
        'fan_out': 5,
        'coupling_ratio': 2.5,
        'max_path_depth': 3,
        'error_rate': 0.02,
        'latency_p99_ms': 850,
        'requests_per_day': 50000,
        'health_score': 85,
        'complexity_score': 0.45,
        'has_tests': 1,
        'deployment_frequency': 15,
        'last_incident_days': 45,
    }
    
    print("\n📋 Predicting impact for service: payment-service")
    print(f"   Scenario: Deployment with magnitude 0.8")
    
    prediction = predictor.predict(
        sample_service,
        change_type='deployment',
        change_magnitude=0.8
    )
    
    print("\n" + "-" * 70)
    print("📊 PREDICTION RESULTS")
    print("-" * 70)
    print(f"\n🎯 Incident Probability: {prediction['incident_probability_percent']}")
    print(f"   Will incident occur? {prediction['will_incident_occur']}")
    print(f"\n⚠️  Severity: {prediction['severity']} ({prediction['severity_score']:.3f})")
    print(f"\n📉 Impact:")
    print(f"   Estimated affected services: {prediction['estimated_affected_services']}")
    print(f"   Estimated recovery time: {prediction['estimated_recovery_minutes']} minutes")
    print(f"\n🚦 Risk Level: {prediction['risk_level']}")
    print(f"\n💡 Recommendation:")
    print(f"   {prediction['recommendation']}")
    print("\n" + "=" * 70)
    
    return prediction


if __name__ == '__main__':
    import sys

    project = sys.argv[1] if len(sys.argv) > 1 else 'online-boutique'
    features_path = f'data/features/{project}_features.csv'

    print(f"\n{'='*70}")
    print(f"🔮 {project.upper()} — BATCH INCIDENT PREDICTION")
    print(f"{'='*70}")

    if not Path(features_path).exists():
        print(f"❌ Features not found: {features_path}")
        print("   Run: python feature_engineering.py data/test_projects/<project>")
        sys.exit(1)

    features_df = pd.read_csv(features_path)
    print(f"\n📂 Loaded {len(features_df)} services from: {features_path}")

    predictor = IncidentPredictor()
    results = predictor.predict_batch(
        features_df.to_dict('records'),
        change_type='deployment',
        change_magnitude=0.5
    )

    # Sort by incident probability
    if 'incident_probability' in results.columns:
        results = results.sort_values('incident_probability', ascending=False)

    print("\n📊 PREDICTION RESULTS (sorted by risk):")
    display_cols = ['service_name', 'incident_probability_percent', 'risk_level',
                    'severity', 'estimated_affected_services',
                    'estimated_recovery_minutes', 'recommendation']
    available = [c for c in display_cols if c in results.columns]
    print(results[available].to_string(index=False))

    # Save CSV
    output_dir = Path('data/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{project}_predictions.csv'
    results.to_csv(output_path, index=False)
    print(f"\n💾 CSV saved: {output_path}")

    # Save JSON for NLP + UI layer
    json_path = output_dir / 'predictions.json'
    results.to_json(json_path, orient='records', indent=2)
    print(f"💾 JSON saved: {json_path}")
    print("   NLP layer and UI will use this file automatically.")