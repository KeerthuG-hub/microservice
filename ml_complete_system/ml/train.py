"""
ML Model Trainer - Train Random Forest models for incident prediction.

Trains 3 models:
1. Incident Classifier: Will an incident occur? (binary classification)
2. Severity Regressor: How severe? (0.0 to 1.0)
3. Impact Regressor: How many services affected? (count)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, roc_auc_score
)
import sys

# Add project root to path (go up: ml/ → ml_complete_system/ → project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MLModelTrainer:
    """Train and evaluate ML models for incident prediction."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.models = {}
        self.feature_names = None
        self.metrics = {}
    
    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """Load training data from CSV."""
        print(f"\n📂 Loading training data from: {data_path}")
        df = pd.DataFrame(pd.read_csv(data_path))
        print(f"   ✅ Loaded {len(df)} training examples")
        return df
    
    def prepare_features_and_labels(self, df: pd.DataFrame) -> tuple:
        """
        Split data into features (X) and labels (y).
        
        Returns:
            X: Feature matrix
            y_incident: Binary labels (incident yes/no)
            y_severity: Severity scores (0-1)
            y_affected: Affected service counts
        """
        print("\n🔧 Preparing features and labels...")
        
        # Identify feature columns (not labels, not metadata)
        label_cols = [c for c in df.columns if c.startswith('label_')]
        meta_cols = ['service_name', 'project']
        
        feature_cols = [c for c in df.columns 
                       if c not in label_cols and c not in meta_cols]
        
        # Handle categorical features
        X = df[feature_cols].copy()
        
        # One-hot encode categorical features
        categorical_cols = ['language', 'change_type']
        for col in categorical_cols:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        
        # Extract labels
        y_incident = df['label_incident_occurred'].values
        y_severity = df['label_severity_score'].values
        y_affected = df['label_affected_services'].values
        
        self.feature_names = list(X.columns)
        
        print(f"   ✅ Features: {len(X.columns)}")
        print(f"   ✅ Samples: {len(X)}")
        print(f"   ✅ Incident rate: {y_incident.mean()*100:.1f}%")
        
        return X, y_incident, y_severity, y_affected
    
    def train_incident_classifier(self, X_train, y_train, X_test, y_test):
        """Train binary classifier: will incident occur?"""
        print("\n🤖 Training Incident Classifier...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # handles class imbalance — improves recall
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}")
        print(f"   ✅ Precision: {metrics['precision']:.3f}")
        print(f"   ✅ Recall: {metrics['recall']:.3f}")
        print(f"   ✅ F1 Score: {metrics['f1']:.3f}")
        print(f"   ✅ ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"   ✅ CV F1: {metrics['cv_f1_mean']:.3f} (±{metrics['cv_f1_std']:.3f})")
        
        self.models['incident_classifier'] = model
        self.metrics['incident_classifier'] = metrics
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   📊 Top 5 features:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        return model, metrics
    
    def train_severity_regressor(self, X_train, y_train, X_test, y_test):
        """Train regressor: how severe will incident be?"""
        print("\n🤖 Training Severity Regressor...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        print(f"   ✅ MAE: {metrics['mae']:.3f}")
        print(f"   ✅ RMSE: {metrics['rmse']:.3f}")
        print(f"   ✅ R²: {metrics['r2']:.3f}")
        print(f"   ✅ CV R²: {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
        
        self.models['severity_regressor'] = model
        self.metrics['severity_regressor'] = metrics
        
        return model, metrics
    
    def train_impact_regressor(self, X_train, y_train, X_test, y_test):
        """Train regressor: how many services will be affected?"""
        print("\n🤖 Training Impact Regressor...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        print(f"   ✅ MAE: {metrics['mae']:.3f} services")
        print(f"   ✅ RMSE: {metrics['rmse']:.3f} services")
        print(f"   ✅ R²: {metrics['r2']:.3f}")
        print(f"   ✅ CV R²: {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
        
        self.models['impact_regressor'] = model
        self.metrics['impact_regressor'] = metrics
        
        return model, metrics
    
    def save_models(self, output_dir: str = 'data/models'):
        """Save trained models to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving models to: {output_dir}")
        
        for model_name, model in self.models.items():
            model_file = output_path / f'{model_name}.pkl'
            joblib.dump(model, model_file)
            print(f"   ✅ Saved: {model_file}")
        
        # Save feature names
        feature_file = output_path / 'feature_names.pkl'
        joblib.dump(self.feature_names, feature_file)
        print(f"   ✅ Saved: {feature_file}")
        
        # Save metrics
        metrics_file = output_path / 'training_metrics.pkl'
        joblib.dump(self.metrics, metrics_file)
        print(f"   ✅ Saved: {metrics_file}")
    
    def train_all(self, training_data_path: str, test_size: float = 0.2):
        """
        Complete training pipeline: load data, train all models, save.
        
        Args:
            training_data_path: Path to training CSV
            test_size: Fraction of data for testing
        """
        print("=" * 70)
        print("🚀 ML MODEL TRAINING PIPELINE")
        print("=" * 70)
        
        # Load data
        df = self.load_training_data(training_data_path)
        
        # Prepare features and labels
        X, y_incident, y_severity, y_affected = self.prepare_features_and_labels(df)
        
        # Train/test split for incident classifier (all rows)
        print(f"\n✂️  Splitting data: {100-test_size*100:.0f}% train, {test_size*100:.0f}% test")
        X_train, X_test, y_inc_train, y_inc_test = train_test_split(
            X, y_incident, test_size=test_size, random_state=self.random_seed, stratify=y_incident
        )

        print(f"   Train: {len(X_train)} samples")
        print(f"   Test:  {len(X_test)} samples")

        # For severity and impact regressors: ONLY use rows where incident occurred
        # Training on non-incident rows adds noise — severity of 0.7 where no incident
        # happened is meaningless
        incident_mask = y_incident == 1
        X_incident = X[incident_mask]
        y_severity_incident = y_severity[incident_mask]
        y_affected_incident = y_affected[incident_mask]

        print(f"\n   Incident rows for regressor training: {incident_mask.sum()} "
              f"({incident_mask.mean()*100:.1f}% of data)")

        X_inc_train, X_inc_test, y_sev_train, y_sev_test, y_aff_train, y_aff_test = train_test_split(
            X_incident,
            y_severity_incident,
            y_affected_incident,
            test_size=test_size,
            random_state=self.random_seed
        )

        # Train models
        self.train_incident_classifier(X_train, y_inc_train, X_test, y_inc_test)
        self.train_severity_regressor(X_inc_train, y_sev_train, X_inc_test, y_sev_test)
        self.train_impact_regressor(X_inc_train, y_aff_train, X_inc_test, y_aff_test)
        
        # Save models
        self.save_models()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        
        return self.models, self.metrics


def main():
    """Main training script."""
    trainer = MLModelTrainer(random_seed=42)
    
    # Train on generated data
    models, metrics = trainer.train_all(
        training_data_path='data/training_data/training_data.csv',
        test_size=0.2
    )
    
    print("\n📊 Final Model Summary:")
    print(f"   Incident Classifier F1: {metrics['incident_classifier']['f1']:.3f}")
    print(f"   Severity Regressor R²: {metrics['severity_regressor']['r2']:.3f}")
    print(f"   Impact Regressor R²: {metrics['impact_regressor']['r2']:.3f}")
    
    print("\n🎉 Models ready for prediction!")
    print("   Use: joblib.load('data/models/incident_classifier.pkl')")


if __name__ == '__main__':
    main()