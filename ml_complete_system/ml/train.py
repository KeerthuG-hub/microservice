"""
ML Model Trainer

CHANGES from original:
1. REMOVED impact_regressor (train_impact_regressor, y_affected)
   label_affected_services is no longer an ML target — cascade_engine.py
   computes affected services deterministically from graph traversal.

2. prepare_features_and_labels() returns (X, y_incident, y_severity) only.
   train_all() no longer unpacks or trains on y_affected.

3. load_training_data() handles legacy CSVs gracefully:
   - renames latency_p99_ms → latency_p99 if old column present
   - renames requests_per_day → traffic_requests_per_day if old column present
   - rescales health_score 0-100 → 0.0-1.0 if needed

Trains 2 models:
  1. incident_classifier  — P(incident occurred)  [binary classification]
  2. severity_regressor   — severity score 0-1     [regression]
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
    roc_auc_score,
)
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MLModelTrainer:
    """Train and evaluate ML models for incident prediction."""

    def __init__(self, random_seed: int = 42):
        self.random_seed   = random_seed
        self.models        = {}
        self.feature_names = None
        self.metrics       = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """Load CSV and apply column alignment for legacy files."""
        print(f"\n📂 Loading training data from: {data_path}")
        df = pd.read_csv(data_path)

        # Handle CSVs written before the column rename fix
        rename_map = {}
        if 'latency_p99_ms' in df.columns and 'latency_p99' not in df.columns:
            rename_map['latency_p99_ms'] = 'latency_p99'
        if 'requests_per_day' in df.columns and 'traffic_requests_per_day' not in df.columns:
            rename_map['requests_per_day'] = 'traffic_requests_per_day'
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            print(f"   ⚠️  Renamed legacy columns: {rename_map}")

        if 'health_score' in df.columns and df['health_score'].max() > 1.5:
            df['health_score'] = (df['health_score'] / 100.0).round(4)
            print(f"   ⚠️  Rescaled health_score 0-100 → 0-1")

        print(f"   ✅ Loaded {len(df)} training examples")
        return df

    # ── Feature / label preparation ───────────────────────────────────────────

    def prepare_features_and_labels(self, df: pd.DataFrame) -> tuple:
        """
        Returns (X, y_incident, y_severity).
        y_affected intentionally removed — cascade_engine.py handles that.
        """
        print("\n🔧 Preparing features and labels...")

        label_cols   = [c for c in df.columns if c.startswith('label_')]
        meta_cols    = ['service_name', 'project']
        feature_cols = [c for c in df.columns if c not in label_cols and c not in meta_cols]

        X = df[feature_cols].copy()

        for col in ['language', 'change_type']:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

        y_incident = df['label_incident_occurred'].values
        y_severity = df['label_severity_score'].values
        # label_affected_services intentionally NOT extracted

        self.feature_names = list(X.columns)

        print(f"   ✅ Features: {len(X.columns)}")
        print(f"   ✅ Samples:  {len(X)}")
        print(f"   ✅ Incident rate: {y_incident.mean()*100:.1f}%")

        return X, y_incident, y_severity

    # ── Model training ────────────────────────────────────────────────────────

    def train_incident_classifier(self, X_train, y_train, X_test, y_test):
        """Binary classifier: will an incident occur?"""
        print("\n🤖 Training Incident Classifier...")

        model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_seed, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_pred_proba),
        }
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = cv.mean()
        metrics['cv_f1_std']  = cv.std()

        print(f"   ✅ Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"   ✅ CV F1: {metrics['cv_f1_mean']:.3f} (±{metrics['cv_f1_std']:.3f})")

        fi = (pd.DataFrame({'feature': self.feature_names, 'importance': model.feature_importances_})
              .sort_values('importance', ascending=False))
        print(f"   📊 Top 5: {', '.join(fi['feature'].head(5).tolist())}")

        self.models['incident_classifier'] = model
        self.metrics['incident_classifier'] = metrics
        return model, metrics

    def train_severity_regressor(self, X_train, y_train, X_test, y_test):
        """Regressor: how severe will the incident be? (0.0-1.0)"""
        print("\n🤖 Training Severity Regressor...")

        model = RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            random_state=self.random_seed, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        metrics = {
            'mae':  mean_absolute_error(y_test, y_pred),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2':   r2_score(y_test, y_pred),
        }
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv.mean()
        metrics['cv_r2_std']  = cv.std()

        print(f"   ✅ MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f}")
        print(f"   ✅ CV R²: {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")

        self.models['severity_regressor'] = model
        self.metrics['severity_regressor'] = metrics
        return model, metrics

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_models(self, output_dir: str = 'data/models'):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving models to: {output_dir}")
        for name, model in self.models.items():
            joblib.dump(model, output_path / f'{name}.pkl')
            print(f"   ✅ {name}.pkl")

        joblib.dump(self.feature_names, output_path / 'feature_names.pkl')
        joblib.dump(self.metrics,       output_path / 'training_metrics.pkl')
        print(f"   ✅ feature_names.pkl")
        print(f"   ✅ training_metrics.pkl")

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def train_all(self, training_data_path: str, test_size: float = 0.2):
        print("=" * 70)
        print("🚀 ML MODEL TRAINING PIPELINE")
        print("   ✅ incident_classifier   — will incident occur?")
        print("   ✅ severity_regressor    — how severe?")
        print("   ❌ impact_regressor      — REMOVED (cascade_engine.py handles this)")
        print("=" * 70)

        df = self.load_training_data(training_data_path)
        X, y_incident, y_severity = self.prepare_features_and_labels(df)

        print(f"\n✂️  Splitting: {100-test_size*100:.0f}% train / {test_size*100:.0f}% test")
        X_train, X_test, y_inc_tr, y_inc_te = train_test_split(
            X, y_incident, test_size=test_size,
            random_state=self.random_seed, stratify=y_incident,
        )
        print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

        # Train regressors on incident-only rows
        # (severity=0 for non-incidents adds noise to regression)
        mask   = y_incident == 1
        X_inc  = X[mask]
        y_sev  = y_severity[mask]
        print(f"\n   Incident rows for regressor: {mask.sum()} ({mask.mean()*100:.1f}%)")

        X_inc_tr, X_inc_te, y_sev_tr, y_sev_te = train_test_split(
            X_inc, y_sev, test_size=test_size, random_state=self.random_seed,
        )

        self.train_incident_classifier(X_train, y_inc_tr, X_test, y_inc_te)
        self.train_severity_regressor(X_inc_tr, y_sev_tr, X_inc_te, y_sev_te)
        self.save_models()

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        return self.models, self.metrics


def main():
    trainer = MLModelTrainer(random_seed=42)
    models, metrics = trainer.train_all('data/training_data/training_data.csv')
    print(f"\n📊 Incident Classifier F1: {metrics['incident_classifier']['f1']:.3f}")
    print(f"   Severity Regressor  R²: {metrics['severity_regressor']['r2']:.3f}")
    print("\n🎉 Load with: joblib.load('data/models/incident_classifier.pkl')")


if __name__ == '__main__':
    main()