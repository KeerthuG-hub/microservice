"""
Evaluator - measures accuracy when ground truth is available.
Reports precision, recall, F1 per dependency type and overall.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from agent.models import AnalysisResult, Dependency


class Evaluator:

    def __init__(self, result: AnalysisResult, ground_truth_path: str):
        self.result = result
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth: List[Tuple[str, str]] = []

    def evaluate(self) -> Dict:
        if not self.ground_truth_path.exists():
            print(f"⚠️  Ground truth not found: {self.ground_truth_path}")
            return {}

        try:
            data = json.loads(self.ground_truth_path.read_text())
            self.ground_truth = [(e[0], e[1]) for e in data.get('edges', [])]
        except Exception as e:
            print(f"⚠️  Failed to load ground truth: {e}")
            return {}

        print(f"\n📏 Evaluating against {len(self.ground_truth)} ground truth edges...")

        # All predictions at confidence >= 0.65 (functional only)
        predictions = [
            (d.from_service, d.to_service)
            for d in self.result.dependencies
            if d.confidence >= 0.65
            and d.dep_type not in ('observability',)
            and d.source not in ('string_cooccurrence', 'git_change_coupling')
        ]

        # Normalize for comparison
        gt_norm = {(self._n(a), self._n(b)) for a, b in self.ground_truth}
        pred_norm = {(self._n(a), self._n(b)) for a, b in predictions}

        tp = gt_norm & pred_norm
        fp = pred_norm - gt_norm
        fn = gt_norm - pred_norm

        precision = len(tp) / max(len(pred_norm), 1)
        recall = len(tp) / max(len(gt_norm), 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-9))

        # Per-type breakdown
        type_metrics = {}
        dep_types = set(d.dep_type for d in self.result.dependencies)
        for dep_type in dep_types:
            type_preds = {
                (self._n(d.from_service), self._n(d.to_service))
                for d in self.result.dependencies
                if d.dep_type == dep_type and d.confidence >= 0.65
            }
            type_tp = gt_norm & type_preds
            type_fp = type_preds - gt_norm
            type_fn = gt_norm - type_preds

            t_prec = len(type_tp) / max(len(type_preds), 1)
            t_rec = len(type_tp) / max(len(gt_norm), 1)
            t_f1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-9)
            type_metrics[dep_type] = {
                'precision': round(t_prec, 3),
                'recall': round(t_rec, 3),
                'f1': round(t_f1, 3),
                'tp': len(type_tp),
                'fp': len(type_fp),
                'fn': len(type_fn),
            }

        results = {
            'overall': {
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1': round(f1, 3),
                'tp': len(tp),
                'fp': len(fp),
                'fn': len(fn),
                'gt_size': len(gt_norm),
                'pred_size': len(pred_norm),
            },
            'per_type': type_metrics,
            'true_positives': sorted([(a, b) for a, b in tp]),
            'false_positives': sorted([(a, b) for a, b in fp]),
            'false_negatives': sorted([(a, b) for a, b in fn]),
        }

        self._print_results(results)
        return results

    def _print_results(self, r: Dict):
        ov = r['overall']
        print(f"\n{'='*50}")
        print(f"📊 EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"   Precision : {ov['precision']:.3f}")
        print(f"   Recall    : {ov['recall']:.3f}")
        print(f"   F1        : {ov['f1']:.3f}")
        print(f"   TP={ov['tp']} FP={ov['fp']} FN={ov['fn']}")
        print(f"\n   Ground truth: {ov['gt_size']} edges")
        print(f"   Predicted:    {ov['pred_size']} edges")

        if r.get('false_negatives'):
            print(f"\n   ❌ Missed ({len(r['false_negatives'])}):")
            for a, b in r['false_negatives'][:10]:
                print(f"      {a} → {b}")

        if r.get('false_positives'):
            print(f"\n   ⚠️  Extra ({len(r['false_positives'])}):")
            for a, b in r['false_positives'][:10]:
                print(f"      {a} → {b}")
        print(f"{'='*50}")

    def _n(self, name: str) -> str:
        return name.lower().replace('-', '').replace('_', '').replace('.', '')
