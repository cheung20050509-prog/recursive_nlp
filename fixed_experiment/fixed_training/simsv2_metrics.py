"""CH-SIMS / CH-SIMSv2 regression metrics aligned with KuDA (MKMaS-GUET/KuDA).

KuDA ``core/metric.py::__eval_sims_regression`` uses ``f1_score(preds, truth, ...)``
(sklearn expects y_true, y_pred first). This module uses the correct argument order.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _multiclass_acc_discrete(preds_cls: np.ndarray, truth_cls: np.ndarray) -> float:
    return float(np.mean(np.round(preds_cls) == np.round(truth_cls)))


def _bucket_continuous(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Assign class index i where edges[i] < v <= edges[i+1]. ``edges`` length K+1 for K bins."""
    out = np.zeros_like(values, dtype=np.int64)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = np.logical_and(values > lo, values <= hi)
        out[mask] = i
    return out


def compute_sims_regression_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Match KuDA SIMS/SIMSV2 eval (clip [-1,1], mult acc 2/3/5, MAE, Corr, weighted F1 on 2-bin)."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)

    test_preds = np.clip(preds, -1.0, 1.0)
    test_truth = np.clip(labels, -1.0, 1.0)

    ms_2 = [-1.01, 0.0, 1.01]
    test_preds_a2 = _bucket_continuous(test_preds, ms_2)
    test_truth_a2 = _bucket_continuous(test_truth, ms_2)

    ms_3 = [-1.01, -0.1, 0.1, 1.01]
    test_preds_a3 = _bucket_continuous(test_preds, ms_3)
    test_truth_a3 = _bucket_continuous(test_truth, ms_3)

    ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
    test_preds_a5 = _bucket_continuous(test_preds, ms_5)
    test_truth_a5 = _bucket_continuous(test_truth, ms_5)

    mae = float(np.mean(np.absolute(test_preds - test_truth)))
    corr = float(np.corrcoef(test_preds, test_truth)[0][1]) if test_preds.size > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    mult_a2 = _multiclass_acc_discrete(test_preds_a2.astype(np.float64), test_truth_a2.astype(np.float64))
    mult_a3 = _multiclass_acc_discrete(test_preds_a3.astype(np.float64), test_truth_a3.astype(np.float64))
    mult_a5 = _multiclass_acc_discrete(test_preds_a5.astype(np.float64), test_truth_a5.astype(np.float64))

    # sklearn: y_true first, y_pred second (KuDA had these reversed).
    f1 = float(f1_score(test_truth_a2, test_preds_a2, average="weighted", zero_division=0))

    return {
        "Mult_acc_2": round(mult_a2, 4),
        "Mult_acc_3": round(mult_a3, 4),
        "Mult_acc_5": round(mult_a5, 4),
        "F1_score": round(f1, 4),
        "MAE": round(mae, 4),
        "Corr": round(corr, 4),
        # Shared keys used by train.py / Optuna for selection and logging
        "mae": mae,
        "corr": corr,
    }
