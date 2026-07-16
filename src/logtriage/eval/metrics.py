"""Evaluation metrics — the heart of the project's validation discipline.

Positives are ~0.1-3% of blocks, so accuracy and ROC-AUC are misleading. The
metrics that matter:

  * PR-AUC (average precision) — discrimination under heavy class imbalance.
  * False alarms per day at a fixed detection rate — the operational number an
    on-call analyst lives with. Pick a target recall, find the score threshold
    that achieves it, count the false positives, divide by real elapsed days.
  * Drift — the same numbers computed over consecutive slices of the test
    period, so degradation over time can't hide inside an aggregate.

All functions take `scores` where higher = more anomalous.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(labels, scores))


@dataclass
class OperatingPoint:
    target_recall: float
    threshold: float
    recall: float
    precision: float
    tp: int
    fp: int
    fn: int


def operating_point_at_recall(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> OperatingPoint:
    """Smallest set of top-scored blocks achieving recall >= target_recall.

    Ties in score are resolved by sort order (negligible at these volumes).
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    P = int(labels.sum())
    if P == 0:
        raise ValueError("no positives in labels — cannot fix a recall target")

    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    recall = tp / P
    k = int(np.searchsorted(recall, target_recall))  # recall is nondecreasing
    k = min(k, len(scores) - 1)

    tp_k = int(tp[k])
    fp_k = int((k + 1) - tp_k)
    return OperatingPoint(
        target_recall=target_recall,
        threshold=float(scores[order][k]),
        recall=float(recall[k]),
        precision=float(tp_k / (k + 1)),
        tp=tp_k,
        fp=fp_k,
        fn=P - tp_k,
    )


def false_alarms_per_day(
    labels: np.ndarray,
    scores: np.ndarray,
    timestamps: pd.Series,
    target_recall: float = 0.90,
) -> tuple[float, OperatingPoint]:
    """False positives per day at a fixed detection rate — the headline metric.

    `timestamps` are the block start times of the evaluated set; elapsed days is
    their real span (blocks are not uniform in time, so this must be measured,
    not assumed).
    """
    op = operating_point_at_recall(labels, scores, target_recall)
    span = pd.to_datetime(timestamps).agg(["min", "max"])
    n_days = (span["max"] - span["min"]).total_seconds() / 86400
    n_days = max(n_days, 1e-9)
    return op.fp / n_days, op


def drift_report(
    labels: np.ndarray,
    scores: np.ndarray,
    timestamps: pd.Series,
    n_bins: int = 4,
    target_recall: float = 0.90,
) -> pd.DataFrame:
    """PR-AUC, anomaly rate, and FA/day over consecutive equal-time slices.

    A single test-set aggregate can hide performance decaying over time; this
    breaks it into `n_bins` chronological windows. The detection threshold is
    fixed once on the whole test set, then applied per bin (an analyst can't
    retune per hour).
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)

    op = operating_point_at_recall(labels, scores, target_recall)
    edges = pd.date_range(ts.min(), ts.max(), periods=n_bins + 1)

    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (ts >= lo) & (ts <= hi if i == n_bins - 1 else ts < hi)
        yl, sc = labels[m.to_numpy()], scores[m.to_numpy()]
        if len(yl) == 0:
            continue
        days = max((hi - lo).total_seconds() / 86400, 1e-9)
        flagged = sc >= op.threshold
        fp = int(((flagged) & (yl == 0)).sum())
        rows.append(
            {
                "bin": i + 1,
                "start": lo,
                "n_blocks": int(len(yl)),
                "anomaly_rate": float(yl.mean()),
                "pr_auc": pr_auc(yl, sc) if yl.sum() and yl.sum() < len(yl) else float("nan"),
                "recall_at_thr": float((flagged & (yl == 1)).sum() / max(yl.sum(), 1)),
                "false_alarms_per_day": fp / days,
            }
        )
    return pd.DataFrame(rows)
