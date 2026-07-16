"""Fit the Isolation Forest baseline and evaluate it with the project metrics.

    python scripts/run_baseline.py

Reads data/processed/ (from build_features.py), fits on the train split, scores
the test split, and reports PR-AUC, false alarms per day at fixed recall, and a
drift breakdown across the test period. Saves test scores for later comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from logtriage.config import DATA_PROCESSED
from logtriage.eval.metrics import drift_report, false_alarms_per_day, pr_auc
from logtriage.models.baseline import IsolationForestBaseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procdir", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--target-recall", type=float, default=0.90)
    args = ap.parse_args()

    X = np.load(args.procdir / "hdfs_counts.npy")
    meta = pd.read_csv(
        args.procdir / "hdfs_meta.csv", index_col=0, parse_dates=["start", "end"]
    )
    assert len(X) == len(meta), "counts / meta misaligned"

    is_train = (meta["split"] == "train").to_numpy()
    is_test = ~is_train
    y_test = meta.loc[is_test, "label"].to_numpy()
    ts_test = meta.loc[is_test, "start"]

    print(f"train {is_train.sum():,} blocks | test {is_test.sum():,} blocks")
    print("fitting Isolation Forest (unsupervised, no labels) ...")
    model = IsolationForestBaseline().fit(X[is_train])
    scores_test = model.score(X[is_test])

    ap_score = pr_auc(y_test, scores_test)
    fa_day, op = false_alarms_per_day(
        y_test, scores_test, ts_test, target_recall=args.target_recall
    )

    print("\n" + "=" * 56)
    print("ISOLATION FOREST BASELINE — TEST")
    print("=" * 56)
    print(f"  PR-AUC (average precision)     : {ap_score:.4f}")
    print(f"  test anomaly rate (chance PR)  : {y_test.mean():.4f}")
    print(f"  @ recall {op.recall:.2f}:")
    print(f"      precision                  : {op.precision:.4f}")
    print(f"      false alarms               : {op.fp:,}")
    print(f"      FALSE ALARMS / DAY         : {fa_day:,.1f}")
    print("=" * 56)

    print("\nDRIFT across test period (threshold fixed on full test):")
    drift = drift_report(y_test, scores_test, ts_test, n_bins=4,
                         target_recall=args.target_recall)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(drift.to_string(index=False))

    np.save(args.procdir / "baseline_scores_test.npy", scores_test)
    (args.procdir / "baseline_result.json").write_text(
        json.dumps(
            {
                "model": "IsolationForest",
                "pr_auc": ap_score,
                "target_recall": args.target_recall,
                "false_alarms_per_day": fa_day,
                "false_positives": op.fp,
                "precision_at_recall": op.precision,
            },
            indent=2,
        )
    )
    print(f"\nsaved test scores + result to {args.procdir}")


if __name__ == "__main__":
    main()
