"""Train the autoencoder detector on HDFS and evaluate it against the baseline.

    python scripts/train_autoencoder.py

Trains on NORMAL train-split blocks only, scores the test split, and reports the
same metrics as the baseline (PR-AUC, false alarms/day, drift) so the comparison
is apples-to-apples. Saves test scores and a result JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from logtriage.config import DATA_PROCESSED
from logtriage.eval.metrics import drift_report, false_alarms_per_day, pr_auc
from logtriage.models.autoencoder import AutoEncoderDetector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procdir", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(args.procdir / "hdfs_counts.npy")
    meta = pd.read_csv(
        args.procdir / "hdfs_meta.csv", index_col=0, parse_dates=["start", "end"]
    )
    is_train = (meta["split"] == "train").to_numpy()
    is_test = ~is_train
    y = meta["label"].to_numpy()

    # train on NORMAL train blocks only (unsupervised: labels pick the clean set)
    normal_train = is_train & (y == 0)
    print(f"train-normal {normal_train.sum():,} | test {is_test.sum():,} blocks")

    model = AutoEncoderDetector(epochs=args.epochs, seed=args.seed).fit(X[normal_train])
    print(f"trained {len(model.history_)} epochs "
          f"(val MSE {model.history_[-1][1]:.4f})")

    scores_test = model.score(X[is_test])
    y_test = y[is_test]
    ts_test = meta.loc[is_test, "start"]

    ap_score = pr_auc(y_test, scores_test)
    fa_day, op = false_alarms_per_day(
        y_test, scores_test, ts_test, target_recall=args.target_recall
    )

    # baseline for comparison, if present
    base = {}
    bpath = args.procdir / "baseline_result.json"
    if bpath.exists():
        base = json.loads(bpath.read_text())

    print("\n" + "=" * 56)
    print("AUTOENCODER — TEST")
    print("=" * 56)
    print(f"  PR-AUC (average precision)     : {ap_score:.4f}"
          + (f"   (baseline {base['pr_auc']:.4f})" if base else ""))
    print(f"  test anomaly rate (chance PR)  : {y_test.mean():.4f}")
    print(f"  @ recall {op.recall:.2f}:")
    print(f"      precision                  : {op.precision:.4f}")
    print(f"      false alarms               : {op.fp:,}")
    print(f"      FALSE ALARMS / DAY         : {fa_day:,.1f}"
          + (f"   (baseline {base['false_alarms_per_day']:,.1f})" if base else ""))
    print("=" * 56)

    print("\nDRIFT across test period (threshold fixed on full test):")
    drift = drift_report(y_test, scores_test, ts_test, n_bins=4,
                         target_recall=args.target_recall)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(drift.to_string(index=False))

    np.save(args.procdir / "ae_scores_test.npy", scores_test)
    (args.procdir / "ae_result.json").write_text(
        json.dumps(
            {
                "model": "AutoEncoder",
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
