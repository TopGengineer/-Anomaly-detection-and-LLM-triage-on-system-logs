"""Build the HDFS block-session feature matrix and temporal split.

    python scripts/build_features.py \
        --input data/interim/HDFS.log_parsed.csv \
        --labels data/raw/anomaly_label.csv

Writes to data/processed/:
    hdfs_counts.npy       float32 [n_blocks x n_templates] count matrix
    hdfs_meta.csv         block_id, start, end, n_events, label, split
    hdfs_templates.json   ordered event_id list (the matrix columns)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from logtriage.config import DATA_PROCESSED
from logtriage.data.hdfs import load_labels, load_parsed
from logtriage.features.sessions import build_count_matrix
from logtriage.features.split import temporal_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="parsed CSV")
    ap.add_argument("--labels", type=Path, required=True, help="anomaly_label.csv")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--outdir", type=Path, default=DATA_PROCESSED)
    args = ap.parse_args()

    print("loading parsed events ...")
    events = load_parsed(args.input)
    labels = load_labels(args.labels)

    print("building block-session count matrix ...")
    X, meta = build_count_matrix(events, labels)
    print(f"  {X.shape[0]:,} blocks x {X.shape[1]} templates")

    split, summary = temporal_split(meta, train_frac=args.train_frac)
    meta["split"] = split
    print(summary)
    # drift-awareness sanity: is the test anomaly rate wildly different?
    ratio = summary.test_anomaly_rate / summary.train_anomaly_rate
    print(f"  test/train anomaly-rate ratio: {ratio:.2f}  (watch for drift)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / "hdfs_counts.npy", X.to_numpy(dtype=np.float32))
    meta.to_csv(args.outdir / "hdfs_meta.csv")
    (args.outdir / "hdfs_templates.json").write_text(json.dumps(list(X.columns)))
    print(f"\nwrote features to {args.outdir}")


if __name__ == "__main__":
    main()
