"""Strict temporal train/test split — the project's core validation discipline.

Blocks are ordered by session start time; the earliest `train_frac` become the
train set, the rest test. NEVER shuffled: no future information reaches training
or threshold-setting. The autoencoder trains on *normal* train blocks only; the
labels are used solely for evaluation.

A note on the <=54 s block overlap: because a block can end up to 54 s after it
starts, a train block near the boundary may finish just after the first test
block starts. Relative to the ~39 h span this is negligible, and splitting by
start time is the standard, reproducible choice. It is documented, not hidden.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitSummary:
    boundary: pd.Timestamp
    n_train: int
    n_test: int
    train_anomaly_rate: float
    test_anomaly_rate: float

    def __str__(self) -> str:
        return (
            f"temporal split @ {self.boundary}\n"
            f"  train: {self.n_train:>7,} blocks  "
            f"anomaly rate {self.train_anomaly_rate:.3%}\n"
            f"  test : {self.n_test:>7,} blocks  "
            f"anomaly rate {self.test_anomaly_rate:.3%}"
        )


def temporal_split(
    meta: pd.DataFrame, train_frac: float = 0.7
) -> tuple[pd.Series, SplitSummary]:
    """Label each block 'train' or 'test' by start-time order.

    `meta` must have a 'start' column (and 'label' for the summary). Returns a
    Series aligned to meta.index with values in {'train','test'}, plus a summary.
    """
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

    order = meta["start"].sort_values(kind="stable").index
    n_train = int(round(len(order) * train_frac))
    train_ids = set(order[:n_train])

    split = pd.Series(
        ["train" if b in train_ids else "test" for b in meta.index],
        index=meta.index,
        name="split",
    )
    boundary = meta.loc[order[n_train - 1], "start"]

    def _rate(mask):
        sub = meta.loc[mask]
        return float(sub["label"].mean()) if "label" in meta and len(sub) else float("nan")

    summary = SplitSummary(
        boundary=boundary,
        n_train=n_train,
        n_test=len(order) - n_train,
        train_anomaly_rate=_rate(split == "train"),
        test_anomaly_rate=_rate(split == "test"),
    )
    return split, summary
