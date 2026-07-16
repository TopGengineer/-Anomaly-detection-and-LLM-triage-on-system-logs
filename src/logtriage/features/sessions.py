"""Block-session features for HDFS detection.

HDFS labels are per block and blocks are short-lived (<=54 s), so the detection
unit is the block session, not a fixed time window. Each block becomes a vector
of template counts (the loglizer "event count matrix"). The autoencoder learns
to reconstruct normal count vectors; reconstruction error is the anomaly score.

`build_count_matrix` returns two aligned objects, indexed by block_id:
    X    : DataFrame [n_blocks x n_templates] of event-id counts
    meta : DataFrame with start, end, n_events, and label (if provided)
"""

import pandas as pd

from logtriage.data.hdfs import explode_blocks


def build_count_matrix(
    events: pd.DataFrame, labels: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate parsed events into a per-block template-count matrix + metadata.

    A line referencing several blocks contributes its event to each block
    (loglizer convention). Rows of X and meta share the same block_id index and
    order (sorted by session start time, so downstream temporal splits are
    trivial and stable).
    """
    per_block = explode_blocks(events)

    # count matrix: block_id x event_id
    X = (
        pd.crosstab(per_block["block_id"], per_block["event_id"])
        .sort_index(axis=1)  # stable template column order
    )

    meta = (
        per_block.groupby("block_id")
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            n_events=("event_id", "size"),
        )
    )
    if labels is not None:
        meta = meta.join(labels)
        missing = meta["label"].isna().sum()
        if missing:
            print(f"warning: {missing} blocks have no label; dropping them")
            keep = meta["label"].notna()
            meta, X = meta[keep], X.loc[keep]
        meta["label"] = meta["label"].astype(int)

    # Order rows by session start so downstream temporal splits are trivial.
    # meta comes out of groupby sorted by block_id; a *stable* sort on start
    # keeps that block_id order within equal timestamps -> fully reproducible.
    order = meta.sort_values("start", kind="stable").index
    return X.loc[order], meta.loc[order]
