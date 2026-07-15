"""HDFS-specific loading: parsed events, per-block labels, block sessions.

HDFS ground truth is per *block* (`anomaly_label.csv`: block_id -> Normal/
Anomaly), so evaluation groups events into block sessions. Detection itself
stays window-based on the raw stream; block sessions are how scores meet
labels.
"""

from pathlib import Path

import pandas as pd

from logtriage.config import DATA_INTERIM, DATA_RAW


def load_parsed(path: Path | None = None) -> pd.DataFrame:
    """Load a canonical parsed CSV (see parsing/drain_parser.py for schema)."""
    path = Path(path) if path else DATA_INTERIM / "HDFS.log_parsed.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], keep_default_na=False)
    df["block_ids"] = df["block_ids"].astype(str)
    return df


def load_labels(path: Path | None = None) -> pd.Series:
    """Per-block labels as a Series: index=block_id, value=1 (anomaly) / 0."""
    path = Path(path) if path else DATA_RAW / "anomaly_label.csv"
    labels = pd.read_csv(path)
    return (
        labels.set_index("BlockId")["Label"]
        .map({"Normal": 0, "Anomaly": 1})
        .rename("label")
    )


def explode_blocks(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (line, block_id) pair; lines without a block id are dropped.

    A handful of HDFS lines reference several blocks — the event belongs to
    every session it mentions (same convention as the loglizer benchmarks).
    """
    df = events.copy()
    df["block_id"] = df["block_ids"].str.split(" ")
    df = df.explode("block_id")
    df = df[df["block_id"].astype(bool)]
    return df.drop(columns=["block_ids"]).reset_index(drop=True)


def build_block_sessions(
    events: pd.DataFrame, labels: pd.Series | None = None
) -> pd.DataFrame:
    """Aggregate events into one row per block: event-id sequence + times.

    Columns: block_id, event_seq (list, in stream order), start, end, n_events,
    and `label` if labels are given.
    """
    per_block = explode_blocks(events)
    sessions = (
        per_block.groupby("block_id")
        .agg(
            event_seq=("event_id", list),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            n_events=("event_id", "size"),
        )
        .reset_index()
    )
    if labels is not None:
        sessions = sessions.merge(
            labels, left_on="block_id", right_index=True, how="left"
        )
        missing = sessions["label"].isna().sum()
        if missing:
            print(f"warning: {missing} blocks have no label; dropping them")
            sessions = sessions.dropna(subset=["label"])
        sessions["label"] = sessions["label"].astype(int)
    return sessions
