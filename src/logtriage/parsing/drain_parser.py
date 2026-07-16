"""Drain parsing: raw log file -> structured event CSV.

Wraps logparser's Drain (loghub benchmark settings from `config.py`) and
post-processes its output into the canonical schema used downstream:

    timestamp   pandas datetime, from the log's own date/time fields
    event_id    Drain template id (e.g. "E5" -> stable per template hash)
    template    the mined template string, e.g. "Receiving block <*> src: <*> ..."
    block_ids   space-joined distinct blk_ ids on the line (HDFS session key)
    level, component, content   passthrough fields

Drain writes `<name>_structured.csv` / `<name>_templates.csv` into the output
dir; the canonical file is written next to them as `<name>_parsed.csv`.
"""

import re
from pathlib import Path

import pandas as pd
from logparser.Drain import LogParser

from logtriage.config import DATA_INTERIM, DatasetConfig

BLOCK_ID_RE = re.compile(r"blk_-?\d+")


def run_drain(log_path: Path, cfg: DatasetConfig, outdir: Path | None = None) -> Path:
    """Run Drain on `log_path`; return the path of the structured CSV."""
    log_path = Path(log_path)
    outdir = Path(outdir) if outdir else DATA_INTERIM
    outdir.mkdir(parents=True, exist_ok=True)

    parser = LogParser(
        log_format=cfg.log_format,
        indir=str(log_path.parent),
        outdir=str(outdir),
        depth=cfg.drain_depth,
        st=cfg.drain_sim_threshold,
        rex=cfg.masking_regexes,
    )
    parser.parse(log_path.name)
    return outdir / f"{log_path.name}_structured.csv"


def _hdfs_timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """HDFS stamps: Date='081109' (yymmdd), Time='203515' (HMMSS or HHMMSS)."""
    raw = date.astype(str).str.zfill(6) + time.astype(str).str.zfill(6)
    return pd.to_datetime(raw, format="%y%m%d%H%M%S")


def _canonical_hdfs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "timestamp": _hdfs_timestamp(df["Date"], df["Time"]),
            "event_id": df["EventId"],
            "template": df["EventTemplate"],
            "level": df["Level"],
            "component": df["Component"],
            "content": df["Content"],
        }
    )
    # A line can mention several blocks; keep all distinct ids, order-preserved.
    out["block_ids"] = df["Content"].map(
        lambda c: " ".join(dict.fromkeys(BLOCK_ID_RE.findall(str(c))))
    )
    return out


def _canonical_bgl(df: pd.DataFrame) -> pd.DataFrame:
    # BGL Time has microsecond resolution: '2005-06-03-15.42.50.675872'.
    ts = pd.to_datetime(df["Time"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
    bad = ts.isna().sum()
    if bad:
        print(f"warning: {bad} BGL lines had unparseable timestamps; dropping them")
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "event_id": df["EventId"],
            "template": df["EventTemplate"],
            # Label '-' = non-alert; any tag (e.g. KERNDTLB) = an alert.
            "label": (df["Label"].astype(str) != "-").astype(int),
            "node": df["Node"],
            "component": df["Component"],
            "level": df["Level"],
            "content": df["Content"],
        }
    )
    return out[out["timestamp"].notna()].reset_index(drop=True)


_CANONICALIZERS = {"HDFS": _canonical_hdfs, "BGL": _canonical_bgl}


def to_canonical(structured_csv: Path, dataset: str = "HDFS") -> pd.DataFrame:
    """Convert Drain's structured CSV into the dataset's canonical event frame."""
    df = pd.read_csv(structured_csv, dtype={"Date": str, "Time": str, "Label": str})
    if dataset not in _CANONICALIZERS:
        raise NotImplementedError(f"no canonicalizer for dataset {dataset!r} yet")
    return _CANONICALIZERS[dataset](df)


def parse_log(log_path: Path, cfg: DatasetConfig, outdir: Path | None = None) -> Path:
    """Full parsing stage: Drain + canonicalization. Returns the parsed CSV path."""
    log_path = Path(log_path)
    outdir = Path(outdir) if outdir else DATA_INTERIM
    structured = run_drain(log_path, cfg, outdir)
    canonical = to_canonical(structured, dataset=cfg.name)
    dest = outdir / f"{log_path.name}_parsed.csv"
    canonical.to_csv(dest, index=False)

    n_templates = canonical["event_id"].nunique()
    print(
        f"parsed {len(canonical):,} lines into {n_templates} templates -> {dest}"
    )
    return dest
