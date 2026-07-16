"""Exploratory data analysis and data-quality checks for parsed logs.

Two jobs:
  * `data_quality_report` — a dict of health checks you run BEFORE trusting the
    data (nulls, duplicates, time ordering, timestamp resolution & ties, block
    coverage). Print it, eyeball it, catch problems early.
  * `make_plots` — the standard visual set (template frequencies, event
    timeline, inter-event-time distribution, burstiness, block-session sizes)
    saved as PNGs.

Everything operates on the canonical parsed frame from parsing/drain_parser.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _block_sizes(df: pd.DataFrame) -> pd.Series:
    """Events per block (a line can name several blocks -> counted in each)."""
    ex = df.assign(_b=df["block_ids"].astype(str).str.split()).explode("_b")
    ex = ex[ex["_b"].astype(bool)]
    return ex.groupby("_b").size()


def data_quality_report(df: pd.DataFrame) -> dict:
    """Compute health checks. Returns a flat dict; also readable when printed."""
    n = len(df)
    ts = df["timestamp"]
    gaps = ts.diff().dropna().dt.total_seconds()
    sizes = _block_sizes(df)
    no_block = (df["block_ids"].astype(str).str.len() == 0).sum()

    rep = {
        "n_rows": n,
        "time_span": str(ts.max() - ts.min()),
        "time_start": str(ts.min()),
        "time_end": str(ts.max()),
        "n_templates": df["event_id"].nunique(),
        # completeness — block_ids is excluded: an empty value there means the
        # line references no block (legitimate), tracked by lines_without_block_id.
        "null_cells": int(
            df.drop(columns=["block_ids"]).replace("", np.nan).isna().sum().sum()
        ),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_content_lines": int(df["content"].duplicated().sum()),
        # time integrity
        "time_monotonic_nondecreasing": bool(ts.is_monotonic_increasing),
        "times_going_backwards": int((gaps < 0).sum()),
        "timestamp_resolution_seconds_only": bool((ts.dt.microsecond == 0).all()),
        "same_timestamp_tie_fraction": round(float((gaps == 0).mean()), 4),
        # structure
        "lines_without_block_id": int(no_block),
        "n_blocks": int(sizes.nunique() if sizes.empty else sizes.index.nunique()),
        "median_events_per_block": float(sizes.median()) if len(sizes) else 0.0,
        "max_events_per_block": int(sizes.max()) if len(sizes) else 0,
        "fraction_single_event_blocks": (
            round(float((sizes == 1).mean()), 4) if len(sizes) else float("nan")
        ),
        "level_counts": df["level"].value_counts().to_dict(),
    }
    return rep


def print_report(rep: dict) -> None:
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    width = max(len(k) for k in rep)
    for k, v in rep.items():
        print(f"  {k:<{width}} : {v}")
    # plain-language flags
    print("-" * 60)
    if rep["fraction_single_event_blocks"] > 0.5:
        print("  ⚠  >50% of blocks have a single event — this looks like a")
        print("     TRUNCATED slice (e.g. the 2k sample), not full sessions.")
    if rep["timestamp_resolution_seconds_only"] and rep["same_timestamp_tie_fraction"] > 0:
        print(f"  ⚠  {rep['same_timestamp_tie_fraction']:.1%} of events share a timestamp")
        print("     (1-second resolution). Break these ties before Hawkes fitting.")
    if not rep["time_monotonic_nondecreasing"]:
        print("  ⚠  timestamps are NOT globally sorted — sort before temporal split.")
    if rep["null_cells"] == 0 and rep["exact_duplicate_rows"] == 0:
        print("  ✓  no nulls, no duplicate rows.")
    print("=" * 60)


def make_plots(df: pd.DataFrame, outdir: Path, prefix: str = "hdfs") -> list[Path]:
    """Write the standard EDA figures as PNGs. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name):
        p = outdir / f"{prefix}_{name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    # 1. Template frequency — how skewed is the event vocabulary?
    counts = df["template"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(counts)), counts.values, color="#4C72B0")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([t[:55] + ("…" if len(t) > 55 else "") for t in counts.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title(f"Event template frequency ({len(counts)} templates)")
    save(fig, "template_frequency")

    # 2. Event timeline — events per minute, reveals bursts
    per_min = df.set_index("timestamp").resample("1min").size()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(per_min.index, per_min.values, lw=0.8, color="#C44E52")
    ax.set_ylabel("events / minute")
    ax.set_title("Event volume over time (bursts = self-excitation)")
    save(fig, "timeline")

    # 3. Inter-event time distribution — the raw material for Hawkes
    gaps = df["timestamp"].diff().dropna().dt.total_seconds()
    gaps_pos = gaps[gaps > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log10(gaps_pos), bins=50, color="#55A868", edgecolor="white")
    ax.set_xlabel("log10(inter-event gap in seconds)")
    ax.set_ylabel("count")
    ax.set_title("Inter-event time distribution")
    save(fig, "inter_event_times")

    # 4. Burstiness — events-per-minute histogram vs Poisson expectation
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(per_min.values, bins=range(0, int(per_min.max()) + 2), color="#8172B3", edgecolor="white")
    fano = per_min.var() / per_min.mean() if per_min.mean() else float("nan")
    ax.set_xlabel("events in a 1-minute window")
    ax.set_ylabel("number of minutes")
    ax.set_title(f"Burstiness — Fano factor = {fano:.2f}  (1.0 = Poisson/random)")
    save(fig, "burstiness")

    # 5. Block-session sizes — are these full sessions or a truncated slice?
    sizes = _block_sizes(df)
    if len(sizes):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(sizes.values, bins=range(1, int(sizes.max()) + 2), color="#CCB974", edgecolor="white")
        ax.set_xlabel("events per block")
        ax.set_ylabel("number of blocks")
        ax.set_title(f"Block-session sizes (median {sizes.median():.0f}, max {sizes.max()})")
        save(fig, "block_sizes")

    return written
