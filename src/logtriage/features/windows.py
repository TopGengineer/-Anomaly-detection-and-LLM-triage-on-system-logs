"""Fixed-time-window features for BGL — the substrate for the timing experiment.

The question the BGL work exists to answer: does the Hawkes self-excitation
signal add anything to a content-only detector? To test it fairly we build, per
fixed time window, two feature blocks that a detector can use separately or
together:

  * CONTENT  — counts of the top-K event templates in the window (the standard
    "bag of events" representation, ignoring timing).
  * TIMING   — the Hawkes exponential-kernel self-excitation A(t) = Σ_{t_i<t}
    e^{−β(t−t_i)} evaluated at each event and summarised per window (mean/max),
    at one or more decay scales β, plus the event count. This is exactly the
    excitation term of the conditional intensity, so it encodes "is this window
    inside a self-exciting cascade?" — computed from event *times only*, no
    labels, so it is a legitimate detector feature.

A window's label is 1 iff it contains at least one BGL alert line.
"""

import numpy as np
import pandas as pd

from logtriage.hawkes.exp_kernel import _recursion_A


def _excitation_per_event(t: np.ndarray, betas) -> dict[float, np.ndarray]:
    """A(t_i) = Σ_{j<i} e^{−β(t_i−t_j)} for each event, per decay β (O(n) each)."""
    return {beta: _recursion_A(t, beta) for beta in betas}


def build_window_features(
    events: pd.DataFrame,
    window_s: float = 60.0,
    top_k: int = 50,
    betas=(1.0, 1 / 30.0),
    top_templates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (X_content, X_timing, meta), one row per non-empty time window.

    `top_templates` fixes the content vocabulary (pass the train-derived list to
    avoid leakage); if None it is taken from `events` by frequency.
    """
    ev = events.sort_values("timestamp").reset_index(drop=True)
    t0 = ev["timestamp"].iloc[0]
    secs = (ev["timestamp"] - t0).dt.total_seconds().to_numpy()
    ev["win"] = (secs // window_s).astype(np.int64)

    # content vocabulary
    if top_templates is None:
        top_templates = ev["event_id"].value_counts().head(top_k).index.tolist()
    ev_top = ev[ev["event_id"].isin(top_templates)]
    content = (
        pd.crosstab(ev_top["win"], ev_top["event_id"])
        .reindex(columns=top_templates, fill_value=0)
    )

    # timing: excitation per event -> per-window mean/max at each beta
    exc = _excitation_per_event(secs, betas)
    timing = pd.DataFrame({"win": ev["win"]})
    for beta, a in exc.items():
        timing[f"exc_mean_{beta:.4g}"] = a
        timing[f"exc_max_{beta:.4g}"] = a
    agg = {c: ("mean" if "mean" in c else "max") for c in timing.columns if c != "win"}
    timing = timing.groupby("win").agg(agg)
    timing["n_events"] = ev.groupby("win").size()

    meta = pd.DataFrame({
        "win": ev.groupby("win").size().index,
        "start": ev.groupby("win")["timestamp"].min().values,
        "n_events": ev.groupby("win").size().values,
        "label": ev.groupby("win")["label"].max().values,  # alert in window?
    }).set_index("win")

    # align all three on the window index, ordered by time
    idx = meta.sort_values("start").index
    content = content.reindex(idx, fill_value=0)
    timing = timing.reindex(idx).fillna(0.0)
    meta = meta.loc[idx]
    return content, timing, meta
