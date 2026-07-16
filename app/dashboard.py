"""Streamlit dashboard — logs in, ranked and explained incidents out.

Surfaces the whole pipeline in one view:
  * Detection — baseline vs autoencoder metrics + PR curve + drift.
  * Thresholding — adaptive vs fixed false-alarm rate under drift.
  * Hawkes — branching ratio and time-rescaling goodness-of-fit (BGL).
  * Incidents — the triage agent's ranked, deduped, correlated incidents.

Reads artifacts produced by the pipeline scripts (data/processed/, docs/,
reports/figures/). Run with:  streamlit run app/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"


def _load_json(path: Path) -> dict | list | None:
    return json.loads(path.read_text()) if path.exists() else None


def _fig(name: str, caption: str = "") -> None:
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"figure not generated yet: {name}")


st.set_page_config(page_title="Log Anomaly Triage", layout="wide")
st.title("Log Anomaly Detection & LLM Triage")
st.caption("Unsupervised detection · Hawkes-process timing · adaptive thresholds · LLM triage")

tab_det, tab_thr, tab_hawkes, tab_inc = st.tabs(
    ["Detection", "Thresholding", "Hawkes (BGL)", "Incidents"]
)

# ---------------- Detection ----------------
with tab_det:
    st.header("HDFS detection — autoencoder vs baseline")
    base = _load_json(PROC / "baseline_result.json")
    ae = _load_json(PROC / "ae_result.json")
    c1, c2, c3 = st.columns(3)
    if ae:
        c1.metric("Autoencoder PR-AUC", f"{ae['pr_auc']:.3f}",
                  delta=None if not base else f"{ae['pr_auc'] - base['pr_auc']:+.3f} vs baseline")
        c2.metric("False alarms / day @ 90% recall", f"{ae['false_alarms_per_day']:,.0f}",
                  delta=None if not base else
                  f"{ae['false_alarms_per_day'] - base['false_alarms_per_day']:+,.0f}",
                  delta_color="inverse")
        c3.metric("Precision @ 90% recall", f"{ae['precision_at_recall']:.3f}")
    else:
        st.info("Run scripts/train_autoencoder.py to populate detection metrics.")
    _fig("hdfs_pr_comparison.png", "Precision–recall on the strict temporal test split.")

# ---------------- Thresholding ----------------
with tab_thr:
    st.header("Adaptive thresholding — stabilising false alarms under drift")
    st.markdown(
        "A fixed threshold's false-alarm rate wanders as the score distribution "
        "drifts; a trailing-window adaptive threshold holds it steady.")
    _fig("hdfs_evt_thresholding.png",
         "Fixed vs adaptive threshold and the resulting false-alarms/day.")

# ---------------- Hawkes ----------------
with tab_hawkes:
    st.header("Hawkes process — event self-excitation on BGL")
    st.markdown(
        "A univariate exponential-kernel Hawkes process fit by MLE; the branching "
        "ratio measures cascade strength, and the time-rescaling QQ plot validates "
        "the fit.")
    col = st.columns(2)
    with col[0]:
        _fig("bgl_hawkes_2005-09-03_all_intensity.png", "Conditional intensity λ(t) — bursts.")
    with col[1]:
        _fig("bgl_hawkes_2005-09-03_all_qq.png", "Time-rescaling QQ vs Exp(1).")

# ---------------- Incidents ----------------
with tab_inc:
    st.header("Triaged incidents")
    incidents = _load_json(PROC / "incidents.json")
    if not incidents:
        st.info("Run scripts/run_triage.py to generate ranked incidents.")
    else:
        df = pd.DataFrame(incidents)
        left, right = st.columns([1, 3])
        with left:
            st.metric("Incidents", len(df))
            st.metric("Raw flags collapsed", int(df["dedup_count"].sum()))
            prios = sorted(df["priority"].unique())
            pick = st.multiselect("Priority", prios, default=prios)
            st.bar_chart(df["priority"].value_counts().sort_index())
        with right:
            shown = df[df["priority"].isin(pick)].sort_values(
                ["priority", "max_score"], ascending=[True, False])
            st.dataframe(
                shown[["id", "priority", "category", "max_score", "n_anomalies",
                       "dedup_count", "summary", "recommended_action"]],
                use_container_width=True, hide_index=True,
            )
            sel = st.selectbox("Inspect incident", shown["id"].tolist())
            if sel:
                inc = shown[shown["id"] == sel].iloc[0]
                st.subheader(f"{inc['id']} — {inc['priority']} · {inc['category']}")
                st.write(inc["summary"])
                st.write(f"**Recommended action:** {inc['recommended_action']}")
                st.write(f"**Explanation:** {inc['explanation']}")
                st.caption(f"{inc['start']} → {inc['end']} · entities: "
                           f"{', '.join(inc['entities'][:8])}")
