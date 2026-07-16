"""Demonstrate adaptive thresholding on the HDFS autoencoder scores.

Calibrate a threshold on an initial chunk of the (time-ordered) test stream,
then run the rest two ways — a FIXED threshold vs an ADAPTIVE trailing-window
threshold — and compare false alarms per day and recall across the test period.
The threshold uses scores only (no labels); labels are used purely to score it.

Note: the HDFS autoencoder scores are strongly discrete (repetitive normal
blocks collapse onto identical reconstruction errors), which breaks the
continuous-tail assumption of EVT/GPD. We therefore use the distribution-free
`RollingQuantile` here; the EVT estimator (`AdaptivePot`) is reserved for the
continuous scores of the BGL timing detector. See docs/thresholding_results.md.

    python scripts/run_evt.py --q 0.03
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from logtriage.config import DATA_PROCESSED, PROJECT_ROOT
from logtriage.eval.evt import RollingQuantile


def _bin_metrics(flags, y, ts, thr, n_bins=6):
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    edges = pd.date_range(ts.min(), ts.max(), periods=n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = ((ts >= lo) & (ts <= hi)) if i == n_bins - 1 else ((ts >= lo) & (ts < hi))
        m = m.to_numpy()
        if m.sum() == 0:
            continue
        days = max((hi - lo).total_seconds() / 86400, 1e-9)
        fl, yl = flags[m], y[m]
        fp = int((fl & (yl == 0)).sum())
        tp = int((fl & (yl == 1)).sum())
        rows.append({
            "bin": i + 1,
            "flagged_rate": float(fl.mean()),
            "recall": float(tp / max((yl == 1).sum(), 1)),
            "false_alarms_per_day": fp / days,
            "threshold": float(np.median(thr[m])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procdir", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--q", type=float, default=0.02, help="target flagged rate")
    ap.add_argument("--calib", type=int, default=20_000, help="calibration blocks")
    ap.add_argument("--window", type=int, default=20_000)
    ap.add_argument("--stride", type=int, default=500)
    ap.add_argument("--figdir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    args = ap.parse_args()

    raw = np.load(args.procdir / "ae_scores_test.npy")
    meta = pd.read_csv(args.procdir / "hdfs_meta.csv", index_col=0, parse_dates=["start"])
    # scores align with meta[test] row order — attach BEFORE sorting so a sort by
    # time keeps scores/labels/timestamps aligned.
    test = meta[meta["split"] == "test"].copy()
    test["raw"] = raw
    test = test.sort_values("start")
    # Reconstruction MSE spans ~17 orders of magnitude (anomalies reconstruct
    # catastrophically); EVT on such a heavy tail is done in log space.
    scores = np.log(np.maximum(test["raw"].to_numpy(), 1e-12))
    y = test["label"].to_numpy()
    ts = test["start"].to_numpy()

    calib, rest = scores[: args.calib], scores[args.calib :]
    y_rest, ts_rest = y[args.calib :], ts[args.calib :]

    # FIXED: (1-q) quantile of the calibration scores, held constant
    z_fixed = float(np.quantile(calib, 1.0 - args.q))
    fixed_flags = rest > z_fixed
    fixed_thr = np.full(len(rest), z_fixed)

    # ADAPTIVE: trailing-window rolling quantile (robust to discrete scores)
    ad = RollingQuantile(q=args.q, window=args.window, stride=args.stride)
    ad.fit(calib)
    adapt_flags, adapt_thr = ad.run(rest)

    print(f"target flagged rate q = {args.q:.3f}   |   deployment blocks = {len(rest):,}")
    for name, fl, thr in [("FIXED", fixed_flags, fixed_thr),
                          ("ADAPTIVE", adapt_flags, adapt_thr)]:
        bm = _bin_metrics(fl, y_rest, ts_rest, thr)
        fp = int((fl & (y_rest == 0)).sum())
        tp = int((fl & (y_rest == 1)).sum())
        print(f"\n=== {name} threshold ===")
        print(f"  overall recall {tp / max((y_rest==1).sum(),1):.3f} | "
              f"flagged {fl.mean():.3%} | "
              f"FA/day range {bm.false_alarms_per_day.min():.0f}–"
              f"{bm.false_alarms_per_day.max():.0f} "
              f"(×{bm.false_alarms_per_day.max()/max(bm.false_alarms_per_day.min(),1):.0f} swing)")
        with pd.option_context("display.width", 120):
            print(bm.to_string(index=False))

    _plot(ts_rest, fixed_flags, fixed_thr, adapt_flags, adapt_thr, y_rest,
          args.q, args.figdir)


def _plot(ts, ff, ft, af, at, y, q, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))

    def fa_series(flags, nb=30):
        edges = pd.date_range(ts.min(), ts.max(), periods=nb + 1)
        mid, fa = [], []
        for i in range(nb):
            lo, hi = edges[i], edges[i + 1]
            m = (((ts >= lo) & (ts <= hi)) if i == nb - 1 else ((ts >= lo) & (ts < hi))).to_numpy()
            if m.sum() == 0:
                continue
            days = max((hi - lo).total_seconds() / 86400, 1e-9)
            mid.append(lo + (hi - lo) / 2)
            fa.append(int((flags[m] & (y[m] == 0)).sum()) / days)
        return mid, fa

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.plot(pd.to_datetime(ts), ft, color="#8172B3", lw=1.2, label="fixed threshold")
    a1.plot(pd.to_datetime(ts), at, color="#C44E52", lw=1.2, label="adaptive (rolling quantile)")
    a1.set_ylabel("alert threshold\n(log reconstruction score)")
    a1.legend(loc="upper left")
    a1.set_title(f"Adaptive thresholding on HDFS autoencoder scores (target flag rate q={q:.0%})")

    for flags, c, lab in [(ff, "#8172B3", "fixed"), (af, "#C44E52", "adaptive (rolling quantile)")]:
        mid, fa = fa_series(flags)
        a2.plot(mid, fa, color=c, lw=1.5, marker="o", ms=3, label=lab)
    a2.set_ylabel("false alarms / day")
    a2.set_xlabel("test period (time)")
    a2.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    p = figdir / "hdfs_evt_thresholding.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
