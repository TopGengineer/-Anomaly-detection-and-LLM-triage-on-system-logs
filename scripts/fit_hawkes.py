"""Fit the exponential-kernel Hawkes process to a BGL time window and validate it.

    python scripts/fit_hawkes.py --start 2005-06-11 --hours 24 --stream all

Fits (mu, alpha, beta) by MLE, reports the branching ratio, runs the
time-rescaling goodness-of-fit test, and writes an intensity plot + a QQ plot.

Note on large N: the KS test has enormous power at 10^5 points and will reject
almost any parametric model on real data. We therefore report the KS number but
lean on the QQ plot and the mean rescaled gap to judge the fit's *magnitude* of
misfit — the honest reading at this sample size.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from logtriage.config import PROJECT_ROOT
from logtriage.hawkes.exp_kernel import fit, intensity_on_grid
from logtriage.hawkes.validation import ks_test_exp1, qq_points, rescaled_gaps


def load_window(input_csv, start, hours, stream) -> tuple[np.ndarray, pd.Timestamp]:
    df = pd.read_csv(input_csv, usecols=["timestamp", "label"], parse_dates=["timestamp"])
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours)
    m = (df["timestamp"] >= start) & (df["timestamp"] < end)
    if stream == "alerts":
        m &= df["label"] == 1
    ts = df.loc[m, "timestamp"].sort_values()
    if len(ts) < 100:
        raise SystemExit(f"only {len(ts)} events in window — widen it")
    # seconds since window start = continuous time axis
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    return t, start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/interim/BGL.log_parsed.csv"))
    ap.add_argument("--start", default="2005-06-11", help="window start (date or datetime)")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--stream", choices=["all", "alerts"], default="all")
    ap.add_argument("--figdir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    args = ap.parse_args()

    print(f"loading window {args.start} +{args.hours}h, stream={args.stream} ...")
    t, start = load_window(args.input, args.start, args.hours, args.stream)
    print(f"  {len(t):,} events over {t[-1]:.0f} s")

    print("fitting (MLE) ...")
    f = fit(t)
    print("\n" + str(f))

    gaps = rescaled_gaps(t, f)
    gof = ks_test_exp1(gaps)
    print("\n" + str(gof))

    _make_plots(t, f, gaps, args.figdir, args.stream, start)


def _make_plots(t, f, gaps, figdir, stream, start):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    tag = f"bgl_hawkes_{start.date()}_{stream}"

    # conditional intensity λ(t) on a grid (O(n+m) sweep)
    grid = np.linspace(0, t[-1], 4000)
    lam = intensity_on_grid(t, grid, f.mu, f.alpha, f.beta)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(grid, lam, lw=0.7, color="#C44E52")
    ax.set_xlabel("seconds since window start")
    ax.set_ylabel("λ(t)  [events/s]")
    ax.set_title(
        f"BGL Hawkes conditional intensity — {start.date()} ({stream}), "
        f"branching ratio η={f.branching_ratio:.3f}"
    )
    fig.tight_layout()
    fig.savefig(figdir / f"{tag}_intensity.png", dpi=120)
    plt.close(fig)

    # QQ plot of rescaled gaps vs Exp(1)
    theo, emp = qq_points(gaps)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(theo, emp, s=3, alpha=0.3, color="#4C72B0")
    hi = float(max(theo[-1], emp[-1]))
    ax.plot([0, hi], [0, hi], "k--", lw=1)
    ax.set_xlabel("theoretical Exp(1) quantiles")
    ax.set_ylabel("empirical rescaled-gap quantiles")
    ax.set_title("Time-rescaling QQ plot (on diagonal = good fit)")
    fig.tight_layout()
    fig.savefig(figdir / f"{tag}_qq.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {tag}_intensity.png and {tag}_qq.png to {figdir}")


if __name__ == "__main__":
    main()
