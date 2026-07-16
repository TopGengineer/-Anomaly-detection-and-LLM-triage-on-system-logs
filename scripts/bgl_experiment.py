"""Does the Hawkes timing signal help detection? The BGL ablation.

Build fixed-time-window features on BGL, then detect anomalous windows three
ways — CONTENT only, TIMING only, CONTENT+TIMING — with the same model and the
same strict temporal split. If content+timing beats content, the Hawkes
self-excitation feature earns its place; if not, we report that honestly.

    python scripts/bgl_experiment.py --window 60 --top-k 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from logtriage.config import DATA_INTERIM, PROJECT_ROOT
from logtriage.eval.metrics import false_alarms_per_day, pr_auc
from logtriage.features.windows import build_window_features
from logtriage.models.baseline import IsolationForestBaseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DATA_INTERIM / "BGL.log_parsed.csv")
    ap.add_argument("--window", type=float, default=60.0, help="window seconds")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--figdir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    args = ap.parse_args()

    print("loading BGL events ...")
    ev = pd.read_csv(args.input, usecols=["timestamp", "event_id", "label"],
                     parse_dates=["timestamp"]).sort_values("timestamp")
    ev = ev.reset_index(drop=True)
    print(f"  {len(ev):,} events, {ev['label'].mean():.2%} alerts")

    # train-only content vocabulary (leakage-safe)
    n = len(ev)
    train_ev = ev.iloc[: int(args.train_frac * n)]
    vocab = train_ev["event_id"].value_counts().head(args.top_k).index.tolist()

    print("building window features ...")
    Xc, Xt, meta = build_window_features(
        ev, window_s=args.window, top_k=args.top_k, top_templates=vocab
    )
    y = meta["label"].to_numpy().astype(int)
    ts = meta["start"]
    n_win = len(meta)
    n_train = int(args.train_frac * n_win)
    is_train = np.arange(n_win) < n_train      # meta is time-ordered
    is_test = ~is_train
    normal_train = is_train & (y == 0)
    print(f"  {n_win:,} windows | {y.mean():.2%} anomalous | "
          f"train {is_train.sum():,} / test {is_test.sum():,}")

    Xc = Xc.to_numpy(dtype=np.float32)
    Xt = Xt.to_numpy(dtype=np.float32)
    Xct = np.hstack([Xc, Xt])
    y_test, ts_test = y[is_test], ts[is_test]

    feature_sets = {"content": Xc, "timing": Xt, "content+timing": Xct}
    results, scores = {}, {}
    for name, X in feature_sets.items():
        model = IsolationForestBaseline(random_state=0).fit(X[normal_train])
        s = model.score(X[is_test])
        scores[name] = s
        ap_score = pr_auc(y_test, s)
        fa, op = false_alarms_per_day(y_test, s, ts_test, target_recall=args.target_recall)
        results[name] = {"pr_auc": ap_score, "fa_per_day": fa, "precision": op.precision}

    print("\n" + "=" * 64)
    print(f"BGL WINDOW DETECTION — feature ablation (test, {y_test.mean():.2%} anomalous)")
    print("=" * 64)
    print(f"{'feature set':<18}{'PR-AUC':>10}{'prec@rec':>12}{'FA/day':>14}")
    for name, r in results.items():
        print(f"{name:<18}{r['pr_auc']:>10.4f}{r['precision']:>12.4f}{r['fa_per_day']:>14,.1f}")
    lift = results["content+timing"]["pr_auc"] - results["content"]["pr_auc"]
    print("-" * 64)
    print(f"timing lift (content+timing − content) in PR-AUC: {lift:+.4f}")
    verdict = ("timing HELPS" if lift > 0.005 else
               "timing does NOT materially help" if abs(lift) <= 0.005 else
               "timing HURTS")
    print(f"verdict: {verdict}")
    print("=" * 64)

    _plot(y_test, scores, args.figdir)
    (Path(PROJECT_ROOT) / "docs" / "_bgl_result.json").write_text(
        json.dumps({k: v for k, v in results.items()}, indent=2))


def _plot(y, scores, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {"content": "#8172B3", "timing": "#55A868", "content+timing": "#C44E52"}
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=colors[name], lw=2,
                label=f"{name}  (PR-AUC={average_precision_score(y, s):.3f})")
    ax.axhline(y.mean(), ls="--", color="gray", lw=1, label=f"chance ({y.mean():.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("BGL window detection — does Hawkes timing help?")
    ax.legend(loc="lower left"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1.02)
    fig.tight_layout()
    fig.savefig(figdir / "bgl_timing_ablation.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {figdir / 'bgl_timing_ablation.png'}")


if __name__ == "__main__":
    main()
