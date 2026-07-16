# Adaptive thresholding — stabilising the false-alarm rate under drift

The detector produces scores; a threshold turns them into alerts. The drift
analysis (`docs/detection_results.md`) showed the operational weakness: at a
**fixed** threshold, the false-alarm rate wanders as the score distribution
drifts, even when discrimination (PR-AUC) stays high. This is the piece that
addresses it. Reproduce with `scripts/run_evt.py`.

## Two estimators, each for the distribution it fits

`src/logtriage/eval/evt.py` provides both, behind the same `fit`/`run` interface:

- **`AdaptivePot`** — Extreme Value Theory (Peaks-Over-Threshold). Fits a
  Generalized Pareto tail to a trailing window and inverts it for the score
  level exceeded with target probability q. The principled tool for **continuous
  heavy-tailed** scores. Verified on simulated data (`tests/test_evt.py`):
  controls the exceedance rate, and tracks drift where a fixed threshold fails.
- **`RollingQuantile`** — the distribution-free (1−q) quantile of a trailing
  window. No tail model, so it is robust to **discrete** score distributions.

## Why EVT is NOT used on HDFS (an honest finding)

The HDFS autoencoder's reconstruction errors are strongly **discrete**: normal
HDFS blocks are so repetitive that large groups reconstruct to *identical*
errors — 58.7% of normal test blocks share the single lowest score value. GPD
assumes a continuous tail, so fitting it to these point masses is unstable
(extrapolated thresholds fall below the data floor and flag everything). This is
a property of the data, not a bug. On HDFS we therefore use `RollingQuantile`;
EVT is reserved for the continuous scores of the BGL timing detector (upcoming),
where its extrapolation to small q genuinely earns its place.

## Result on HDFS (autoencoder scores, target flag rate q = 3%)

Calibrate on the first 20k blocks of the time-ordered test stream, then run the
remaining 152,518 two ways. The threshold uses **scores only** (no labels).

| | Fixed threshold | **Adaptive (rolling quantile)** |
|---|---|---|
| False alarms / day, range over test period | 3,664 – **861,686** (×235) | 0 – **7,584** |
| Flagged fraction | up to 100% mid-period | stable ≈ 2% |
| Overall recall | 1.00 | 0.72 |

See `reports/figures/hdfs_evt_thresholding.png`: mid-period the *normal* blocks'
scores drift up, so the fixed threshold sits far into the normal bulk and its
false alarms explode (~860k/day, flagging essentially everything); the adaptive
threshold rises to track the drift and holds the alert rate near the 3% budget
(single-digit-thousands/day).

**The honest tradeoff.** Holding a flag-rate budget means recall dips (1.00 →
0.72) in the early bins, where the local anomaly rate exceeds the budget so the
threshold cannot catch every anomaly without exceeding the false-alarm target.
That is the real operating-point tension — the adaptive threshold makes it
explicit and controllable instead of letting the false-alarm rate run wild.

## Takeaways

- A fixed threshold is not deployable under drift; the false-alarm rate is the
  thing that moves, and it moves by orders of magnitude.
- A trailing-window adaptive threshold stabilises it, at a transparent recall
  cost that is itself a tunable knob (q).
- EVT is the right adaptive tail model for continuous scores and is validated
  and ready for the BGL timing detector; HDFS's discrete scores call for the
  distribution-free quantile instead.
