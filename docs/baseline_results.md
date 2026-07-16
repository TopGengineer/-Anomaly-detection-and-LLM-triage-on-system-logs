# Baseline results — Isolation Forest (HDFS)

The unsupervised detector the autoencoder must beat. Fit on the 402,543 train
blocks (no labels), evaluated on the 172,518 test blocks. Reproduce with
`scripts/run_baseline.py`.

## Headline (test split)

| Metric | Value |
|---|---|
| PR-AUC (average precision) | **0.748** |
| Chance PR-AUC (test anomaly rate) | 0.0216 |
| Precision @ 90% recall | 0.556 |
| False positives @ 90% recall | 2,681 |
| **False alarms / day @ 90% recall** | **~12,700** |

PR-AUC of 0.75 against a 0.022 chance line is strong discrimination for a
baseline — that is the number to beat. But the operational metric is brutal:
at 90% detection, ~12,700 false alarms per day. That figure is real, not an
artifact — the test window runs at ~34,000 blocks/hour — and it is exactly why
a raw detector with a single global threshold is not deployable. It motivates
(a) a better detector (the autoencoder) and (b) self-calibrating thresholds
(the EVT stretch goal).

## Drift across the test period (threshold fixed once on full test)

| bin | start | anomaly rate | PR-AUC | recall | FA/day |
|---|---|---|---|---|---|
| 1 | 06:00 | 2.90% | 0.977 | 0.938 | 95 |
| 2 | 07:16 | 3.15% | 0.692 | 0.939 | 22,019 |
| 3 | 08:32 | 1.03% | 0.762 | 0.884 | 27,030 |
| 4 | 09:48 | 1.57% | 0.829 | 0.810 | 1,746 |

The point of the project's drift discipline, made concrete: across a **single
5-hour test window**, PR-AUC swings 0.69 → 0.98 and false alarms per day swing
95 → 27,000 — three orders of magnitude. A threshold fixed on the aggregate is
fragile against this non-stationarity. This is the central weakness the later
components (better features incl. the Hawkes λ(t) on BGL, and EVT adaptive
thresholding) are meant to address.

## Method notes

- Anomaly score = `-IsolationForest.score_samples` (higher = more anomalous),
  same orientation the autoencoder's reconstruction error will use.
- Counts are `log1p`-transformed (a few templates dominate the count columns).
- Operating point: smallest set of top-scored blocks reaching the target
  recall; false alarms per day uses the **real** elapsed test span (blocks are
  not uniform in time).
