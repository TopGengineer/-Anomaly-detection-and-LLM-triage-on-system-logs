# Log Anomaly Detection with Hawkes-Process Timing and LLM Triage — Full Report

*A start-to-finish account of what was built, why, what the data forced us to
change, and what the results actually say — including the negative ones.*

---

## 0. What this project is, and what it is not

The application is anomaly detection on system logs; the **real** goal is a
technically strong, rigorously validated study of event/time-series data. The
detector accuracy is not the interesting part — HDFS anomaly detection is one of
the most reproduced results in the field, and our numbers match it. The
interesting parts are: modeling the *timing* of events as a point process
(Hawkes), a validation protocol built around the metric an on-call analyst
actually cares about (false alarms per day, under distribution drift), a
self-calibrating alert threshold from extreme-value theory, and an LLM triage
layer that turns raw alerts into ranked incidents.

Everything runs on **public data only** (the [loghub](https://github.com/logpai/loghub)
HDFS and BGL datasets). Every result below is reproducible from the scripts in
this repo; 34 unit tests cover the load-bearing logic.

---

## 1. The two datasets, and why we need both

We use two datasets because they play complementary roles that neither can fill
alone — a decision **driven by the data**, documented in `docs/eda_findings.md`.

| | **HDFS** | **BGL** |
|---|---|---|
| Role | Detection + triage showcase | Hawkes-process centerpiece |
| Size | 11,175,629 lines → 48 templates | 4,713,493 lines → 429 templates |
| Labels | per **block** (2.93% anomalous) | per **line** (7.39% alerts) |
| Timestamp resolution | **1 second** | **microsecond** |

The timestamp resolution is the crux. We measured it before building anything:

- **HDFS is too coarse for a continuous-time point process.** At ~80 events/second
  and 1-second stamps, **98.8%** of consecutive events share a timestamp; even
  within a single block, 90.5% tie and every block's whole life is ≤ 54 s. A
  Hawkes process needs a strict event ordering, which simply is not present.
  Adding sub-second jitter would *fabricate* the structure the model is supposed
  to measure, so we rejected it.
- **BGL has genuine microsecond timestamps** (`2005-06-03-15.42.50.675872`), 0%
  ties, events spread over seconds — exactly what a continuous-time Hawkes fit
  needs.

So HDFS hosts detection (its clean per-block labels are ideal for the headline
metrics); BGL hosts the Hawkes work (its timing resolution is the only place the
math is well-posed). This is a strength — validated on two independent
benchmarks — not redundancy.

---

## 2. From raw logs to structured events (Drain)

Raw log lines are unstructured text. We parse them with **Drain** (the loghub
benchmark configuration) into a fixed vocabulary of event *templates*:

```
081109 203515 148 INFO ... PacketResponder 1 for block blk_38865049064139660 terminating
        →  template "PacketResponder <*> for block <*> terminating",  event id E?
```

Each line becomes `(timestamp, event_id, template, …)`. HDFS lines carry a
`block_id` (the session key for its per-block labels); BGL lines carry an inline
alert label and a microsecond timestamp. Full HDFS parses to **48 templates**,
BGL to **429**. A coherence check: the HDFS `allocateBlock` template occurs
exactly 575,061 times — once per block, matching the label count.

Code: `src/logtriage/parsing/`, `scripts/parse.py`.

---

## 3. Exploratory analysis — the findings that shaped the design

EDA was not a formality; three measured findings changed the plan.

1. **Events cluster hard (self-excitation is real).** The Fano factor — variance
   over mean of per-minute counts, 1.0 for a random/Poisson process — is **5.3**
   on the HDFS sample and **698–2286** on BGL. This is the empirical justification
   for a Hawkes model, before any fit.
2. **The HDFS timestamp problem** (§1) — decisive for where Hawkes can live.
3. **Anomaly-rate drift.** The temporal train/test split (below) has a **3.26%**
   anomaly rate in the earlier 70% and **2.16%** in the later 30% — the base rate
   is not stationary, which foreshadows the drift results.

Code: `src/logtriage/eda.py`, `scripts/explore_hdfs.py`; figures in `reports/figures/`.

---

## 4. Detection on HDFS

### 4.1 The unit, the features, the split

HDFS labels are per **block**, and a block lives ≤ 54 s, so the detection unit is
the **block session** — all of a block's events grouped together, represented as
a **48-dimensional vector of template counts** (the standard loglizer "event
count matrix"). 575,061 blocks × 48 templates.

The validation discipline that the whole project rests on:

- **Strict temporal split, never shuffled.** Blocks are ordered by start time;
  the earliest 70% train, the latest 30% test. No test-period information reaches
  training, scaling, or thresholds.
- **PR-AUC, not accuracy or ROC** — positives are ~2% of blocks, so accuracy
  reads ~98% while telling you nothing.
- **Headline metric: false alarms per day at a fixed detection rate** — computed
  on the *actual* elapsed test time (blocks are not uniform in time).
- **Drift check** — every metric is also reported over consecutive slices of the
  test period, so degradation over time cannot hide in an aggregate.

Code: `src/logtriage/features/`, `src/logtriage/eval/metrics.py`.

### 4.2 Baseline vs autoencoder

Both are unsupervised, both scored through the *same* metrics harness on the
*same* split.

- **Isolation Forest** (baseline to beat): fit on log-transformed count vectors.
- **Autoencoder** (the detector): a small MLP (48→32→16→8→16→32→48) trained on
  **normal** train blocks only; per-block reconstruction error is the anomaly
  score. No attack labels in training — labels are for evaluation only.

| Metric (test split) | Isolation Forest | **Autoencoder** |
|---|---|---|
| PR-AUC | 0.748 | **0.987** |
| Precision @ 90% recall | 0.556 | **0.961** |
| False alarms / day @ 90% recall | ~12,700 | **~650** |

The autoencoder cuts false alarms by ~20× at the same detection rate. Its
discrimination is also **stable under drift** (PR-AUC 0.99+ in every test
sub-period, where the baseline swung 0.69–0.98). See
`reports/figures/hdfs_pr_comparison.png` and `docs/detection_results.md`.

**Honest framing:** this is a strong *reproduction*, not a novelty. The value of
the HDFS work is (a) it validates the pipeline and validation discipline on the
canonical benchmark, and (b) it is the substrate for the parts that *are*
un-commodity — the drift analysis, adaptive thresholding, and triage below.

Code: `src/logtriage/models/`, `scripts/run_baseline.py`, `scripts/train_autoencoder.py`.

---

## 5. The mathematical core — Hawkes process on BGL

### 5.1 The model and why it's correct before we trust it

A univariate Hawkes process with exponential kernel has conditional intensity

    λ(t) = μ + Σ_{tᵢ < t} α · e^{−β (t − tᵢ)}

— each event raises the instantaneous rate by α, decaying at rate β. The
**branching ratio** η = α/β is the expected number of direct offspring per event
(the single number describing how cascade-prone the stream is); the process is
subcritical iff η < 1. We fit (μ, α, β) by **maximum likelihood**, using the O(n)
Ogata recursion for the log-likelihood.

Because this is the centerpiece, it is **proven on simulated data before it ever
touches BGL**: we simulate a Hawkes process with known parameters (Ogata
thinning), fit it, and confirm the fitter recovers them; and we confirm the
time-rescaling goodness-of-fit test *passes* a correct fit and *rejects* a wrong
model. (`tests/test_hawkes.py`.) We implemented this directly with scipy rather
than the `tick` library, which has no wheels for modern Python.

### 5.2 Validation — the time-rescaling theorem

If the fitted model is right, the transformed inter-event times
Λ(tᵢ) − Λ(tᵢ₋₁) are i.i.d. Exp(1). We test that with a KS test and a QQ plot —
this is the model-independent check the project's rigor rests on: a branching
ratio means nothing if the fit fails rescaling.

### 5.3 Results

| BGL window | Events | **branching ratio η** | KS stat | mean rescaled gap |
|---|---|---|---|---|
| 2005-09-03 (moderate) | 6,858 | **0.971** | 0.095 | 1.0001 |
| 2005-06-11 (fault cascade) | 152,669 | **0.9998** | 0.431 | 1.0000 |

**The branching ratio is near-critical everywhere (η ≈ 0.97–1.0):** BGL's event
stream is strongly, almost self-sustaining, self-exciting — a precise, estimated
cascade strength, not just "bursty." The QQ plots show the *body* of the
rescaled-gap distribution on the Exp(1) diagonal but the *upper tail* deviating:
the deepest quiet gaps are longer than a single exponential predicts. Reading:
the exp-kernel captures the bulk clustering but not the full multi-scale
burst/silence structure — mild on the calm window, severe on the extreme
cascade. (Mean rescaled gap = 1.0000 confirms the compensator is correctly
normalized even where the shape deviates. At 10⁵ points the KS test rejects
almost anything, so we lean on the QQ shape and the magnitude, not the p-value.)

This is exactly the kind of result a rigorous study should produce: a validated
estimate *plus* an honest characterization of where the model breaks — which
motivates a sum-of-exponentials kernel as future work.

Code: `src/logtriage/hawkes/`, `scripts/fit_hawkes.py`, `docs/hawkes_results.md`.

---

## 6. Does the Hawkes timing signal actually help detection?

This is the question the whole BGL half exists to answer, and we tested it
honestly rather than assuming. On fixed 60-second BGL windows we detect
alert-containing windows three ways — **content** (top-50 template counts),
**timing** (per-window Hawkes self-excitation Σ e^{−β Δt} at two decay scales),
and **content+timing** — same model, same temporal split.

| feature set | PR-AUC | precision @ 90% recall | false alarms / day |
|---|---|---|---|
| content | 0.081 | 0.046 | 114 |
| **timing** | **0.087** | **0.083** | **61** |
| content+timing | 0.090 | 0.071 | 72 |
| chance | 0.047 | — | — |

**The honest, nuanced answer:** absolute detection is *weak* here (~0.09 PR-AUC,
~2× chance) — detecting BGL fault windows at this resolution is genuinely hard,
and we do not oversell it. **But the timing signal is the more useful one and it
cuts false alarms**: at 90% recall, timing roughly *halves* the false-alarm rate
versus content (114 → 61 per day) and nearly doubles precision. So the Hawkes
self-excitation, as a feature, is the single most useful signal and moves the
metric the project cares about — it just doesn't turn window-level BGL detection
into a strong detector on its own. A rigorously-validated "timing beats content
and halves false alarms, on a hard task" is a real finding; a manufactured 0.98
would not be. (`docs/bgl_timing_results.md`.)

Code: `src/logtriage/features/windows.py`, `scripts/bgl_experiment.py`.

---

## 7. Adaptive thresholding — stabilizing false alarms under drift

The drift analysis exposed the real operational weakness: at a *fixed* threshold,
the false-alarm rate wanders as the score distribution drifts, even while
discrimination stays high. On the HDFS autoencoder scores, across a single 5-hour
test window, a fixed threshold's false alarms swing from **3,664 to 861,686 per
day** — flagging up to 100% of blocks mid-period.

The fix is a self-calibrating threshold. We built the principled tool — **EVT /
Peaks-Over-Threshold**, fitting a Generalized Pareto tail to a trailing window
and inverting it for a target alert probability (`AdaptivePot`, verified on
simulated data) — *and* discovered, honestly, that it does not fit the HDFS
scores: those scores are strongly **discrete** (58.7% of normal blocks reconstruct
to the *identical* value, because normal HDFS blocks are so repetitive), which
breaks GPD's continuous-tail assumption. So on HDFS we use the distribution-free
**rolling quantile**; EVT is reserved for the *continuous* BGL timing scores where
its extrapolation genuinely earns its place.

| | Fixed threshold | **Adaptive (rolling quantile)** |
|---|---|---|
| False alarms / day, over test period | 3,664 – **861,686** | 0 – **7,584** |
| Flagged fraction | up to 100% | stable ≈ 2% |
| Recall | 1.00 | 0.72 |

The adaptive threshold holds the alert rate steady at the transparent cost of
recall in periods where anomalies surge past the flag budget — the honest
operating-point tradeoff, now explicit and tunable. See
`reports/figures/hdfs_evt_thresholding.png` and `docs/thresholding_results.md`.

Code: `src/logtriage/eval/evt.py`, `scripts/run_evt.py`.

---

## 8. LLM triage — from alerts to ranked incidents

Paging an analyst once per flagged block is unusable. The triage agent reduces
volume, then explains:

1. **Deduplicate** — flags with the same signature (entity + template multiset)
   collapse into one, with a count.
2. **Correlate into incidents** — flags close in time are grouped into a single
   incident. This is the Hawkes idea made operational: self-exciting events are
   causally linked, so a cascade becomes *one* incident, not 200 pages.
3. **Explain + rank** — each incident is classified into structured JSON
   (`priority`, `category`, `summary`, `recommended_action`, `explanation`) and
   ranked P1→P4. The explainer is pluggable: a deterministic rule-based backend
   (category from template keywords, priority from severity + cascade size) runs
   self-contained, and an **Anthropic backend (`claude-opus-4-8`, JSON-schema
   structured output)** drops in when a credential is configured, falling back
   automatically. One model call per *incident*, not per raw flag.

On the top-200 flagged HDFS blocks: **200 raw flags → 90 incidents**, prioritized
P1(13)/P2(53)/P3(2)/P4(22) across data-loss / replication / network categories.

Code: `src/logtriage/triage/`, `scripts/run_triage.py`.

> **Note on the LLM backend:** this environment has no model credential, so the
> demo ran on the rule-based explainer. The Anthropic path is written to the
> current API (structured outputs, `claude-opus-4-8`) and activates with a key —
> `python scripts/run_triage.py --top 200 --llm`.

---

## 9. The dashboard

`app/dashboard.py` (Streamlit) surfaces the whole pipeline in four tabs —
Detection (metrics + PR curve), Thresholding (the drift-stabilization plot),
Hawkes (intensity + QQ), and Incidents (the ranked table with per-incident
drill-down). Verified to start and render headless. Run with
`streamlit run app/dashboard.py`.

---

## 10. Results at a glance

| Result | Number |
|---|---|
| HDFS parsed | 11.2M lines → 48 templates |
| BGL parsed | 4.71M lines → 429 templates |
| Autoencoder PR-AUC (test) | **0.987** (baseline 0.748) |
| False alarms/day @ 90% recall | **~650** (baseline ~12,700) |
| Hawkes branching ratio (BGL) | **η ≈ 0.97–1.0** (near-critical) |
| Timing feature effect (BGL) | **halves** false alarms vs content |
| Fixed → adaptive threshold FA/day swing | **×235 → held near target** |
| Triage reduction | 200 flags → **90 ranked incidents** |
| Unit tests | **34 passing** |

---

## 11. Honest limitations

- **HDFS detection is a reproduction**, not a contribution — its role is
  validation and substrate, and we say so.
- **BGL window detection is weak in absolute terms** (~0.09 PR-AUC); the timing
  result is a *relative* win (fewer false alarms), on a hard task.
- **The single-exponential Hawkes kernel underfits** BGL's deepest lulls
  (time-rescaling upper tail) — a sum-of-exponentials kernel is the fix.
- **EVT does not fit the discrete HDFS scores**; the rolling quantile is used
  there, and EVT is reserved for continuous scores.
- **The LLM triage backend was not exercised live** (no credential in this
  environment); only the rule-based path ran end-to-end here.
- Window size, feature vocabulary, and decay scales are first choices, not
  swept.

---

## 12. What I would do next

1. **Richer Hawkes kernel** (sum-of-exponentials / power-law) to fix the
   upper-tail misfit the QQ plots expose.
2. **Time-rescaling residual as a detector** — use the surprise-under-the-fitted-
   model directly as a calibrated timing-anomaly score, and put EVT thresholding
   on those (continuous) scores.
3. **Sweep** window size and decay scales on BGL; try the autoencoder in place of
   Isolation Forest for the timing ablation.
4. **Exercise the LLM triage** end-to-end with a real model and evaluate incident
   quality against held-out labels.

---

## 13. How to reproduce

```bash
pip install -r requirements.txt && pip install --no-deps logparser3 && pip install -e .

python scripts/download.py --dataset HDFS --full
python scripts/parse.py    --dataset HDFS --input data/raw/HDFS.log
python scripts/build_features.py --input data/interim/HDFS.log_parsed.csv --labels data/raw/anomaly_label.csv
python scripts/run_baseline.py
python scripts/train_autoencoder.py
python scripts/run_evt.py --q 0.03
python scripts/run_triage.py --top 200

python scripts/download.py --dataset BGL --full
python scripts/parse.py    --dataset BGL --input data/raw/BGL.log
python scripts/fit_hawkes.py --start 2005-06-11 --hours 24 --stream all
python scripts/bgl_experiment.py --window 60 --top-k 50

streamlit run app/dashboard.py
```

Per-component write-ups with the full numbers live alongside this file:
`eda_findings.md`, `detection_results.md`, `hawkes_results.md`,
`bgl_timing_results.md`, `thresholding_results.md`.
