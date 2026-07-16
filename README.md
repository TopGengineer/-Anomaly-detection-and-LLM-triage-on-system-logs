# Log Anomaly Detection with Hawkes-Process Features and LLM Triage

Unsupervised anomaly detection on system logs, with a stochastic-process core and an
LLM-based triage layer that turns raw alerts into ranked, explained incidents.

Built entirely on public data ([loghub](https://github.com/logpai/loghub) datasets:
HDFS first, then BGL). Self-contained — no proprietary data or internal APIs.

## System overview

```
raw logs ──> Drain parsing ──> event windows ──┐
                                               ├──> features ──> autoencoder ──> anomaly scores
event timestamps ──> Hawkes process λ(t) ──────┘                     │
                                                                     v
                                            dedup ──> LLM triage agent ──> ranked incidents
                                                                     │
                                                                     v
                                                          Streamlit dashboard
```

### 1. Detection

An autoencoder is trained on **normal** log windows only; reconstruction error is the
anomaly score. No attack labels are needed for training — labels are used exclusively
for evaluation. An Isolation Forest on the same features serves as the baseline to beat.

### 2. Mathematical core — Hawkes process

Log events cluster in time (self-excitation): one failure triggers cascades of related
events. We model event timestamps with a univariate Hawkes process with exponential
kernel,

```
λ(t) = μ + Σ_{tᵢ < t} α β e^{−β (t − tᵢ)}
```

fit by maximum likelihood. From the fit we get:

- the **branching ratio** n = α (expected number of direct offspring per event) — a
  single interpretable number for how cascade-prone the log stream is;
- a **goodness-of-fit test via the time-rescaling theorem**: if the model is right, the
  transformed inter-event times Λ(tᵢ) − Λ(tᵢ₋₁) are i.i.d. Exp(1) (checked with a QQ
  plot / KS test);
- the **conditional intensity λ(t)** evaluated per window, fed to the detector as an
  additional feature — "how surprising is this burst given the process's own history?"

### 3. Triage agent

An LLM (API-based, with local Ollama fallback) receives flagged windows, pulls
surrounding log context, correlates related anomalies into a single incident, and emits
structured JSON — `priority`, `category`, `summary`, `recommended_action` — plus a short
human-readable explanation. Alerts are deduplicated before any model call.

### Validation protocol (the important part)

- **Strict temporal split** — train on the earlier portion of the stream, test on the
  later. Never shuffled. No test-period information leaks into training or thresholds.
- **Precision–recall AUC**, not accuracy or ROC — positives are ~0.1% of windows.
- **Headline metric: false alarms per day at a fixed detection rate** — the number an
  on-call analyst actually cares about.
- **Drift check** — performance is reported across sub-periods of the test window, not
  as a single aggregate that can hide degradation.

### Stretch: EVT adaptive thresholding

Fit a Generalized Pareto distribution to the tail of the anomaly-score distribution
(Peaks-Over-Threshold), so the alert threshold self-calibrates to a target false-alarm
rate instead of being hand-tuned.

## Repository layout

```
├── src/logtriage/          # the package
│   ├── config.py           # dataset definitions (formats, regexes, URLs)
│   ├── data/               # download + dataset loading (HDFS, later BGL)
│   ├── parsing/            # Drain wrapper for log -> structured events
│   ├── features/           # windowing + feature construction        (upcoming)
│   ├── hawkes/             # Hawkes fitting, rescaling GOF, λ(t)     (upcoming)
│   ├── models/             # autoencoder, Isolation Forest baseline  (upcoming)
│   ├── triage/             # LLM triage agent                        (upcoming)
│   └── eval/               # temporal split, PR-AUC, FA/day, drift   (upcoming)
├── scripts/                # runnable entry points, one per pipeline stage
├── data/                   # raw/ interim/ processed/  (gitignored)
├── notebooks/              # exploration only — no pipeline logic lives here
└── tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps logparser3   # see note below
pip install -e .
```

> **Note on `logparser3`**: it pins `regex==2022.3.2`, which has no wheels for
> Python ≥ 3.11 and fails to build from source. Installing with `--no-deps` (its
> actual dependencies — pandas, a modern `regex`, etc. — already come from
> `requirements.txt`) works fine; Drain is unaffected.

> **Note on `tick`** (Hawkes library): its prebuilt wheels stop at Python 3.8/3.9 and it
> rarely builds from source on modern Pythons. If it fails to install, the fallback —
> already planned — is a direct implementation of the exponential-kernel Hawkes
> log-likelihood (it has a well-known O(n) recursive form) fit with `scipy.optimize`.
> The validation via time-rescaling is library-independent.

## Running the pipeline

Each stage is a script; each reads the previous stage's output from `data/`.

```bash
# 1. Get data. --sample = 2k-line sample from GitHub (fast, for development);
#    --full = full dataset from Zenodo.
python scripts/download.py --dataset HDFS --sample
python scripts/download.py --dataset HDFS --full      # HDFS_v1, ~1.5 GB + labels
python scripts/download.py --dataset BGL  --full      # BGL, ~700 MB, labels inline

# 2. Parse raw logs into structured events with Drain
python scripts/parse.py --dataset HDFS --input data/raw/HDFS.log
python scripts/parse.py --dataset BGL  --input data/raw/BGL.log

# 3. Explore + quality-check (writes figures to reports/figures/)
python scripts/explore_hdfs.py --input data/interim/HDFS.log_parsed.csv

# 4. HDFS detection: block-session features + strict temporal split
python scripts/build_features.py \
    --input data/interim/HDFS.log_parsed.csv --labels data/raw/anomaly_label.csv
python scripts/run_baseline.py          # Isolation Forest baseline  (PR-AUC 0.748)
python scripts/train_autoencoder.py     # autoencoder detector       (PR-AUC 0.987)

# 5. BGL Hawkes centerpiece: fit + branching ratio + time-rescaling GOF
python scripts/fit_hawkes.py --start 2005-06-11 --hours 24 --stream all

# 6. Adaptive thresholding (stabilise false alarms under drift)
python scripts/run_evt.py --q 0.03

# 7. LLM triage: flagged blocks -> deduped, correlated, ranked incidents (JSON)
python scripts/run_triage.py --top 200            # rule-based explainer
python scripts/run_triage.py --top 200 --llm      # Claude (claude-opus-4-8) when a key is set
```

Results and the decisions behind them are written up under `docs/`
(`eda_findings.md`, `detection_results.md`, `hawkes_results.md`).

Parsed output lands in `data/interim/` as CSV, one row per log line. HDFS rows
carry `timestamp, event_id, template, block_ids, …`; BGL rows carry
`timestamp, event_id, template, label, node, …` (BGL labels are inline).

## Data

| Dataset | Role | Source | Labels | Time resolution |
|---|---|---|---|---|
| HDFS_v1 | detection + triage | [Zenodo](https://zenodo.org/records/8196385) via loghub | per-block (`anomaly_label.csv`, ~2.9%) | 1 s |
| BGL | Hawkes centerpiece | Zenodo via loghub | per-line (`Label` field, `-` = normal) | microsecond |

HDFS log lines look like:

```
081109 203515 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
```

Drain (loghub benchmark configuration) turns these into a fixed template
vocabulary; each line becomes `(timestamp, event_id, block_id)`. The `block_id`
is the HDFS session key (labels are per block). BGL lines are labeled inline and
carry microsecond timestamps — see `docs/eda_findings.md` for why that split of
roles (HDFS for detection, BGL for the Hawkes process) is driven by the data.
