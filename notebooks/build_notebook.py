"""Construct the full-pipeline analysis notebook (notebooks/full_pipeline.ipynb).

Builds a markdown-rich notebook that runs the entire project end to end using
the logtriage package, so every table and chart is visible and analysable.
Run:  python notebooks/build_notebook.py   (then execute with nbclient)
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------- title
md(r"""# Log Anomaly Detection with Hawkes-Process Timing and LLM Triage
### Full pipeline notebook — data → parsing → EDA → detection → Hawkes → ablation → adaptive thresholding → triage

**Author:** El Alami Amine — Internship at OCP Group, Digital Infrastructure and Operations.

This notebook walks the *entire* project, stage by stage, with the reasoning (**what / why / how**) in markdown
and every computation visible. It imports the project package `logtriage` (installed with `pip install -e .`),
so the logic executed here is *identical* to the pipeline scripts — nothing is re-implemented.

**Pipeline overview**

```
raw logs ──► Drain parsing ──► features (count vectors / windows) ──► detection (IsoForest vs Autoencoder)
                 │                                                            │
                 └──► EDA & data-quality checks                               ▼
BGL timestamps ──► Hawkes process (MLE, branching ratio, time-rescaling) ──► adaptive thresholding ──► LLM triage
```

**Datasets** (public, loghub): **HDFS** (11.2M lines, per-block labels, 1-second timestamps) hosts detection & triage;
**BGL** (4.7M lines, per-line labels, microsecond timestamps) hosts the Hawkes timing model.
Heavy artifacts (full parses, trained scores) were produced by the pipeline scripts and are loaded here;
light computations (fits, metrics, plots) are re-executed live.""")

# ----------------------------------------------------------------- setup
md(r"""## 0. Setup

We import the standard stack and the project package. `PROJECT_ROOT` anchors all data paths.""")
code(r"""%matplotlib inline
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)

from logtriage.config import PROJECT_ROOT, DATA_RAW, DATA_INTERIM, DATA_PROCESSED, HDFS, BGL
print("project root:", PROJECT_ROOT)""")

# ----------------------------------------------------------------- parsing
md(r"""## 1. Data & parsing (Drain)

**What.** Raw log lines are free text. *Log parsing* discovers the finite vocabulary of **templates**
(event types) and converts each line into a structured record `(timestamp, event_id, template, …)`.

**Why.** Statistics and ML need categorical events, not strings. After parsing, the stream of millions of
text lines becomes a sequence over ~48 (HDFS) / ~429 (BGL) event types.

**How.** We use **Drain** (He et al., ICWS 2017) with the published loghub benchmark configuration
(fixed tree depth 4, similarity 0.5, masking of variable tokens like block ids / IPs), so our templates are
comparable to the literature. Below: a raw line, and the head of the parsed HDFS sample.""")
code(r"""print(open(DATA_RAW/"HDFS_2k.log").readline().strip()[:120], "…")
sample = pd.read_csv(DATA_INTERIM/"HDFS_2k.log_parsed.csv", parse_dates=["timestamp"], keep_default_na=False)
sample.head(5)""")
md(r"""**Correctness invariant.** On the *full* HDFS parse (11,175,629 lines → 48 templates), the template
`BLOCK* NameSystem.allocateBlock` fires exactly once per block. Its count equals the number of labelled
blocks **exactly** (575,061) — a structural proof that parsing and label alignment are correct.""")
code(r"""tmpl = pd.read_csv(DATA_INTERIM/"HDFS.log_templates.csv").sort_values("Occurrences", ascending=False)
print("full-parse templates:", len(tmpl))
tmpl.head(8)""")

# ----------------------------------------------------------------- EDA
md(r"""## 2. Exploratory data analysis

Three EDA findings **redirected the design** — we reproduce the key measurements here.

### 2.1 Data-quality report (sample)
`logtriage.eda.data_quality_report` checks completeness, duplicates, time ordering, timestamp
resolution/ties, and block coverage.""")
code(r"""from logtriage.eda import data_quality_report, print_report
print_report(data_quality_report(sample))""")
md(r"""### 2.2 Finding 1 — events cluster (self-excitation is real)

The **Fano factor** = var/mean of per-minute event counts; a memoryless Poisson process has Fano = 1.
Values ≫ 1 mean bursts. This justifies a *self-exciting* point process **before any model is fit**.""")
code(r"""per_min = sample.set_index("timestamp").resample("1min").size()
fano = per_min.var()/per_min.mean()
fig, ax = plt.subplots(figsize=(11,3))
ax.plot(per_min.index, per_min.values, lw=.8, color="#C44E52")
ax.set_title(f"HDFS sample — events/minute (Fano factor = {fano:.1f}; Poisson would be 1.0)")
ax.set_ylabel("events/min"); plt.show()
print(f"Fano factor (sample) = {fano:.2f}   |   full BGL windows measured at 698–2286")""")
md(r"""### 2.3 Finding 2 — HDFS timestamps cannot support a continuous-time model *(decisive)*

HDFS records time to the **second** while the full stream averages ~80 events/s. Measured on the full parse:
**98.8%** of consecutive events share a timestamp; within one block **90.5%** still tie; every block lives
≤ 54 s. A continuous-time Hawkes process needs a strict event ordering — absent here. Jittering timestamps
would *fabricate* the sub-second structure the model is meant to measure, so it was **rejected**; the Hawkes
analysis moved to **BGL** (microsecond stamps, 0% ties). We verify the tie rate on the sample:""")
code(r"""gaps = sample["timestamp"].diff().dropna().dt.total_seconds()
print(f"sample same-second tie fraction: {(gaps==0).mean():.1%}  (full HDFS: 98.8% global, 90.5% within-block)")""")
md(r"""### 2.4 Finding 3 — the anomaly base rate drifts

Loaded from the feature stage below: the earlier 70% of blocks are **3.26%** anomalous, the later 30%
**2.16%** — the world is non-stationary, which foreshadows the drift results in §4–§6.

Full-data EDA figures (produced by `scripts/explore_hdfs.py` on all 11.2M lines):""")
code(r"""from IPython.display import Image, display
display(Image(str(PROJECT_ROOT/"docs/figures/HDFS_timeline.png"), width=880))
display(Image(str(PROJECT_ROOT/"docs/figures/HDFS_block_sizes.png"), width=600))""")

# ----------------------------------------------------------------- features/split
md(r"""## 3. Features and the temporal split (HDFS)

**Unit of analysis = block session.** HDFS ground truth labels *blocks* (Normal/Anomaly), so each block's
lines are grouped and represented by a **48-dim template-count vector** — the classic "event count matrix".

**Validation protocol (the heart of the project).**
* **Temporal split, never shuffled**: earliest 70% train / latest 30% test → no leakage from the future.
* Autoencoder trains on **normal training blocks only**; labels are used exclusively to evaluate.
* Metrics: **PR-AUC** (chance = anomaly rate; accuracy is meaningless at ~2% positives) and
  **false alarms/day at 90% recall** on the *actual* elapsed test time; plus **per-period drift**.""")
code(r"""meta = pd.read_csv(DATA_PROCESSED/"hdfs_meta.csv", index_col=0, parse_dates=["start","end"])
X = np.load(DATA_PROCESSED/"hdfs_counts.npy")
tr, te = meta[meta.split=="train"], meta[meta.split=="test"]
print(f"count matrix: {X.shape[0]:,} blocks × {X.shape[1]} templates")
print(f"train: {len(tr):,} blocks, anomaly rate {tr.label.mean():.3%}")
print(f"test : {len(te):,} blocks, anomaly rate {te.label.mean():.3%}   (rate drift ×{te.label.mean()/tr.label.mean():.2f})")
print(f"split boundary: {tr.start.max()}   test span: {te.start.min()} → {te.start.max()}")
meta.head(3)""")

# ----------------------------------------------------------------- detection
md(r"""## 4. Detection: Isolation Forest baseline vs Autoencoder

**Isolation Forest (baseline).** Anomalies are "few and different" → random axis-aligned splits isolate them
in fewer steps; the average isolation depth becomes the score. Unsupervised, fast — the standard detector to beat.

**Autoencoder (detector).** MLP `48→32→16→8→16→32→48`, trained **only on normal training blocks** to
reconstruct their (log-transformed, standardised) count vectors; the per-block **reconstruction MSE** is the
anomaly score — an expert at normal patterns fumbles on patterns it never saw.

Both models were trained by the pipeline scripts; their **test scores** are loaded here and all evaluation is
recomputed live with the same `logtriage.eval.metrics` code.""")
code(r"""from sklearn.metrics import precision_recall_curve, average_precision_score
from logtriage.eval.metrics import false_alarms_per_day, drift_report

y  = te["label"].to_numpy()
ts = te["start"]
s_if = np.load(DATA_PROCESSED/"baseline_scores_test.npy")
s_ae = np.load(DATA_PROCESSED/"ae_scores_test.npy")

fig, ax = plt.subplots(figsize=(6.5,5))
for name, s, c in [("Isolation Forest", s_if, "#8172B3"), ("Autoencoder", s_ae, "#C44E52")]:
    p, r, _ = precision_recall_curve(y, s)
    ax.plot(r, p, c, lw=2, label=f"{name}  (PR-AUC={average_precision_score(y, s):.3f})")
ax.axhline(y.mean(), ls="--", c="gray", lw=1, label=f"chance ({y.mean():.3f})")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(loc="lower left")
ax.set_title("HDFS test split — precision-recall"); plt.show()

for name, s in [("Isolation Forest", s_if), ("Autoencoder", s_ae)]:
    fa, op = false_alarms_per_day(y, s, ts, target_recall=0.90)
    print(f"{name:17s}: PR-AUC {average_precision_score(y,s):.3f} | precision@90%rec {op.precision:.3f} | false alarms/day {fa:,.0f}")""")
md(r"""**Reading.** The autoencoder (PR-AUC **0.987** vs chance 0.022) cuts false alarms **~20×** at the same
90% detection rate. *Positioning*: this reproduces a heavily-published benchmark result — its role is to
validate the pipeline and feed realistic scores to the later stages, not to claim novelty.

### 4.1 Drift decomposition
Aggregate numbers can hide instability. Same threshold, four consecutive time slices:""")
code(r"""print("Isolation Forest:"); display(drift_report(y, s_if, ts, n_bins=4, target_recall=0.90).round(3))
print("Autoencoder:");      display(drift_report(y, s_ae, ts, n_bins=4, target_recall=0.90).round(3))""")
md(r"""The baseline's **ranking quality itself swings** (PR-AUC 0.69→0.98) while the autoencoder holds ≥0.99 —
but even for the autoencoder the **false-alarm rate at a fixed threshold varies by orders of magnitude**
across slices. Ranking is stable; *alerting* is not → §6 fixes this.""")

# ----------------------------------------------------------------- Hawkes theory + sim
md(r"""## 5. The mathematical core — Hawkes process

**Model.** Conditional intensity (instantaneous event rate given the past):

$$\lambda(t)=\mu+\sum_{t_i<t}\alpha e^{-\beta(t-t_i)},\qquad \mu,\alpha,\beta>0$$

Each event bumps the rate by $\alpha$; the bump decays with timescale $1/\beta$. The **branching ratio**
$\eta=\int_0^\infty \alpha e^{-\beta s}ds=\alpha/\beta$ is the mean number of *direct offspring* per event:
$\eta<1$ → stable (subcritical), $\eta\ge 1$ → explosive cascades.

**Estimation.** Maximum likelihood: $\ \ell=\sum_i\log\lambda(t_i)-\int_0^T\lambda$, with the integral in
closed form and the sum computed in **O(n)** by Ogata's recursion $A_i=e^{-\beta\Delta t_i}(1+A_{i-1})$.
Here is the actual implementation used everywhere in the project:""")
code(r"""import inspect
from logtriage.hawkes import exp_kernel
print(inspect.getsource(exp_kernel._recursion_A))
print(inspect.getsource(exp_kernel.neg_log_likelihood))""")
md(r"""### 5.1 Correctness before trust — recovery from simulation

A fitted number is worthless unless the fitter is proven. We simulate a Hawkes process with **known**
parameters (Ogata thinning) and check the MLE recovers them — the same test that gates the estimator in
the automated test suite.""")
code(r"""from logtriage.hawkes.exp_kernel import simulate, fit
from logtriage.hawkes.validation import rescaled_gaps, ks_test_exp1, qq_points

mu0, a0, b0 = 0.8, 1.2, 2.0          # truth: eta = 0.6
ev = simulate(mu0, a0, b0, T=20_000, seed=1)
f_sim = fit(ev)
print(f"simulated {len(ev):,} events   truth: mu={mu0}, alpha={a0}, beta={b0}, eta={a0/b0:.2f}")
print(f"recovered: mu={f_sim.mu:.3f}, alpha={f_sim.alpha:.3f}, beta={f_sim.beta:.3f}, eta={f_sim.branching_ratio:.3f}")

gof = ks_test_exp1(rescaled_gaps(ev, f_sim))
print(f"time-rescaling on the TRUE model: KS={gof.ks_stat:.4f}, mean gap={gof.mean_gap:.3f}  (≈Exp(1) ⇒ fit accepted)")""")
md(r"""### 5.2 Validation tool — the time-rescaling theorem

If the fitted intensity is correct, the transformed gaps $\Delta\tau_i=\Lambda(t_i)-\Lambda(t_{i-1})$
(where $\Lambda(t)=\int_0^t\lambda$) are i.i.d. **Exp(1)**. So model checking = "do these rescaled gaps
look like a fair exponential?" — assessed with a QQ-plot (diagonal = good) and the KS statistic.
The sample mean of the gaps must be exactly 1 under a correctly-normalised compensator (an independent
implementation check).

### 5.3 Fit on real data — BGL (microsecond timestamps)""")
code(r"""bgl = pd.read_csv(DATA_INTERIM/"BGL.log_parsed.csv", usecols=["timestamp","label"], parse_dates=["timestamp"])
bgl = bgl.sort_values("timestamp").reset_index(drop=True)
print(f"BGL: {len(bgl):,} events over {(bgl.timestamp.max()-bgl.timestamp.min()).days} days, {bgl.label.mean():.2%} alert lines")
d = bgl.timestamp.diff().dropna().dt.total_seconds()
print(f"timestamp ties: {(d==0).mean():.4%}  (HDFS was 98.8% — this is why Hawkes lives on BGL)")""")
code(r"""from logtriage.hawkes.exp_kernel import intensity_on_grid

def window_events(day, hours=24):
    t0 = pd.Timestamp(day); m = (bgl.timestamp>=t0)&(bgl.timestamp<t0+pd.Timedelta(hours=hours))
    tt = bgl.loc[m,"timestamp"]; return (tt - tt.iloc[0]).dt.total_seconds().to_numpy()

t_calm = window_events("2005-09-03")
f_calm = fit(t_calm)
print(f"calm window 2005-09-03: {len(t_calm):,} events")
print(f"  mu={f_calm.mu:.4g}, alpha={f_calm.alpha:.4g}, beta={f_calm.beta:.4g}  →  branching ratio eta={f_calm.branching_ratio:.3f}")
gof_c = ks_test_exp1(rescaled_gaps(t_calm, f_calm))
print(f"  time-rescaling: KS={gof_c.ks_stat:.3f}, mean gap={gof_c.mean_gap:.4f}")""")
code(r"""grid = np.linspace(0, t_calm[-1], 3000)
lam = intensity_on_grid(t_calm, grid, f_calm.mu, f_calm.alpha, f_calm.beta)
fig, axes = plt.subplots(1, 2, figsize=(13,4), gridspec_kw={"width_ratios":[2,1]})
axes[0].plot(grid, lam, lw=.7, color="#C44E52")
axes[0].set_title(f"Fitted conditional intensity λ(t) — 2005-09-03 (η={f_calm.branching_ratio:.3f})")
axes[0].set_xlabel("seconds"); axes[0].set_ylabel("events/s")
theo, emp = qq_points(rescaled_gaps(t_calm, f_calm))
axes[1].scatter(theo, emp, s=3, alpha=.3, color="#4C72B0"); hi=float(max(theo[-1],emp[-1]))
axes[1].plot([0,hi],[0,hi],"k--",lw=1); axes[1].set_title("Time-rescaling QQ vs Exp(1)")
axes[1].set_xlabel("theoretical"); axes[1].set_ylabel("empirical"); plt.tight_layout(); plt.show()""")
md(r"""### 5.4 The extreme cascade window, and the honest reading

The same fit on the **2005-06-11 fault cascade** (152,669 events) gives (recomputed below —
takes ~1 minute):""")
code(r"""t_casc = window_events("2005-06-11")
f_casc = fit(t_casc)
gof_k = ks_test_exp1(rescaled_gaps(t_casc, f_casc))
summary = pd.DataFrame({
 "window":["2005-09-03 (calm)","2005-06-11 (cascade)"],
 "events":[len(t_calm),len(t_casc)],
 "eta = alpha/beta":[round(f_calm.branching_ratio,3), round(f_casc.branching_ratio,4)],
 "KS stat":[round(gof_c.ks_stat,3), round(gof_k.ks_stat,3)],
 "mean rescaled gap":[round(gof_c.mean_gap,4), round(gof_k.mean_gap,4)]})
summary""")
md(r"""**Headline finding.** The branching ratio is **near-critical (η ≈ 0.97–1.0)** on both windows: each BGL
event triggers ≈1 direct offspring — the stream sits at the edge of self-sustaining cascades. A precise,
validated statement, far stronger than "logs are bursty".

**Honest caveat.** The QQ body lies on the diagonal (bulk fits), but the **upper tail bends upward**: the
deepest quiet gaps are longer than one exponential allows — after a burst, the model over-predicts continued
activity. Mild when calm (KS 0.095), severe in the cascade (KS 0.43). One timescale cannot fit both tight
bursts and deep lulls → motivates a **sum-of-exponentials kernel** (future work). At n~10⁵ the KS *p*-value
rejects any parametric model, so the QQ shape and KS magnitude are the meaningful diagnostics.""")

# ----------------------------------------------------------------- ablation
md(r"""## 6. Does timing actually help detection? — the ablation

The Hawkes model earns its place only if its signal **helps**. On fixed 60-s BGL windows we run the *same*
detector (Isolation Forest) under the *same* temporal split on three feature sets:
* **content** — counts of the top-50 templates (what happened);
* **timing** — the self-excitation level $A(t)=\sum e^{-\beta\Delta t}$ (mean/max at 1 s & 30 s scales) + event count (when/how bursty) — computed from *times only*, so leakage-free;
* **content+timing** — both.

(~2–3 min: one pass over 4.7M events for the excitation recursion.)""")
code(r"""from logtriage.features.windows import build_window_features
from logtriage.models.baseline import IsolationForestBaseline
from logtriage.eval.metrics import pr_auc

ev3 = pd.read_csv(DATA_INTERIM/"BGL.log_parsed.csv", usecols=["timestamp","event_id","label"],
                  parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
vocab = ev3.iloc[:int(.7*len(ev3))]["event_id"].value_counts().head(50).index.tolist()
Xc, Xt, wmeta = build_window_features(ev3, window_s=60, top_k=50, top_templates=vocab)
yw = wmeta["label"].to_numpy().astype(int); n_tr = int(.7*len(wmeta))
is_tr = np.arange(len(wmeta)) < n_tr; is_te = ~is_tr; normal_tr = is_tr & (yw==0)
Xc, Xt = Xc.to_numpy(np.float32), Xt.to_numpy(np.float32)
res = {}
for name, F in [("content",Xc), ("timing",Xt), ("content+timing",np.hstack([Xc,Xt]))]:
    m = IsolationForestBaseline(random_state=0).fit(F[normal_tr]); s = m.score(F[is_te])
    fa, op = false_alarms_per_day(yw[is_te], s, wmeta["start"][is_te], target_recall=0.90)
    res[name] = dict(pr_auc=round(pr_auc(yw[is_te],s),3), precision_at_90rec=round(op.precision,3), fa_per_day=round(fa,1))
del ev3
pd.DataFrame(res).T""")
md(r"""**Honest, two-sided reading.** Absolute detection is *weak* (all ≈0.09 PR-AUC vs 0.047 chance — a 60-s
window dilutes a few alert lines in heavy traffic; the task is genuinely hard). **Relatively**, timing is
the strongest signal: at 90% recall it **halves the false alarms** vs content and ~doubles precision.
The Hawkes-derived feature carries operational information that counts do not — a real, measured (if
modest) win, reported without overselling.""")

# ----------------------------------------------------------------- EVT
md(r"""## 7. Adaptive thresholding — stabilising the false-alarm rate under drift

**Problem (measured in §4.1).** With a threshold fixed once, the autoencoder's false alarms on HDFS swing
from ~3.7k to ~862k/day across five hours — undeployable.

**Principled tool: EVT / Peaks-Over-Threshold.** Exceedances over a high threshold follow a Generalized
Pareto law (Pickands–Balkema–de Haan); fitting its tail and inverting gives the level $z_q$ crossed with
target probability $q$, re-estimated on a trailing window → a self-calibrating threshold (SPOT).

**Honest negative finding.** HDFS scores are **discrete** — 58.7% of normal blocks share *one identical*
reconstruction error (normal sessions are byte-identical) — violating GPD's continuous-tail hypothesis; the
extrapolation misbehaves. Fix: a **distribution-free rolling quantile** on HDFS; EVT is reserved for
continuous scores (e.g. the BGL timing features). Below, fixed vs adaptive, live:""")
code(r"""from logtriage.eval.evt import RollingQuantile
test_sorted = te.copy(); test_sorted["raw"] = s_ae; test_sorted = test_sorted.sort_values("start")
ls = np.log(np.maximum(test_sorted["raw"].to_numpy(), 1e-12))
y_s = test_sorted["label"].to_numpy(); ts_s = pd.to_datetime(test_sorted["start"].to_numpy())
calib_n, q = 20_000, 0.03
calib, rest, y_r, ts_r = ls[:calib_n], ls[calib_n:], y_s[calib_n:], ts_s[calib_n:]
z_fix = np.quantile(calib, 1-q); fl_fix = rest > z_fix
rq = RollingQuantile(q=q, window=20_000, stride=500).fit(calib); fl_ad, thr_ad = rq.run(rest)

def fa_series(flags, nb=30):
    edges = pd.date_range(ts_r.min(), ts_r.max(), periods=nb+1); mid, fa = [], []
    for i in range(nb):
        m = np.asarray((ts_r>=edges[i]) & (ts_r<edges[i+1])) if i<nb-1 else np.asarray((ts_r>=edges[i]) & (ts_r<=edges[i+1]))
        if m.sum()==0: continue
        days = (edges[i+1]-edges[i]).total_seconds()/86400
        mid.append(edges[i]); fa.append(int((flags[m] & (y_r[m]==0)).sum())/days)
    return mid, fa

fig, ax = plt.subplots(figsize=(11,3.5))
for fl, cme, lab in [(fl_fix,"#8172B3","fixed threshold"), (fl_ad,"#C44E52","adaptive (rolling quantile)")]:
    mid, fa = fa_series(fl); ax.plot(mid, fa, color=cme, marker="o", ms=3, lw=1.5, label=lab)
ax.set_ylabel("false alarms / day"); ax.legend(); ax.set_title(f"False-alarm stability under drift (target flag rate q={q:.0%})")
plt.show()
for fl, lab in [(fl_fix,"fixed"), (fl_ad,"adaptive")]:
    rec = (fl & (y_r==1)).sum()/max((y_r==1).sum(),1)
    print(f"{lab:9s}: flagged {fl.mean():.1%} | recall over period {rec:.2f}")""")
md(r"""The adaptive threshold tracks the drifting score distribution and **holds the alert budget** while the
fixed threshold explodes mid-period. The trade-off is explicit: holding the budget costs recall
(1.00 → ~0.72) when true anomalies exceed it — a *transparent, tunable* operating point (the knob is $q$),
instead of a hidden failure.""")

# ----------------------------------------------------------------- triage
md(r"""## 8. Triage — from flags to ranked incidents

Paging an analyst once per flag is unusable. The agent applies **deterministic reduction, then explanation**:
1. **Deduplicate** — identical signatures (entity + template multiset) merge, keeping a count;
2. **Correlate** — flags close in time form one **incident**: the operational translation of the Hawkes
   finding (a near-critically self-exciting stream ⇒ temporally clustered anomalies are cascade members
   sharing a root cause);
3. **Explain & rank** — each incident becomes strict JSON (`priority, category, summary, recommended_action,
   explanation`), ranked P1→P4. The explainer is pluggable: a deterministic rule-based engine (used here)
   or an LLM backend with schema-constrained output (one call per *incident*, not per flag).

(~2 min: pulls template context for the top-200 flagged blocks from the full parsed HDFS log.)""")
code(r"""import sys; sys.path.insert(0, str(PROJECT_ROOT/"scripts"))
from run_triage import build_anomalies
from logtriage.triage.agent import TriageAgent

anoms = build_anomalies(top=200)
agent = TriageAgent(gap_seconds=30)
incidents = agent.run(anoms)
inc_df = pd.DataFrame(agent.to_records(incidents))
print(f"{len(anoms)} raw flags → {len(inc_df)} incidents")
fig, ax = plt.subplots(figsize=(4.5,3))
inc_df["priority"].value_counts().sort_index().plot.bar(ax=ax, color="#4C72B0")
ax.set_title("Incidents by priority"); ax.set_ylabel("count"); plt.show()
inc_df[["id","priority","category","max_score","n_anomalies","dedup_count","summary"]].head(8)""")
code(r"""print(json.dumps(agent.to_records(incidents)[0], indent=2)[:900])""")

# ----------------------------------------------------------------- conclusions
md(r"""## 9. Consolidated results & conclusions

| Item | Result |
|---|---|
| HDFS parsing | 11.2M lines → 48 templates (invariant-checked) |
| BGL parsing | 4.71M lines → 429 templates |
| Detection (HDFS test) | PR-AUC **0.987** vs 0.748 baseline; **~20× fewer** false alarms @ 90% recall |
| Hawkes branching ratio (BGL) | **η ≈ 0.97–1.0** — near-critical self-excitation, validated by time-rescaling |
| Kernel limitation | single-exponential underfits deepest lulls (QQ upper tail) → multi-scale kernel next |
| Timing ablation (BGL) | timing features **halve false alarms** vs content, on a hard task (PR-AUC ≈ 0.09 abs.) |
| Thresholding under drift | FA/day swing ×235 with fixed threshold → **held near budget** adaptively |
| Triage | 200 flags → **90 ranked, explained incidents** |

**Methodological through-line.** Estimators proven on synthetic data before real data; strictly out-of-time
evaluation; the operationally meaningful metric (false alarms/day) as headline; drift measured, not assumed
away; negative findings (HDFS timestamps, EVT-on-discrete-scores, kernel tail misfit) reported as
first-class results. That discipline — more than any single number — is what makes each claim defensible.""")

nb.cells = C
nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb, "notebooks/full_pipeline.ipynb")
print("notebook written:", len(C), "cells")
