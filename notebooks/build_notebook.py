"""Construct the fully SELF-CONTAINED pipeline notebook.

Every algorithm (preprocessing, EDA, features, models, metrics, Hawkes, EVT,
triage) is implemented INSIDE the notebook cells. Only dataset files are read
from disk. Run:  python notebooks/build_notebook.py  then execute.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# Log Anomaly Detection with Hawkes-Process Timing and LLM Triage
### Fully self-contained pipeline notebook — every algorithm implemented in the cells below

**Author:** El Alami Amine — Internship at OCP Group, Digital Infrastructure and Operations.

This notebook contains the **entire project inline**: data preprocessing, EDA, feature engineering,
the two detectors, the evaluation metrics, the Hawkes process (likelihood, fitting, simulation,
validation), the timing-feature ablation, adaptive thresholding (EVT + rolling quantile), and the
triage engine. **No project modules are imported — all code is in the cells** (only standard
libraries: numpy/pandas/matplotlib/scipy/sklearn/torch, and the `logparser` pip package for the
Drain parsing demo). The only things read from disk are the **dataset files** under `data/`
(downloaded/parsed once by the repo scripts because full parsing takes ~35 min; §1 demonstrates the
same parsing live on a sample).

**Pipeline**

```
raw logs ─► Drain parsing ─► EDA & quality ─► features + temporal split ─► detection (IF vs AE)
                                                                                │
BGL timestamps ─► Hawkes (MLE, branching ratio, time-rescaling) ─► timing ablation
                                                                                ▼
                                                     adaptive thresholding ─► triage (ranked incidents)
```
""")

# ---------------------------------------------------------------- setup
md(r"""## 0. Setup

Standard imports only, and the data paths. Everything else is defined in the cells that follow.""")
code(r"""%matplotlib inline
import json, re, warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)

ROOT = Path.cwd() if (Path.cwd()/"data").exists() else Path.cwd().parent
RAW, INTERIM, PROC = ROOT/"data/raw", ROOT/"data/interim", ROOT/"data/processed"
print("project root:", ROOT)""")

# ---------------------------------------------------------------- parsing
md(r"""## 1. Data preprocessing — parsing raw logs with Drain

**What.** A raw log line is free text. *Parsing* discovers the finite vocabulary of **templates**
(event types) and converts each line into `(timestamp, event_id, template, …)`.

**Why.** Statistics needs categorical events, not strings: after parsing, 11.2M HDFS lines become a
sequence over **48** event types (BGL: 4.7M lines → **429**).

**How.** **Drain** (He et al., 2017) routes each line down a fixed-depth tree (by token count, then
leading tokens) to a template cluster, comparing token-wise similarity (threshold 0.5); variable
tokens (block ids, IPs) are pre-masked by regex. We demonstrate it **live on the 2k sample** with
the loghub benchmark configuration, then canonicalise the output ourselves.""")
code(r"""print("RAW LINE:"); print(" ", open(RAW/"HDFS_2k.log").readline().strip()[:110], "...")

from logparser.Drain import LogParser   # pip library used for the parsing step only
import tempfile, os
outdir = tempfile.mkdtemp()
LogParser(log_format="<Date> <Time> <Pid> <Level> <Component>: <Content>",
          indir=str(RAW), outdir=outdir, depth=4, st=0.5,
          rex=[r"blk_(|-)[0-9]+", r"(\d+\.){3}\d+(:\d+)?"]).parse("HDFS_2k.log")
structured = pd.read_csv(os.path.join(outdir, "HDFS_2k.log_structured.csv"),
                         dtype={"Date":str,"Time":str})
structured[["Date","Time","Level","Component","Content","EventId","EventTemplate"]].head(4)""")
md(r"""**Canonicalisation** (our code): build real datetimes from the `Date`/`Time` fields and extract the
`blk_...` identifiers each line references — the session keys that HDFS labels attach to.""")
code(r"""BLOCK_RE = re.compile(r"blk_-?\d+")

def canonicalize_hdfs(df):
    raw = df["Date"].str.zfill(6) + df["Time"].str.zfill(6)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw, format="%y%m%d%H%M%S"),
        "event_id": df["EventId"], "template": df["EventTemplate"],
        "level": df["Level"], "content": df["Content"]})
    out["block_ids"] = df["Content"].map(lambda c: " ".join(dict.fromkeys(BLOCK_RE.findall(str(c)))))
    return out

sample = canonicalize_hdfs(structured)
print(f"sample: {len(sample):,} lines -> {sample.event_id.nunique()} templates")
sample.head(3)""")
md(r"""**Scale-up + correctness invariant.** The identical code (in script form) parsed the full datasets.
A structural check proves correctness: the `allocateBlock` template must fire **exactly once per
block**, so its count must equal the number of labelled blocks — and it does, exactly.""")
code(r"""tmpl_full = pd.read_csv(INTERIM/"HDFS.log_templates.csv").sort_values("Occurrences", ascending=False)
labels_full = pd.read_csv(RAW/"anomaly_label.csv")
alloc = tmpl_full[tmpl_full.EventTemplate.str.contains("allocateBlock")]["Occurrences"].iloc[0]
print(f"full parse: {tmpl_full.Occurrences.sum():,} lines -> {len(tmpl_full)} templates")
print(f"allocateBlock occurrences = {alloc:,}  vs  labelled blocks = {len(labels_full):,}  ->  match: {alloc==len(labels_full)}")
print(f"label balance: {(labels_full.Label=='Anomaly').mean():.2%} anomalous blocks")
tmpl_full.head(6)""")

# ---------------------------------------------------------------- EDA
md(r"""## 2. Exploratory data analysis & data quality

Three findings **redirected the design**. All checks are computed here, inline.

### 2.1 Data-quality checks""")
code(r"""def data_quality(df):
    gaps = df["timestamp"].diff().dropna().dt.total_seconds()
    return pd.Series({
        "rows": len(df),
        "null_cells": int(df.drop(columns=["block_ids"]).replace("", np.nan).isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "time_sorted": bool(df["timestamp"].is_monotonic_increasing),
        "second_resolution_only": bool((df["timestamp"].dt.microsecond == 0).all()),
        "same_second_tie_fraction": float((gaps == 0).mean()),
        "lines_without_block": int((df["block_ids"] == "").sum()),
    })
data_quality(sample)""")
md(r"""### 2.2 Finding 1 — events cluster (self-excitation is real)

**Fano factor** = variance/mean of per-minute counts. Poisson (memoryless) ⇒ exactly 1.
Measured ≫ 1 ⇒ bursts. This justifies a self-exciting model *before any fit*.""")
code(r"""per_min = sample.set_index("timestamp").resample("1min").size()
fano = per_min.var()/per_min.mean()
fig, ax = plt.subplots(figsize=(11,3))
ax.plot(per_min.index, per_min.values, lw=.8, color="#C44E52")
ax.set_title(f"HDFS sample — events/minute  (Fano = {fano:.1f};  Poisson would be 1.0)")
ax.set_ylabel("events/min"); plt.show()
print(f"Fano factor: sample {fano:.1f}   (full-BGL windows measure 698-2286)")""")
md(r"""### 2.3 Finding 2 — HDFS timestamps cannot support a continuous-time model *(decisive)*

HDFS stamps to the **second** at ~80 events/s ⇒ consecutive events tie constantly. A Hawkes process
needs a strict event ordering; adding artificial jitter would fabricate the very structure we want
to measure — **rejected**. We verify on the full parsed file (timestamps only, streamed):""")
code(r"""ts_full = pd.read_csv(INTERIM/"HDFS.log_parsed.csv", usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"]
g = ts_full.diff().dropna().dt.total_seconds()
print(f"full HDFS: {len(ts_full):,} events over {(ts_full.max()-ts_full.min())}")
print(f"same-second ties between consecutive events: {(g==0).mean():.1%}   -> continuous-time Hawkes ill-posed on HDFS")
del ts_full, g""")
md(r"""⇒ **Consequence:** the Hawkes analysis lives on **BGL** (microsecond stamps — verified in §5.3 to
have ~0% ties). HDFS keeps the detection role (its per-block labels are ideal for that).

### 2.4 Finding 3 — the anomaly base rate drifts
Computed in §3 after the split: train period 3.26% anomalous vs test 2.16% — non-stationarity that
foreshadows every drift result below.""")

# ---------------------------------------------------------------- features
md(r"""## 3. Feature engineering + temporal split — implemented and run on the full 11.2M lines

**Unit = block session** (labels are per block). Each block → a **48-dim template-count vector**.

**The split discipline (heart of the project):** order blocks by *start time*; earliest 70% train,
latest 30% test; **never shuffled** — nothing from the test period (labels, scalers, vocabularies,
thresholds) may influence training. Below: session building, count matrix, and split — all inline.
(~3–5 min on the full data.)""")
code(r"""events = pd.read_csv(INTERIM/"HDFS.log_parsed.csv",
                     usecols=["timestamp","event_id","block_ids"],
                     parse_dates=["timestamp"], keep_default_na=False)
print(f"loaded {len(events):,} events")

# one row per (line, block): a line naming several blocks belongs to each session
per_block = events.assign(block_id=events.block_ids.str.split()).explode("block_id")
per_block = per_block[per_block.block_id.astype(bool)]

# 48-dim count matrix + per-session metadata
Xdf = pd.crosstab(per_block.block_id, per_block.event_id).sort_index(axis=1)
meta = per_block.groupby("block_id").agg(start=("timestamp","min"),
                                         end=("timestamp","max"),
                                         n_events=("event_id","size"))
lab = labels_full.set_index("BlockId")["Label"].map({"Normal":0,"Anomaly":1})
meta = meta.join(lab.rename("label")).dropna(subset=["label"]); meta["label"]=meta["label"].astype(int)
Xdf = Xdf.loc[meta.index]

# temporal split by session start time — never shuffled
order = meta.start.sort_values(kind="stable").index
n_train = int(round(len(order)*0.70)); train_ids = set(order[:n_train])
meta["split"] = ["train" if b in train_ids else "test" for b in meta.index]
meta, Xdf = meta.loc[order], Xdf.loc[order]
X = Xdf.to_numpy(np.float32)

tr, te = meta[meta.split=="train"], meta[meta.split=="test"]
print(f"count matrix: {X.shape[0]:,} blocks x {X.shape[1]} templates")
print(f"train {len(tr):,} blocks ({tr.label.mean():.3%} anomalous) | test {len(te):,} ({te.label.mean():.3%})")
print(f"boundary: {tr.start.max()}   test span: {te.start.min()} -> {te.start.max()}")
del events, per_block
meta.head(3)""")
md(r"""### 3.1 Block-session structure
Median **19 events/block**; the long tail is disproportionately anomalous (failing blocks emit extra
retry/error lines) — visible below.""")
code(r"""fig, axes = plt.subplots(1,2, figsize=(12,3.4))
axes[0].hist(meta.n_events, bins=range(1,60), color="#CCB974", edgecolor="white")
axes[0].set_title(f"events per block (median {meta.n_events.median():.0f})"); axes[0].set_xlabel("events")
m = meta.groupby(pd.cut(meta.n_events, [0,10,20,30,50,300]))["label"].mean()
axes[1].bar([str(i) for i in m.index], m.values, color="#C44E52")
axes[1].set_title("anomaly rate by session size"); axes[1].set_ylabel("P(anomaly)")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- metrics
md(r"""## 4. Evaluation metrics — implemented inline

Positives are ~2% ⇒ **accuracy is useless** (the always-normal classifier scores 98%). We implement:
* **PR-AUC** — ranking quality; its *chance level equals the anomaly rate* (0.022 here);
* **operating point at fixed recall** — smallest top-scored set catching 90% of anomalies;
* **false alarms/day** at that recall, using the *measured* test duration;
* **drift report** — same metrics on consecutive time slices, threshold fixed once.""")
code(r"""from sklearn.metrics import average_precision_score, precision_recall_curve

def operating_point(labels, scores, target_recall=0.90):
    labels = np.asarray(labels); scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable"); y_sorted = labels[order]
    tp = np.cumsum(y_sorted); P = int(labels.sum())
    k = int(np.searchsorted(tp / P, target_recall)); k = min(k, len(scores)-1)
    return dict(threshold=float(scores[order][k]), recall=float(tp[k]/P),
                precision=float(tp[k]/(k+1)), tp=int(tp[k]), fp=int(k+1-tp[k]))

def false_alarms_per_day(labels, scores, times, target_recall=0.90):
    op = operating_point(labels, scores, target_recall)
    days = (pd.to_datetime(times).max()-pd.to_datetime(times).min()).total_seconds()/86400
    return op["fp"]/max(days,1e-9), op

def drift_report(labels, scores, times, n_bins=4, target_recall=0.90):
    t = pd.to_datetime(pd.Series(times).reset_index(drop=True))
    labels = np.asarray(labels); scores = np.asarray(scores, float)
    thr = operating_point(labels, scores, target_recall)["threshold"]
    edges = pd.date_range(t.min(), t.max(), periods=n_bins+1); rows=[]
    for i in range(n_bins):
        m = ((t>=edges[i]) & ((t<edges[i+1]) if i<n_bins-1 else (t<=edges[i+1]))).to_numpy()
        yl, sc = labels[m], scores[m]
        days = (edges[i+1]-edges[i]).total_seconds()/86400
        fl = sc >= thr
        rows.append(dict(bin=i+1, n=int(m.sum()), anomaly_rate=float(yl.mean()),
                         pr_auc=float(average_precision_score(yl, sc)) if 0<yl.sum()<len(yl) else np.nan,
                         recall=float((fl & (yl==1)).sum()/max(yl.sum(),1)),
                         fa_per_day=float((fl & (yl==0)).sum()/days)))
    return pd.DataFrame(rows)
print("metrics defined")""")

# ---------------------------------------------------------------- models
md(r"""## 5. Detection models — implemented and trained here

### 5.1 Isolation Forest (baseline)
Anomalies are "few and different": random axis-aligned splits isolate them in fewer steps; average
isolation depth ⇒ score. We wrap sklearn's implementation with a `log1p` transform (counts are
heavy-tailed) and score orientation "higher = more anomalous".

*Training-set note:* the IF is fit on the **full unlabelled training period** (its standard usage —
the algorithm tolerates the ~3% contamination by design and needs the anomalous region represented
to isolate it), whereas the autoencoder (§5.2) uses **normal blocks only** (it must model pure
normality). Both respect the leakage rules: no test-period data, no labels used by the IF.""")
code(r"""from sklearn.ensemble import IsolationForest

class IFDetector:
    def __init__(self, seed=0): self.clf = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
    def fit(self, Xn): self.clf.fit(np.log1p(Xn)); return self
    def score(self, Xs): return -self.clf.score_samples(np.log1p(Xs))

is_train = (meta.split=="train").to_numpy(); is_test = ~is_train
y_all = meta.label.to_numpy(); normal_train = is_train & (y_all==0)
y = y_all[is_test]; ts = meta.start[is_test]

if_model = IFDetector(seed=0).fit(X[is_train])      # full unlabelled training period
s_if = if_model.score(X[is_test])
print(f"IsolationForest: PR-AUC {average_precision_score(y, s_if):.3f}")""")
md(r"""### 5.2 Autoencoder (the detector) — full implementation

MLP `48→32→16→8→16→32→48` (ReLU), trained **only on normal training blocks** to reconstruct their
log-transformed, standardised count vectors; per-block reconstruction MSE = anomaly score.
Safeguards: scaler fitted on *normal training data only* (no leakage); early stopping on a held-out
slice of the normal training set. (~3–4 min on CPU.)""")
code(r"""import torch
from torch import nn

class AE(nn.Module):
    def __init__(self, d=48, dims=(32,16,8)):
        super().__init__()
        enc=[]; last=d
        for h in dims: enc += [nn.Linear(last,h), nn.ReLU()]; last=h
        self.encoder = nn.Sequential(*enc[:-1])                    # no ReLU on the code
        dec=[]; rdims=(*dims[::-1][1:], d)
        for h in rdims: dec += [nn.Linear(last,h), nn.ReLU()]; last=h
        self.decoder = nn.Sequential(*dec[:-1])                    # linear output
    def forward(self,x): return self.decoder(self.encoder(x))

class AEDetector:
    def __init__(self, epochs=25, lr=1e-3, batch=256, patience=5, seed=0):
        self.epochs, self.lr, self.batch, self.patience, self.seed = epochs, lr, batch, patience, seed
        self.history = []
    def _prep_fit(self, Xn):
        Z = np.log1p(Xn.astype(np.float32)); self.mu_, self.sd_ = Z.mean(0), Z.std(0)+1e-6
        return (Z-self.mu_)/self.sd_
    def _prep(self, Xs): return (np.log1p(Xs.astype(np.float32))-self.mu_)/self.sd_
    def fit(self, X_normal):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        Z = self._prep_fit(X_normal)
        idx = rng.permutation(len(Z)); n_val = max(1, len(Z)//10)
        Xtr, Xval = torch.from_numpy(Z[idx[n_val:]]), torch.from_numpy(Z[idx[:n_val]])
        self.net = AE(Z.shape[1]); opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(len(Xtr))
            for s in range(0, len(Xtr), self.batch):
                b = Xtr[perm[s:s+self.batch]]
                opt.zero_grad(); loss = ((self.net(b)-b)**2).mean(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                vt = float(((self.net(Xtr)-Xtr)**2).mean()); vv = float(((self.net(Xval)-Xval)**2).mean())
            self.history.append((vt, vv))
            if vv < best-1e-6: best, best_state, bad = vv, {k:v.clone() for k,v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience: break
        self.net.load_state_dict(best_state); return self
    def score(self, Xs):
        Z = torch.from_numpy(self._prep(Xs)); self.net.eval()
        with torch.no_grad(): return ((self.net(Z)-Z)**2).mean(1).numpy()

rng = np.random.default_rng(0)
sub = rng.choice(np.flatnonzero(normal_train), size=120_000, replace=False)   # subsample for notebook runtime
ae = AEDetector(epochs=25, seed=0).fit(X[sub])
s_ae = ae.score(X[is_test])
print(f"Autoencoder: PR-AUC {average_precision_score(y, s_ae):.3f}")
h = np.array(ae.history)
fig, ax = plt.subplots(figsize=(6,3))
ax.plot(h[:,0], label="train MSE"); ax.plot(h[:,1], label="val MSE"); ax.legend()
ax.set_xlabel("epoch"); ax.set_title("Autoencoder training (normal blocks only)"); plt.show()""")
md(r"""### 5.3 Why it works — anatomy of a reconstruction
One normal vs one anomalous test block: input count vector vs the network's reconstruction. The
anomaly activates templates the model never learned to encode ⇒ large residual ⇒ high score.""")
code(r"""test_idx = np.flatnonzero(is_test)
i_norm = test_idx[np.flatnonzero(y==0)[0]]
i_anom = test_idx[np.flatnonzero(y==1)[np.argmax(s_ae[np.flatnonzero(y==1)])]]
fig, axes = plt.subplots(1,2, figsize=(13,3.5), sharey=True)
for ax, i, ttl in [(axes[0], i_norm, "normal block"), (axes[1], i_anom, "anomalous block")]:
    z = ae._prep(X[i:i+1])
    with torch.no_grad(): rec = ae.net(torch.from_numpy(z)).numpy()
    ax.bar(np.arange(48)-0.2, z[0], width=.4, label="input", color="#4C72B0")
    ax.bar(np.arange(48)+0.2, rec[0], width=.4, label="reconstruction", color="#C44E52")
    ax.set_title(f"{ttl} — MSE {float(((rec-z)**2).mean()):.3f}"); ax.set_xlabel("template idx"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")
md(r"""### 5.4 Score distributions, head-to-head PR curves, and false alarms/day""")
code(r"""fig, axes = plt.subplots(1,2, figsize=(13,4.2))
ls = np.log10(np.maximum(s_ae,1e-12))
axes[0].hist(ls[y==0], bins=80, alpha=.7, label="normal", color="#4C72B0", log=True)
axes[0].hist(ls[y==1], bins=80, alpha=.7, label="anomaly", color="#C44E52", log=True)
axes[0].set_title("AE score distributions (log10) — note the atom of identical normal scores")
axes[0].set_xlabel("log10 reconstruction MSE"); axes[0].legend()
for name, s, c in [("Isolation Forest", s_if, "#8172B3"), ("Autoencoder", s_ae, "#C44E52")]:
    p, r, _ = precision_recall_curve(y, s)
    axes[1].plot(r, p, c, lw=2, label=f"{name} (PR-AUC={average_precision_score(y,s):.3f})")
axes[1].axhline(y.mean(), ls="--", c="gray", lw=1, label=f"chance ({y.mean():.3f})")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].legend(loc="lower left")
axes[1].set_title("Precision-recall, temporal test split"); plt.tight_layout(); plt.show()

vals, counts = np.unique(s_ae[y==0], return_counts=True)
print(f"largest normal-score atom: {counts.max():,}/{(y==0).sum():,} = {counts.max()/(y==0).sum():.1%} share ONE value (breaks EVT later)")
for name, s in [("Isolation Forest", s_if), ("Autoencoder", s_ae)]:
    fa, op = false_alarms_per_day(y, s, ts)
    print(f"{name:17s}: precision@90% {op['precision']:.3f} | false alarms/day {fa:,.0f}")""")
md(r"""### 5.5 Drift decomposition
Same threshold, four consecutive slices of the ~5 h test window: the baseline's *ranking itself*
swings; the AE ranks stably — but even its **false-alarm rate at a fixed threshold varies by orders
of magnitude** ⇒ ranking is solved, *alerting* is not (fixed in §8).""")
code(r"""print("Isolation Forest:"); display(drift_report(y, s_if, ts).round(3))
print("Autoencoder:");      display(drift_report(y, s_ae, ts).round(3))""")

# ---------------------------------------------------------------- Hawkes
md(r"""## 6. The mathematical core — Hawkes process, implemented from scratch

**Model.** Conditional intensity (instantaneous event rate given the past):
$$\lambda(t)=\mu+\sum_{t_i<t}\alpha e^{-\beta(t-t_i)}$$
Each event lifts the rate by $\alpha$, decaying with timescale $1/\beta$. **Branching ratio**
$\eta=\alpha/\beta$ = mean direct offspring per event: $\eta<1$ stable, $\eta\ge1$ explosive.

**Estimation.** Maximise $\ \ell=\sum_i\log\lambda(t_i)-\int_0^T\lambda\;$ — the first term rewards
intensity on real events, the second (expected count) punishes indiscriminate intensity. The integral
is closed-form; the sum uses **Ogata's O(n) recursion** $A_i=e^{-\beta\Delta t_i}(1+A_{i-1})$.
Everything below — likelihood, fit, simulation (Ogata thinning), compensator, time-rescaling
validation — is implemented in this cell.""")
code(r"""def recursion_A(t, beta):
    A = np.empty_like(t); A[0] = 0.0
    for i in range(1, len(t)):
        A[i] = np.exp(-beta*(t[i]-t[i-1])) * (1.0 + A[i-1])
    return A

def hawkes_nll(params, t, T):
    mu, alpha, beta = np.exp(params)                    # log-space -> positivity for free
    lam = mu + alpha*recursion_A(t, beta)
    integral = mu*T + (alpha/beta)*np.sum(1 - np.exp(-beta*(T - t)))
    return -(np.sum(np.log(lam)) - integral)

def hawkes_fit(events, T=None):
    t = np.sort(np.asarray(events, float)); t = t - t[0]
    T = float(t[-1]) if T is None else T
    gaps = np.diff(t); med = np.median(gaps[gaps>0]) if np.any(gaps>0) else 1.0
    x0 = np.log([0.5*len(t)/T, 0.5/med, 1.0/med])       # data-driven start, eta0=0.5
    res = minimize(hawkes_nll, x0, args=(t, T), method="L-BFGS-B", options=dict(maxiter=500))
    mu, alpha, beta = np.exp(res.x)
    return dict(mu=mu, alpha=alpha, beta=beta, eta=alpha/beta, ll=-res.fun, T=T, n=len(t))

def hawkes_simulate(mu, alpha, beta, T, seed=None):     # Ogata thinning (exact)
    rng = np.random.default_rng(seed); ev=[]; t=0.0; ex=0.0
    while True:
        M = mu + ex
        t_new = t + rng.exponential(1.0/M)
        if t_new > T: break
        ex *= np.exp(-beta*(t_new - t))
        if rng.uniform() <= (mu+ex)/M: ev.append(t_new); ex += alpha
        t = t_new
    return np.asarray(ev)

def compensator(events, mu, alpha, beta):               # Lambda(t_i), O(n)
    t = np.sort(np.asarray(events, float)); t = t - t[0]
    A = recursion_A(t, beta)
    return mu*t + (alpha/beta)*(np.arange(len(t)) - A)

def rescaled_gaps(events, f): return np.diff(compensator(events, f["mu"], f["alpha"], f["beta"]))

def intensity_grid(events, grid, mu, alpha, beta):      # O(n+m) sweep for plotting
    t = np.sort(np.asarray(events, float)); out = np.full(len(grid), mu); ex=0.0; last=min(grid[0], t[0]); i=0
    for k, g in enumerate(grid):
        while i < len(t) and t[i] < g:
            ex = ex*np.exp(-beta*(t[i]-last)) + 1.0; last = t[i]; i += 1
        out[k] = mu + alpha*ex*np.exp(-beta*(g-last))
    return out
print("Hawkes machinery defined")""")
md(r"""### 6.1 Correctness before trust — parameter recovery from simulation
A fitted number is worthless unless the fitter is proven: simulate with **known** $(\mu,\alpha,\beta)$,
refit, verify recovery — and verify the time-rescaling test *accepts* the true model.
**Time-rescaling theorem:** under a correct model, $\Delta\tau_i=\Lambda(t_i)-\Lambda(t_{i-1})$ are
i.i.d. Exp(1) ⇒ model checking = QQ-plot vs Exp(1) + KS statistic; mean gap must be 1 exactly.""")
code(r"""truth = dict(mu=0.8, alpha=1.2, beta=2.0)
sim = hawkes_simulate(**truth, T=20_000, seed=1)
f_sim = hawkes_fit(sim)
print(f"truth:     mu=0.80 alpha=1.20 beta=2.00 eta=0.60")
print(f"recovered: mu={f_sim['mu']:.2f} alpha={f_sim['alpha']:.2f} beta={f_sim['beta']:.2f} eta={f_sim['eta']:.3f}   ({f_sim['n']:,} simulated events)")
gsim = rescaled_gaps(sim, f_sim)
ks = stats.kstest(gsim, "expon", args=(0,1))
print(f"time-rescaling on true model: KS={ks.statistic:.4f}, mean gap={gsim.mean():.4f}  (accepted)")
wrong = dict(mu=len(sim)/20_000, alpha=1e-9, beta=1.0)   # Poisson-like wrong model
gw = np.diff(compensator(sim, **wrong))
print(f"same data, WRONG (memoryless) model: KS={stats.kstest(gw,'expon',args=(0,1)).statistic:.3f}  (rejected)")""")
md(r"""### 6.2 Fits on real data — BGL
BGL carries **microsecond** timestamps (verified: ~0% ties) — the well-posed home for continuous
time. Two contrasting 24 h windows: a calm day and the 2005-06-11 fault cascade. (~1–2 min.)""")
code(r"""bgl = pd.read_csv(INTERIM/"BGL.log_parsed.csv", usecols=["timestamp","label"], parse_dates=["timestamp"])
bgl = bgl.sort_values("timestamp").reset_index(drop=True)
d = bgl.timestamp.diff().dropna().dt.total_seconds()
print(f"BGL: {len(bgl):,} events over {(bgl.timestamp.max()-bgl.timestamp.min()).days} days | {bgl.label.mean():.2%} alert lines | ties {(d==0).mean():.4%}")

def window_seconds(day, hours=24):
    t0 = pd.Timestamp(day); m = (bgl.timestamp>=t0)&(bgl.timestamp<t0+pd.Timedelta(hours=hours))
    tt = bgl.loc[m,"timestamp"]; return (tt-tt.iloc[0]).dt.total_seconds().to_numpy()

fits = {}
for day in ["2005-09-03", "2005-06-11"]:
    tw = window_seconds(day); f = hawkes_fit(tw); g = rescaled_gaps(tw, f)
    fits[day] = (tw, f, g)
    ksv = stats.kstest(g, "expon", args=(0,1)).statistic
    print(f"{day}: {f['n']:>7,} events | eta={f['eta']:.4f} | KS={ksv:.3f} | mean gap={g.mean():.4f}")""")
code(r"""tw, f, g = fits["2005-09-03"]
grid = np.linspace(0, tw[-1], 3000)
fig = plt.figure(figsize=(13,7))
ax0 = fig.add_subplot(2,1,1)
ax0.plot(grid, intensity_grid(tw, grid, f["mu"], f["alpha"], f["beta"]), lw=.7, color="#C44E52")
ax0.set_title(f"Fitted conditional intensity λ(t) — 2005-09-03 (η={f['eta']:.3f})"); ax0.set_ylabel("events/s")
for j, day in enumerate(["2005-09-03","2005-06-11"]):
    _, fd, gd = fits[day]
    gs = np.sort(gd); probs = (np.arange(1,len(gs)+1)-.5)/len(gs)
    theo = stats.expon.ppf(probs)
    axq = fig.add_subplot(2,2,3+j)
    axq.scatter(theo, gs, s=3, alpha=.3, color="#4C72B0"); hi=float(max(theo[-1],gs[-1]))
    axq.plot([0,hi],[0,hi],"k--",lw=1); axq.set_title(f"QQ vs Exp(1) — {day} (η={fd['eta']:.3f})")
    axq.set_xlabel("theoretical"); axq.set_ylabel("empirical")
plt.tight_layout(); plt.show()""")
md(r"""**Headline finding — near-critical self-excitation.** η ≈ **0.97–1.0** on both windows: each BGL
event triggers ≈1 direct offspring; the stream operates at the edge of self-sustaining cascades
(mean cascade size $1/(1-\eta)$ → huge). Validated: mean rescaled gap = 1.0000 (compensator correct),
QQ body on the diagonal (bulk fits).

**Honest caveat.** The **upper QQ tail bends upward**: the deepest quiet gaps exceed what one
exponential timescale allows (after a burst the kernel over-predicts continued activity). Mild on
the calm day (KS 0.095), severe in the cascade (KS 0.43). One timescale can't fit both tight bursts
and deep lulls ⇒ motivates a sum-of-exponentials kernel (future work). At n~10⁵ the KS *p*-value
rejects any parametric model — the QQ geometry and KS magnitude carry the meaning.""")

# ---------------------------------------------------------------- ablation
md(r"""## 7. Does timing help detection? — ablation with inline window features

Fixed 60-s BGL windows; three feature sets; same detector (our `IFDetector`), same temporal split:
* **content** — counts of the top-50 templates (vocabulary from the train period only);
* **timing** — Hawkes excitation $A(t)=\sum e^{-\beta\Delta t}$ per event, summarised per window
  (mean/max) at two decay scales (1 s, 30 s) + event count — computed from *times only* (leakage-free);
* **content+timing**. (~3–4 min: one O(n) excitation pass over 4.7M events per scale.)""")
code(r"""ev3 = pd.read_csv(INTERIM/"BGL.log_parsed.csv", usecols=["timestamp","event_id","label"],
                  parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
secs = (ev3.timestamp - ev3.timestamp.iloc[0]).dt.total_seconds().to_numpy()
ev3["win"] = (secs // 60).astype(np.int64)

vocab = ev3.iloc[:int(.7*len(ev3))]["event_id"].value_counts().head(50).index.tolist()
content = pd.crosstab(ev3[ev3.event_id.isin(vocab)].win, ev3[ev3.event_id.isin(vocab)].event_id)\
            .reindex(columns=vocab, fill_value=0)

timing = pd.DataFrame({"win": ev3["win"]})
for beta in (1.0, 1/30.0):
    a = recursion_A(secs, beta)
    timing[f"exc_mean_{beta:.3g}"] = a; timing[f"exc_max_{beta:.3g}"] = a
agg = {c: ("mean" if "mean" in c else "max") for c in timing.columns if c!="win"}
timing = timing.groupby("win").agg(agg); timing["n_events"] = ev3.groupby("win").size()

wmeta = ev3.groupby("win").agg(start=("timestamp","min"), label=("label","max"))
idx = wmeta.sort_values("start").index
Xc = content.reindex(idx, fill_value=0).to_numpy(np.float32)
Xt = timing.reindex(idx).fillna(0).to_numpy(np.float32)
yw = wmeta.loc[idx,"label"].to_numpy().astype(int); tsw = wmeta.loc[idx,"start"]
n_tr = int(.7*len(idx)); w_tr = np.arange(len(idx))<n_tr; w_te = ~w_tr; w_norm = w_tr & (yw==0)
print(f"{len(idx):,} windows | {yw.mean():.2%} anomalous | train {w_tr.sum():,} / test {w_te.sum():,}")

resl = {}
for name, F in [("content",Xc), ("timing",Xt), ("content+timing",np.hstack([Xc,Xt]))]:
    s = IFDetector(seed=0).fit(F[w_norm]).score(F[w_te])
    fa, op = false_alarms_per_day(yw[w_te], s, tsw[w_te])
    resl[name] = dict(pr_auc=round(average_precision_score(yw[w_te],s),3),
                      precision_at_90rec=round(op["precision"],3), fa_per_day=round(fa,1))
del ev3
pd.DataFrame(resl).T""")
md(r"""**Two-sided honest reading.** Absolute detection is *weak* (~0.09 PR-AUC vs 0.047 chance — a 60 s
window dilutes a few alert lines in heavy traffic; the task is hard). **Relatively, timing wins**: at
90% recall it **halves the false alarms** of content and ~doubles precision — the Hawkes-derived
signal carries operational information that counts do not. Reported as measured, without overselling.""")

# ---------------------------------------------------------------- EVT
md(r"""## 8. Adaptive thresholding — EVT implemented, demonstrated, and honestly bounded

**Problem (from §5.5):** at a fixed threshold the AE's false alarms swing by orders of magnitude
across five hours. **EVT/POT:** exceedances over a high threshold follow a Generalized Pareto law
(Pickands–Balkema–de Haan); fit its tail, invert for the level $z_q$ crossed with probability $q$,
re-estimate on a trailing window ⇒ self-calibrating threshold. All implemented here (method-of-
moments GPD for O(1) updates), plus the distribution-free rolling quantile.""")
code(r"""def gpd_moments(exc):
    m, v = float(np.mean(exc)), float(np.var(exc))
    if not np.isfinite(v) or v<=0 or m<=0: return 0.0, max(m,1e-12)
    g = 0.5*(1 - m*m/v); s = m*(1-g)
    return (0.0, max(m,1e-12)) if s<=0 else (g, s)

def pot_threshold(scores, q, init_level=0.95):
    t = float(np.quantile(scores, init_level)); exc = scores[scores>t]-t
    g, s = gpd_moments(exc); r = q*len(scores)/max(len(exc),1)
    z = t + s*np.log(1/r) if abs(g)<1e-8 else t + (s/g)*(r**(-g)-1)
    return z

class AdaptivePot:                          # trailing-window POT (SPOT-style)
    def __init__(self, q, window=20_000, stride=500, init_level=0.95):
        self.q,self.w,self.st,self.il = q,window,stride,init_level
    def fit(self, calib):
        self.buf = list(np.asarray(calib,float)[-self.w:]); self.z = pot_threshold(np.array(self.buf), self.q, self.il); return self
    def run(self, scores):
        fl, th, since = [], [], 0
        for x in np.asarray(scores,float):
            fl.append(x > self.z); self.buf.append(x); del self.buf[:-self.w]; since += 1
            if since >= self.st: self.z = pot_threshold(np.array(self.buf), self.q, self.il); since = 0
            th.append(self.z)
        return np.array(fl), np.array(th)

class RollingQuantile:                      # distribution-free fallback
    def __init__(self, q, window=20_000, stride=500): self.q,self.w,self.st = q,window,stride
    def fit(self, calib):
        self.buf = list(np.asarray(calib,float)[-self.w:]); self.z = float(np.quantile(self.buf, 1-self.q)); return self
    def run(self, scores):
        fl, th, since = [], [], 0
        for x in np.asarray(scores,float):
            fl.append(x > self.z); self.buf.append(x); del self.buf[:-self.w]; since += 1
            if since >= self.st: self.z = float(np.quantile(self.buf, 1-self.q)); since = 0
            th.append(self.z)
        return np.array(fl), np.array(th)
print("thresholding machinery defined")""")
md(r"""### 8.1 EVT verified on continuous data — then its documented failure on HDFS
(a) POT holds a 1% target rate on stationary continuous scores; (b) tracks drift where a fixed
threshold explodes. (c) **But** the HDFS scores are *discrete* — §5.4 measured ~59% of normal blocks
sharing ONE value — violating GPD's continuity hypothesis, so on HDFS we use the rolling quantile.""")
code(r"""rng = np.random.default_rng(0)
calib_c = rng.exponential(1.0, 200_000); fresh = rng.exponential(1.0, 200_000)
z = pot_threshold(calib_c, q=0.01)
print(f"(a) stationary: target 1.00% -> empirical exceedance {(fresh>z).mean():.2%}")
n=120_000; drifting = rng.exponential(np.linspace(1,3,n))
fl,_ = AdaptivePot(q=0.01).fit(calib_c).run(drifting)
print(f"(b) drift (scale 1->3), last third: fixed flags {(drifting[-40_000:]>z).mean():.1%} | adaptive EVT flags {fl[-40_000:].mean():.1%} (target 1%)")""")
code(r"""order_t = np.argsort(ts.values, kind="stable")
ls_t = np.log(np.maximum(s_ae[order_t], 1e-12)); y_t = y[order_t]; tt = pd.to_datetime(ts.values[order_t])
calib_n, q = 20_000, 0.03
z_fix = float(np.quantile(ls_t[:calib_n], 1-q))
fl_fix = ls_t[calib_n:] > z_fix
fl_ad,_ = RollingQuantile(q=q).fit(ls_t[:calib_n]).run(ls_t[calib_n:])
y_r, t_r = y_t[calib_n:], tt[calib_n:]

def fa_curve(flags, nb=30):
    edges = pd.date_range(t_r.min(), t_r.max(), periods=nb+1); mid, fa = [], []
    for i in range(nb):
        m = np.asarray((t_r>=edges[i]) & ((t_r<edges[i+1]) if i<nb-1 else (t_r<=edges[i+1])))
        if m.sum()==0: continue
        mid.append(edges[i]); fa.append(int((flags[m]&(y_r[m]==0)).sum())/((edges[i+1]-edges[i]).total_seconds()/86400))
    return mid, fa

fig, ax = plt.subplots(figsize=(11,3.4))
for flg, cme, lab in [(fl_fix,"#8172B3","fixed threshold"), (fl_ad,"#C44E52","adaptive (rolling quantile)")]:
    mid, fa = fa_curve(flg); ax.plot(mid, fa, color=cme, marker="o", ms=3, lw=1.4, label=lab)
ax.set_ylabel("false alarms/day"); ax.legend(); ax.set_title(f"(c) HDFS: false-alarm stability under drift (q={q:.0%})"); plt.show()
for flg, lab in [(fl_fix,"fixed"),(fl_ad,"adaptive")]:
    print(f"{lab:9s}: flagged {flg.mean():.1%} | recall {(flg&(y_r==1)).sum()/max((y_r==1).sum(),1):.2f}")""")
md(r"""The adaptive threshold holds the alert budget through the drift; the fixed one explodes mid-period.
Explicit trade-off: holding the budget costs recall when anomalies exceed it — a *tunable* operating
point (the knob is q) instead of a hidden failure.""")

# ---------------------------------------------------------------- triage
md(r"""## 9. Triage — deterministic reduction + explanation, implemented inline

Paging once per flag is unusable. Three steps: **deduplicate** (same entity + template signature ⇒
merge with count) → **correlate** (flags close in time = one incident — the operational translation
of near-critical self-excitation: clustered anomalies are cascade members) → **explain & rank**
(strict fields: priority, category, summary, action; rule-based backend here — an LLM backend plugs
into the same schema, one call per *incident*). (~2 min to pull template context for top-200 flags.)""")
code(r"""@dataclass
class Anomaly:
    id: str; timestamp: pd.Timestamp; score: float
    entity: str = ""; templates: tuple = (); context: list = field(default_factory=list)
    @property
    def signature(self): return (self.entity, tuple(sorted(self.templates)))

def dedup(anoms):
    best, counts = {}, {}
    for a in anoms:
        k = a.signature; counts[k] = counts.get(k,0)+1
        if k not in best or a.score > best[k].score: best[k] = a
    return list(best.values()), counts

def correlate(anoms, gap_s=30.0):
    ordered = sorted(anoms, key=lambda a: a.timestamp); groups=[[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        (groups.append([cur]) if (cur.timestamp-prev.timestamp).total_seconds()>gap_s else groups[-1].append(cur))
    return groups

KEYWORDS = {"data_loss":("delete","deleting","corrupt","missing"),
            "replication_failure":("replicat","addstoredblock","ask"),
            "network":("connection","socket","timeout","unreachable","reset"),
            "hardware_fault":("ecc","parity","dimm","temperature"),
            "kernel_fault":("kernel","panic","exception","fatal")}
def category_of(templates):
    blob = " ".join(templates).lower()
    return next((c for c,kws in KEYWORDS.items() if any(k in blob for k in kws)), "unknown")

ACTIONS = {"data_loss":"Verify replicas; halt further deletions on affected entities.",
           "replication_failure":"Check DataNode health; trigger re-replication.",
           "network":"Inspect connectivity between involved nodes.",
           "hardware_fault":"Schedule hardware diagnostic.",
           "kernel_fault":"Capture kernel logs; consider draining the node.",
           "unknown":"Review surrounding context; correlate with recent changes."}

def triage(anoms, gap_s=30.0):
    deduped, counts = dedup(anoms); out=[]
    for i, grp in enumerate(correlate(deduped, gap_s)):
        tpls=[t for a in grp for t in a.templates]; cat=category_of(tpls)
        smax=max(a.score for a in grp); n=len(grp)
        prio = "P1" if smax>=.8 or n>=20 else "P2" if smax>=.5 or n>=5 else "P3" if smax>=.2 else "P4"
        out.append(dict(id=f"INC-{i+1:04d}", priority=prio, category=cat,
            n_anomalies=n, dedup_count=sum(counts.get(a.signature,1) for a in grp),
            max_score=round(smax,3), start=str(min(a.timestamp for a in grp)),
            end=str(max(a.timestamp for a in grp)),
            summary=f"{n} correlated anomalies; dominant category '{cat}'.",
            recommended_action=ACTIONS[cat]))
    return sorted(out, key=lambda r: ({"P1":0,"P2":1,"P3":2,"P4":3}[r["priority"]], -r["max_score"]))
print("triage engine defined")""")
code(r"""flag_meta = meta[meta.split=="test"].copy(); flag_meta["score"] = s_ae
top = flag_meta.sort_values("score", ascending=False).head(200).copy()
top["sev"] = top["score"].rank(pct=True)          # unbounded MSE -> [0,1] severity by rank

ctx = pd.read_csv(INTERIM/"HDFS.log_parsed.csv", usecols=["template","block_ids"], keep_default_na=False)
wanted = set(top.index); tpl = {b: [] for b in wanted}
for t, blocks in zip(ctx["template"], ctx["block_ids"].str.split()):
    for b in blocks:
        if b in wanted and t not in tpl[b]: tpl[b].append(t)
del ctx

anoms = [Anomaly(id=b, timestamp=r.start, score=float(r.sev), entity=b, templates=tuple(tpl.get(b,[])[:8]))
         for b, r in top.iterrows()]
incidents = triage(anoms)
inc_df = pd.DataFrame(incidents)
print(f"200 raw flags -> {len(inc_df)} ranked incidents")
fig, axes = plt.subplots(1,2, figsize=(10,3))
inc_df.priority.value_counts().sort_index().plot.bar(ax=axes[0], color="#4C72B0"); axes[0].set_title("by priority")
inc_df.category.value_counts().plot.bar(ax=axes[1], color="#55A868"); axes[1].set_title("by category")
plt.tight_layout(); plt.show()
inc_df.head(8)""")
code(r"""print(json.dumps(incidents[0], indent=2))""")

# ---------------------------------------------------------------- conclusions
md(r"""## 10. Consolidated results & conclusions

| Item | Result |
|---|---|
| HDFS parsing | 11.2M lines → 48 templates (invariant-checked: 575,061 = label count) |
| Features & split | 575k blocks × 48 templates; temporal 70/30, never shuffled |
| Detection (test) | AE PR-AUC ≈ **0.987** vs IF ≈ 0.75; **~20× fewer** false alarms @ 90% recall |
| Hawkes (BGL) | **η ≈ 0.97–1.0** near-critical, rescaling-validated; exp-kernel underfits deepest lulls |
| Timing ablation | timing features **halve false alarms** vs content on a hard task (~0.09 PR-AUC abs.) |
| Thresholding | fixed threshold FA/day explodes under drift → adaptive holds the budget |
| Triage | 200 flags → ~90 ranked, categorised, explained incidents |

**Method through-line:** every estimator proven on synthetic data before real data; strictly
out-of-time evaluation; the operational metric (false alarms/day) as headline; drift measured, not
averaged away; negative findings (HDFS timestamp ties, EVT-on-discrete-scores, kernel tail misfit)
reported as first-class results. All of it implemented in this notebook.""")

nb.cells = C
nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb, "notebooks/full_pipeline.ipynb")
print("notebook written:", len(C), "cells")
