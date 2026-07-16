# EDA findings — HDFS (full) and BGL timestamp resolution

Empirical results from the parsed full HDFS_v1 log (11,175,629 lines) and the
BGL sample. These drive the architecture decisions below; every number here is
reproducible with `scripts/explore_hdfs.py` and the checks noted.

## HDFS full dataset

| Property | Value |
|---|---|
| Log lines | 11,175,629 |
| Drain templates | 48 |
| Time span | 2008-11-09 20:35:18 → 2008-11-11 11:16:28 (~39 h) |
| Blocks | 575,061 |
| Anomalous blocks | 16,838 (**2.93%**) |
| Median events / block | 19 (max 298) |
| Nulls / unsorted timestamps | none |
| Exact duplicate rows | 490,388 (genuine repeated log emissions) |

### The timestamp-resolution problem (decisive for Hawkes)

HDFS timestamps have **1-second resolution**, and the stream averages ~80
events/second. Consequences, measured:

- **98.8%** of consecutive events in the global stream share a timestamp (tie).
- Per block, **90.5%** of consecutive events still tie; a median block occupies
  only **~2 distinct seconds** (distinct-second ratio 0.105).
- **Every** block's full lifetime is ≤ 54 s (median 7 s); no block spans > 60 s.

A continuous-time Hawkes process requires a strict ordering of event times. At
this resolution and density that ordering is absent — **neither the global
stream nor per-block sequences support a continuous-time Hawkes fit** on HDFS.
Adding sub-second jitter would fabricate the very structure the process is meant
to measure (the branching ratio would become an artifact of the jitter), so it
is rejected.

## BGL timestamp resolution

BGL log lines carry **microsecond** timestamps and events are naturally spread
over seconds to minutes:

```
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 ... KERNEL INFO ...
- 1117838573 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.53.276129 ... KERNEL INFO ...
```

This is genuine continuous time, suitable for continuous-time Hawkes. BGL is
labeled **per line** (a leading `-` marks a normal line; anything else is an
alert category), so the Hawkes intensity λ(t) and the labels to validate it as a
detector feature coexist at the event level.

## Architecture decisions driven by these findings

1. **HDFS = detection + triage showcase.** Detection unit is the **block
   session** (events grouped by `block_id` → 48-template count vector), because
   HDFS labels are per block and blocks are short-lived. Temporal split is by
   block start time.
2. **BGL = Hawkes centerpiece.** Continuous-time exponential-kernel Hawkes on
   BGL event timestamps: branching ratio, time-rescaling goodness-of-fit, and
   λ(t) fed to the detector as a feature — validated against BGL's per-line
   labels.
3. **Duplicates** are kept in the canonical parsed data and handled per stage
   (repetition can itself be signal for detection; specific methods dedup as
   needed).
