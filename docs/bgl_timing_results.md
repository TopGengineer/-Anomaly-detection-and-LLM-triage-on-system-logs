# Does the Hawkes timing signal help detection? — the BGL ablation

The question the whole BGL half exists to answer, tested honestly. On fixed
60-second windows we detect alert-containing windows three ways with the same
model (Isolation Forest) and the same strict temporal split, changing only the
features. Reproduce with `scripts/bgl_experiment.py`.

- **content** — counts of the top-50 event templates (bag of events, no timing).
- **timing** — per-window Hawkes self-excitation A(t)=Σ e^{−β(t−tᵢ)} at two decay
  scales (1 s, 30 s) plus event count. Computed from event *times only*.
- **content+timing** — both.

## Result (test split, 8,298 windows, 4.71% anomalous)

| feature set | PR-AUC | precision @ 90% recall | false alarms / day |
|---|---|---|---|
| content | 0.081 | 0.046 | 114.3 |
| **timing** | **0.087** | **0.083** | **60.9** |
| content+timing | 0.090 | 0.071 | 71.6 |
| chance | 0.047 | — | — |

## Honest reading

**Absolute detection is weak — this is a hard task.** All three sit around
PR-AUC 0.09, only ~2× the 0.047 chance line. At 60-second windows most alert
windows fold a handful of alert lines into heavy normal traffic, so the
unsupervised signal is faint. We do not oversell this: window-level BGL detection
with these features is not a strong detector.

**But the timing signal is the more useful one, and it cuts false alarms.** The
PR curves (`reports/figures/bgl_timing_ablation.png`) show content collapsing to
the chance line at recall > 0.5, while timing holds ~2× chance precision across
the operational range. At 90% recall, timing roughly **halves** the false-alarm
rate versus content (114 → 61 per day) and nearly doubles precision (0.046 →
0.083). This is exactly the axis the project cares about — fewer false alarms at
a fixed detection rate — and the Hawkes self-excitation feature delivers it.

**So does Hawkes "help"?** Nuanced yes: the self-excitation cascade is real (the
branching ratio is near-critical, `docs/hawkes_results.md`) and, as a feature, it
is the single most useful signal here and reduces false alarms — but it does not
turn window-level BGL detection into a strong detector on its own. The honest
headline is *"timing beats content and halves false alarms, on a task where
absolute detection is hard"*, not *"Hawkes solves BGL detection."*

## Caveats / what would sharpen this

- Isolation Forest has run-to-run randomness; the ~0.006 PR-AUC gap between
  content and content+timing is within the noise, so the robust claim rests on
  the operating-point (false-alarm) improvement, not the PR-AUC decimal.
- Window size (60 s), the top-50 vocabulary, and a single-exponential
  excitation are first choices, not tuned. A richer timing feature (the
  time-rescaling surprise under a fitted model) and a sensitivity sweep over
  window size are the natural next steps.
