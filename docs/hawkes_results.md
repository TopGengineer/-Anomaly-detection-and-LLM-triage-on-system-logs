# Hawkes process results — BGL

The mathematical centerpiece: a univariate exponential-kernel Hawkes process fit
to BGL event timestamps by maximum likelihood, validated with the time-rescaling
theorem. Implemented directly (`src/logtriage/hawkes/`, scipy) because `tick`
has no wheels for modern Python. Reproduce with `scripts/fit_hawkes.py`.

## Implementation is verified before use

The fitter is proven on **simulated** data (Ogata thinning with known
parameters) before touching BGL — see `tests/test_hawkes.py`:

- recovers known (μ, α, β) from a simulated process within ~15%;
- time-rescaling **passes** a correct fit (KS p > 0.01, mean rescaled gap ≈ 1)
  and **rejects** a deliberately wrong model;
- compensator is monotone; the mean rescaled gap on real fits is 1.0000, an
  independent check that Λ(·) is computed correctly.

## Fits on real BGL windows

| Window | Events | μ | β (1/β decay) | **branching ratio η = α/β** | KS stat | mean gap |
|---|---|---|---|---|---|---|
| 2005-09-03 (24 h, moderate) | 6,858 | 2.4e-3 | 7.49 (0.13 s) | **0.971** | 0.095 | 1.0001 |
| 2005-06-11 (24 h, fault cascade) | 152,669 | 5.4e-4 | 0.034 (29.5 s) | **0.9998** | 0.431 | 1.0000 |

### Reading the results

**Branching ratio is near-critical (η ≈ 0.97–1.0) in every window.** Each event
triggers on the order of one direct offspring — BGL's event stream is strongly,
almost self-sustaining, self-exciting. This is the quantitative confirmation of
the clustering the EDA measured (Fano ≫ 1): not just "bursty", but bursty with a
specific, estimated cascade strength.

**Time-rescaling exposes where the single-exponential kernel breaks.** The QQ
plots (`reports/figures/bgl_hawkes_*_qq.png`) show the body of the rescaled-gap
distribution on the Exp(1) diagonal, but the **upper tail deviates upward**: the
longest quiet gaps are larger than a single exponential predicts. Interpretation:
right after a burst the fitted excitation (decaying at rate β) predicts many more
events; when silence follows instead, the integrated intensity over that gap is
huge, producing the outlying rescaled gaps. A single exponential cannot hold both
tight bursts and deep lulls. The effect is mild on the moderate window
(KS 0.095) and severe on the extreme cascade (KS 0.43, η pinned at ~1).

> On the KS p-values: at 10^4–10^5 events the KS test has enormous power and
> rejects almost any parametric model, so the *magnitude* (KS statistic, QQ
> shape, mean gap) is the honest read, not the p-value. Mean rescaled gap =
> 1.0000 confirms the fit is correctly normalised even where the shape deviates.

## What this motivates

- A richer kernel — a **sum of exponentials** (fast burst + slow tail) or a
  power-law kernel — to capture the multi-scale structure the QQ tails reveal.
- A **non-stationary background** μ(t) for very long / multi-regime windows.
- As a detector feature: λ(t) is computed from the (label-free) event stream, so
  feeding it to the anomaly detector is leakage-free. High λ(t) = "a cascade is
  in progress" — see the intensity plots.
