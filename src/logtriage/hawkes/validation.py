"""Goodness-of-fit for a fitted Hawkes process via the time-rescaling theorem.

If {t_i} is a realisation of a point process with conditional intensity λ(·),
then the rescaled times τ_i = Λ(t_i) form a unit-rate Poisson process, so the
increments Δτ_i = Λ(t_i) − Λ(t_{i−1}) are i.i.d. Exp(1). We therefore fit the
Hawkes model, compute Δτ, and test it against Exp(1):

  * KS test against Exp(1) — a p-value and statistic;
  * a QQ plot of empirical vs theoretical Exp(1) quantiles (points here, plot
    in the script) — the honest visual that reveals *where* a fit breaks.

This is the model-independent check the whole project's rigor rests on: a good
branching ratio means nothing if the fit fails time-rescaling.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from logtriage.hawkes.exp_kernel import HawkesFit, compensator


def rescaled_gaps(events: np.ndarray, fit: HawkesFit) -> np.ndarray:
    """Δτ_i = Λ(t_i) − Λ(t_{i−1}); i.i.d. Exp(1) under a correct fit."""
    Lam = compensator(events, fit.mu, fit.alpha, fit.beta)
    return np.diff(Lam)


@dataclass
class GofResult:
    ks_stat: float
    ks_pvalue: float
    mean_gap: float   # should be ~1.0 for Exp(1)
    n: int

    def __str__(self) -> str:
        verdict = "consistent with Exp(1)" if self.ks_pvalue > 0.05 else "REJECTS Exp(1)"
        return (
            f"time-rescaling GOF on {self.n:,} rescaled gaps\n"
            f"  mean gap   = {self.mean_gap:.4f}   (Exp(1) -> 1.0)\n"
            f"  KS stat    = {self.ks_stat:.4f}\n"
            f"  KS p-value = {self.ks_pvalue:.4g}   -> {verdict}"
        )


def ks_test_exp1(gaps: np.ndarray) -> GofResult:
    """Kolmogorov–Smirnov test of the rescaled gaps against a unit exponential."""
    gaps = np.asarray(gaps, dtype=float)
    ks = stats.kstest(gaps, "expon", args=(0.0, 1.0))
    return GofResult(
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        mean_gap=float(gaps.mean()),
        n=len(gaps),
    )


def qq_points(gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(theoretical, empirical) Exp(1) quantiles for a QQ plot."""
    g = np.sort(np.asarray(gaps, dtype=float))
    n = len(g)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.expon.ppf(probs)  # Exp(1) quantiles
    return theoretical, g
