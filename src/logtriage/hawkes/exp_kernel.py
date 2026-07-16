"""Univariate Hawkes process with an exponential kernel — the mathematical core.

Conditional intensity:

    λ(t) = μ + Σ_{t_i < t} α · exp(−β (t − t_i)),     μ, α, β > 0

Each event raises the intensity by α and the excitation decays at rate β. The
expected number of direct offspring per event (the **branching ratio**) is

    η = ∫₀^∞ α e^{−β s} ds = α / β,

and the process is stationary/subcritical iff η < 1.

The log-likelihood on an observation window [0, T] is

    ℓ = Σ_i log λ(t_i) − ∫₀^T λ(s) ds,

with the integral available in closed form and Σ_i log λ(t_i) computable in
O(n) via Ogata's recursion

    A_i = e^{−β (t_i − t_{i−1})} (1 + A_{i−1}),   A_0 = 0,   λ(t_i) = μ + α A_i.

Fit by maximum likelihood (`fit`); validate by the time-rescaling theorem
(`compensator` here, tests in `validation.py`); simulate by Ogata thinning
(`simulate`, used to prove the fitter recovers known parameters).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class HawkesFit:
    mu: float
    alpha: float
    beta: float
    loglik: float
    n_events: int
    T: float
    converged: bool

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    def __str__(self) -> str:
        return (
            f"Hawkes(exp) fit on {self.n_events:,} events over T={self.T:.1f}\n"
            f"  mu    = {self.mu:.6g}   (background rate)\n"
            f"  alpha = {self.alpha:.6g}\n"
            f"  beta  = {self.beta:.6g}   (1/beta = {1/self.beta:.4g} decay time)\n"
            f"  branching ratio eta = alpha/beta = {self.branching_ratio:.4f}"
            f"   ({'subcritical' if self.branching_ratio < 1 else 'CRITICAL/explosive'})\n"
            f"  log-likelihood = {self.loglik:.2f}   converged={self.converged}"
        )


def _recursion_A(t: np.ndarray, beta: float) -> np.ndarray:
    """A_i = e^{-beta (t_i - t_{i-1})} (1 + A_{i-1}), A_0 = 0. O(n)."""
    A = np.empty_like(t)
    A[0] = 0.0
    for i in range(1, len(t)):
        A[i] = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + A[i - 1])
    return A


def neg_log_likelihood(params: np.ndarray, t: np.ndarray, T: float) -> float:
    """Negative log-likelihood. `params` = (log mu, log alpha, log beta).

    Optimised in log-space so the positivity constraints are automatic and the
    optimiser sees an unconstrained problem.
    """
    mu, alpha, beta = np.exp(params)
    A = _recursion_A(t, beta)
    lam = mu + alpha * A
    if np.any(lam <= 0):
        return np.inf
    sum_log = np.sum(np.log(lam))
    integral = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    ll = sum_log - integral
    return -ll


def fit(events: np.ndarray, T: float | None = None) -> HawkesFit:
    """Maximum-likelihood fit of (μ, α, β) to an event-time sequence.

    `events` need not start at 0; they are shifted so the window is [0, T].
    """
    t = np.sort(np.asarray(events, dtype=float))
    t = t - t[0]
    if T is None:
        T = float(t[-1])
    n = len(t)

    # Data-driven starting point: background rate ~ half the mean rate, decay
    # ~ inverse median gap, branching ratio ~ 0.5.
    gaps = np.diff(t)
    med_gap = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 1.0
    beta0 = 1.0 / med_gap
    mu0 = 0.5 * n / T
    alpha0 = 0.5 * beta0  # eta = 0.5
    x0 = np.log([mu0, alpha0, beta0])

    res = minimize(
        neg_log_likelihood, x0, args=(t, T), method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    mu, alpha, beta = np.exp(res.x)
    return HawkesFit(
        mu=mu, alpha=alpha, beta=beta, loglik=-res.fun,
        n_events=n, T=T, converged=bool(res.success),
    )


def compensator(events: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Integrated intensity Λ(t_i) at each event — the time-rescaling transform.

    Λ(t) = μ t + (α/β) Σ_{t_i < t} (1 − e^{−β (t − t_i)}). If the model is
    correct, the increments Λ(t_i) − Λ(t_{i−1}) are i.i.d. Exp(1).
    """
    t = np.sort(np.asarray(events, dtype=float))
    t = t - t[0]
    # cumulative sum with an O(n) recursion for Σ_{t_j < t_i}(1 - e^{-β(t_i-t_j)})
    A = _recursion_A(t, beta)  # A_i = Σ_{j<i} e^{-β(t_i - t_j)}
    # number of prior events at each i is just i (0-indexed)
    counts = np.arange(len(t))
    excite = (alpha / beta) * (counts - A)
    return mu * t + excite


def simulate(
    mu: float, alpha: float, beta: float, T: float, seed: int | None = None
) -> np.ndarray:
    """Simulate the process on [0, T] via Ogata's thinning algorithm.

    Used to validate the fitter: simulate with known (μ, α, β), then check that
    `fit` recovers them.
    """
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0
    lam_excite = 0.0  # Σ α e^{-β(t - t_i)} carried forward
    while True:
        M = mu + lam_excite  # upper bound on intensity just after t
        w = rng.exponential(1.0 / M)
        t_new = t + w
        if t_new > T:
            break
        # decay excitation to t_new
        lam_excite *= np.exp(-beta * (t_new - t))
        lam = mu + lam_excite
        if rng.uniform() <= lam / M:  # accept
            events.append(t_new)
            lam_excite += alpha  # the new event's jump
        t = t_new
    return np.asarray(events)
