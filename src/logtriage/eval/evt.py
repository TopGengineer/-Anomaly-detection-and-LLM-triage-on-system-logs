"""Extreme Value Theory thresholding — Peaks-Over-Threshold (POT / SPOT).

The detectors produce anomaly scores; we must turn scores into alerts with a
threshold. A *fixed* threshold is fragile: when the score distribution drifts
(as it does on HDFS — see docs/detection_results.md), the fraction of blocks it
flags — and therefore the false-alarm rate — drifts with it.

EVT gives a principled, self-calibrating alternative. By the Pickands–Balkema–
de Haan theorem, exceedances of a high threshold t converge to a Generalized
Pareto Distribution (GPD):

    P(X − t > x | X > t) ≈ (1 + γ x / σ)^(−1/γ).

Fit the GPD to the observed exceedances, then invert it for the score level z_q
that is exceeded with a target probability q:

    z_q = t + (σ/γ) [ (q n / N_t)^(−γ) − 1 ]        (γ → 0: t + σ ln(N_t / (q n)))

Holding q fixed while re-estimating (t, γ, σ) from recent data keeps the flagged
rate — hence the false-alarm rate — stable as the distribution moves.
`pot_threshold` is the static estimator; `AdaptivePot` recalibrates it on a
trailing window so the threshold tracks drift online.
"""

from dataclasses import dataclass

import numpy as np


def _gpd_from_moments(mean: float, var: float) -> tuple[float, float]:
    """Method-of-moments GPD (loc=0), scipy `c` convention: mean = σ/(1−γ).

    γ = ½(1 − mean²/var),  σ = mean(1 − γ). Closed-form and O(1), so the
    streaming detector can refit on every peak. Falls back to an exponential
    tail (γ=0) when the variance is unusable.
    """
    if not np.isfinite(mean) or not np.isfinite(var) or var <= 0 or mean <= 0:
        return 0.0, max(mean, 1e-12)
    gamma = 0.5 * (1.0 - mean * mean / var)
    sigma = mean * (1.0 - gamma)
    if sigma <= 0:
        return 0.0, max(mean, 1e-12)
    return float(gamma), float(sigma)


def fit_gpd(excesses: np.ndarray) -> tuple[float, float]:
    """Fit GPD shape γ and scale σ to positive exceedances (loc=0), via moments."""
    excesses = np.asarray(excesses, dtype=float)
    if len(excesses) < 10:
        return 0.0, max(float(excesses.mean()) if len(excesses) else 1e-12, 1e-12)
    return _gpd_from_moments(float(excesses.mean()), float(excesses.var()))


def _zq(t: float, gamma: float, sigma: float, Nt: int, n: int, q: float) -> float:
    """Invert the GPD tail for the level exceeded with probability q."""
    r = q * n / Nt
    if r <= 0:
        return np.inf
    if abs(gamma) < 1e-8:
        return t + sigma * np.log(1.0 / r)  # = t + σ ln(Nt/(qn))
    return t + (sigma / gamma) * (r ** (-gamma) - 1.0)


@dataclass
class PotThreshold:
    z: float          # the alert threshold
    t: float          # initial high threshold
    gamma: float
    sigma: float
    n_peaks: int


def pot_threshold(scores: np.ndarray, q: float, init_level: float = 0.98) -> PotThreshold:
    """Static POT: fit the GPD tail of `scores` and return the level for rate q."""
    scores = np.asarray(scores, dtype=float)
    t = float(np.quantile(scores, init_level))
    excesses = scores[scores > t] - t
    gamma, sigma = fit_gpd(excesses)
    z = _zq(t, gamma, sigma, len(excesses), len(scores), q)
    return PotThreshold(z=z, t=t, gamma=gamma, sigma=sigma, n_peaks=len(excesses))


class AdaptivePot:
    """Trailing-window POT: recalibrate the EVT threshold from recent scores.

    A fixed-size window of the most recent scores is re-fit with `pot_threshold`
    every `stride` observations, so the alert level tracks a drifting score
    distribution while always holding the target tail probability q. Unlike
    cumulative streaming SPOT there is no compounding truncation bias — each
    refit sees the actual recent empirical tail.

    Tradeoff (documented, not hidden): a *sustained* burst that fills the window
    will raise the threshold and can suppress detection. That is the price of
    self-calibration; it is fine for drift, and the window length bounds how long
    any burst can influence the threshold.
    """

    def __init__(
        self, q: float = 1e-3, window: int = 20_000, stride: int = 500,
        init_level: float = 0.95,
    ):
        self.q, self.window, self.stride, self.init_level = q, window, stride, init_level

    def fit(self, calib: np.ndarray) -> "AdaptivePot":
        calib = np.asarray(calib, dtype=float)
        self._buf = list(calib[-self.window :])
        self.z = pot_threshold(np.asarray(self._buf), self.q, self.init_level).z
        return self

    def run(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stream a whole array; return (flags, threshold_at_each_step)."""
        scores = np.asarray(scores, dtype=float)
        flags = np.empty(len(scores), dtype=bool)
        thr = np.empty(len(scores), dtype=float)
        buf = self._buf
        since = 0
        for i, x in enumerate(scores):
            flags[i] = x > self.z
            buf.append(float(x))
            if len(buf) > self.window:
                del buf[: len(buf) - self.window]
            since += 1
            if since >= self.stride:
                self.z = pot_threshold(np.asarray(buf), self.q, self.init_level).z
                since = 0
            thr[i] = self.z
        return flags, thr


class RollingQuantile:
    """Distribution-free adaptive threshold: the (1−q) quantile of a trailing
    window, recomputed every `stride` observations.

    The robust counterpart to `AdaptivePot` for scores whose tail is NOT
    continuous — e.g. the HDFS autoencoder's reconstruction errors, where
    repetitive normal blocks collapse onto identical values (heavy point masses)
    and GPD extrapolation is unstable. It makes no distributional assumption, so
    it tracks drift without the EVT tail model.
    """

    def __init__(self, q: float = 0.02, window: int = 20_000, stride: int = 500):
        self.q, self.window, self.stride = q, window, stride

    def fit(self, calib: np.ndarray) -> "RollingQuantile":
        self._buf = list(np.asarray(calib, dtype=float)[-self.window :])
        self.z = float(np.quantile(self._buf, 1.0 - self.q))
        return self

    def run(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=float)
        flags = np.empty(len(scores), dtype=bool)
        thr = np.empty(len(scores), dtype=float)
        buf = self._buf
        since = 0
        for i, x in enumerate(scores):
            flags[i] = x > self.z
            buf.append(float(x))
            if len(buf) > self.window:
                del buf[: len(buf) - self.window]
            since += 1
            if since >= self.stride:
                self.z = float(np.quantile(buf, 1.0 - self.q))
                since = 0
            thr[i] = self.z
        return flags, thr
