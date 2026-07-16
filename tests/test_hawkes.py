"""Correctness tests for the Hawkes core: parameter recovery + GOF calibration.

The centerpiece is only trustworthy if the fitter recovers known parameters
from simulated data and the time-rescaling test behaves correctly (passes a
true fit, rejects a wrong one).
"""

import numpy as np

from logtriage.hawkes.exp_kernel import compensator, fit, neg_log_likelihood, simulate
from logtriage.hawkes.validation import ks_test_exp1, rescaled_gaps


def test_simulate_recovers_parameters():
    # true params, subcritical (eta = alpha/beta = 0.6)
    mu, alpha, beta = 0.8, 1.2, 2.0
    events = simulate(mu, alpha, beta, T=20000.0, seed=1)
    assert len(events) > 5000, "simulation produced too few events to fit"

    f = fit(events)
    assert f.converged
    # recover within ~15% — MLE on a finite sample won't be exact
    assert abs(f.mu - mu) / mu < 0.15
    assert abs(f.alpha - alpha) / alpha < 0.20
    assert abs(f.beta - beta) / beta < 0.20
    assert abs(f.branching_ratio - alpha / beta) < 0.08


def test_branching_ratio_property():
    events = simulate(0.5, 0.5, 1.0, T=5000.0, seed=2)
    f = fit(events)
    assert abs(f.branching_ratio - f.alpha / f.beta) < 1e-12


def test_time_rescaling_passes_on_true_fit():
    mu, alpha, beta = 1.0, 0.8, 1.6  # eta = 0.5
    events = simulate(mu, alpha, beta, T=20000.0, seed=3)
    f = fit(events)
    gaps = rescaled_gaps(events, f)
    gof = ks_test_exp1(gaps)
    # a correct fit should not be rejected, and rescaled gaps average ~1
    assert abs(gof.mean_gap - 1.0) < 0.1
    assert gof.ks_pvalue > 0.01


def test_time_rescaling_rejects_wrong_model():
    # simulate a strongly self-exciting process, then evaluate GOF with a
    # WRONG (Poisson-like, no excitation) parameter set -> should be rejected
    events = simulate(0.5, 1.5, 2.0, T=20000.0, seed=4)
    from logtriage.hawkes.exp_kernel import HawkesFit

    wrong = HawkesFit(mu=len(events) / 20000.0, alpha=1e-6, beta=1.0,
                      loglik=0.0, n_events=len(events), T=20000.0, converged=True)
    gof = ks_test_exp1(rescaled_gaps(events, wrong))
    assert gof.ks_pvalue < 0.01, "wrong model should be rejected by time-rescaling"


def test_compensator_is_increasing():
    events = simulate(1.0, 0.5, 1.0, T=2000.0, seed=5)
    Lam = compensator(events, 1.0, 0.5, 1.0)
    assert np.all(np.diff(Lam) > 0)  # integrated intensity strictly increases


def test_nll_finite_and_beats_bad_params():
    events = simulate(1.0, 0.8, 1.6, T=3000.0, seed=6)
    t = np.sort(events) - events.min()
    good = neg_log_likelihood(np.log([1.0, 0.8, 1.6]), t, t[-1])
    bad = neg_log_likelihood(np.log([0.01, 0.01, 0.01]), t, t[-1])
    assert np.isfinite(good) and good < bad
