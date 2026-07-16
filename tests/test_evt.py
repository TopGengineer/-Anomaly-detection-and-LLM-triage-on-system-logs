import numpy as np

from logtriage.eval.evt import AdaptivePot, RollingQuantile, fit_gpd, pot_threshold


def test_pot_threshold_controls_exceedance_rate():
    # calibrate on one exponential sample, check the target rate on a fresh one
    rng = np.random.default_rng(0)
    calib = rng.exponential(1.0, size=200_000)
    q = 0.01
    pot = pot_threshold(calib, q=q, init_level=0.95)
    fresh = rng.exponential(1.0, size=200_000)
    emp_rate = (fresh > pot.z).mean()
    # should be within a factor of ~2 of the 1% target
    assert 0.005 < emp_rate < 0.02


def test_fit_gpd_recovers_exponential_tail():
    # gamma = 0 is an exponential tail; genpareto with c=0, scale=s
    rng = np.random.default_rng(1)
    exc = rng.exponential(2.0, size=50_000)
    gamma, sigma = fit_gpd(exc)
    assert abs(gamma) < 0.05
    assert abs(sigma - 2.0) < 0.15


def test_adaptive_holds_rate_on_stationary_stream():
    rng = np.random.default_rng(2)
    calib = rng.exponential(1.0, size=20_000)
    stream = rng.exponential(1.0, size=100_000)
    ad = AdaptivePot(q=0.01, window=20_000, stride=500, init_level=0.95).fit(calib)
    flags, _ = ad.run(stream)
    assert 0.005 < flags.mean() < 0.02  # flagged rate near target 1%


def test_adaptive_tracks_drift_where_fixed_fails():
    rng = np.random.default_rng(3)
    calib = rng.exponential(1.0, size=20_000)
    # stream drifts upward: scale ramps from 1 -> 3 across the stream
    n = 120_000
    scale = np.linspace(1.0, 3.0, n)
    stream = rng.exponential(scale)

    # fixed threshold from calibration (static POT, held constant)
    fixed = pot_threshold(calib, q=0.01, init_level=0.95).z
    fixed_rate_late = (stream[-40_000:] > fixed).mean()

    # adaptive trailing-window POT
    ad = AdaptivePot(q=0.01, window=20_000, stride=500, init_level=0.95).fit(calib)
    flags, _ = ad.run(stream)
    adapt_rate_late = flags[-40_000:].mean()

    # under drift the fixed threshold flags far too much; adaptive stays near target
    assert fixed_rate_late > 0.10           # fixed blows up
    assert adapt_rate_late < 0.03           # adaptive stays controlled
    assert adapt_rate_late < fixed_rate_late / 3


def test_rolling_quantile_stabilizes_discrete_scores_under_drift():
    # discrete scores (point masses) that drift upward — the HDFS-like case
    rng = np.random.default_rng(4)
    levels = np.array([-12.0, -8.0, -4.0, 0.0])
    calib = rng.choice(levels, size=20_000, p=[0.6, 0.25, 0.1, 0.05])
    # stream: mass shifts toward higher levels over time
    n = 90_000
    early = rng.choice(levels, size=n // 2, p=[0.6, 0.25, 0.1, 0.05])
    late = rng.choice(levels, size=n - n // 2, p=[0.1, 0.2, 0.3, 0.4])
    stream = np.concatenate([early, late])

    q = 0.05
    z_fixed = np.quantile(calib, 1 - q)
    fixed_rate_late = (stream[n // 2 :] > z_fixed).mean()

    rq = RollingQuantile(q=q, window=20_000, stride=500).fit(calib)
    flags, _ = rq.run(stream)
    adapt_rate_late = flags[n // 2 :].mean()

    assert fixed_rate_late > 0.20           # fixed flags a huge fraction after drift
    assert adapt_rate_late < 0.12           # adaptive stays near the q budget
    assert adapt_rate_late < fixed_rate_late / 2
