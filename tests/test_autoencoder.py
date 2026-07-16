import numpy as np

from logtriage.models.autoencoder import AutoEncoderDetector


def test_autoencoder_flags_out_of_distribution():
    rng = np.random.default_rng(0)
    # "normal": counts concentrated in the first few templates
    d = 12
    normal = np.zeros((400, d), dtype=np.float32)
    normal[:, :3] = rng.poisson(5, size=(400, 3))
    # "anomalies": mass in templates the model never saw active
    anom = np.zeros((40, d), dtype=np.float32)
    anom[:, 8:] = rng.poisson(5, size=(40, 4))

    model = AutoEncoderDetector(hidden=(8,), latent=3, epochs=40, seed=0).fit(normal)
    s_normal = model.score(normal)
    s_anom = model.score(anom)

    # anomalies should score clearly higher on average
    assert s_anom.mean() > s_normal.mean() * 3
    # scores are per-sample and finite
    assert s_normal.shape == (400,) and np.all(np.isfinite(s_normal))


def test_score_requires_fit():
    try:
        AutoEncoderDetector().score(np.zeros((2, 4), dtype=np.float32))
        assert False, "expected assertion before fit"
    except AssertionError:
        pass
