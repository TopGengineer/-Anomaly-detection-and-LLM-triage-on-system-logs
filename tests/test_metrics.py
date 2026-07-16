import numpy as np
import pandas as pd

from logtriage.eval.metrics import (
    false_alarms_per_day,
    operating_point_at_recall,
    pr_auc,
)


def test_operating_point_perfect_separation():
    # two positives are the two highest scores -> full recall, zero false alarms
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 1, 0, 0, 0])
    op = operating_point_at_recall(labels, scores, target_recall=1.0)
    assert op.recall == 1.0
    assert op.fp == 0
    assert op.tp == 2
    assert op.precision == 1.0


def test_operating_point_requires_a_false_positive():
    # a normal block outranks a true anomaly -> catching both costs 1 false alarm
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 0, 1, 0, 0])
    op = operating_point_at_recall(labels, scores, target_recall=1.0)
    assert op.recall == 1.0
    assert op.tp == 2
    assert op.fp == 1
    assert abs(op.precision - 2 / 3) < 1e-9


def test_partial_recall_target():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 1, 0, 0, 0])
    op = operating_point_at_recall(labels, scores, target_recall=0.5)
    assert op.recall == 0.5
    assert op.tp == 1
    assert op.fp == 0


def test_false_alarms_per_day_uses_real_span():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 0, 1, 0, 0])  # 1 FP needed for full recall
    # span exactly 2 days -> 1 FP / 2 days = 0.5 per day
    ts = pd.to_datetime(
        [
            "2008-11-09 00:00:00",
            "2008-11-09 06:00:00",
            "2008-11-10 06:00:00",
            "2008-11-10 12:00:00",
            "2008-11-11 00:00:00",
        ]
    )
    fa, op = false_alarms_per_day(labels, scores, pd.Series(ts), target_recall=1.0)
    assert op.fp == 1
    assert abs(fa - 0.5) < 1e-9


def test_pr_auc_perfect_and_ranking():
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.9, 0.8])
    assert pr_auc(labels, perfect) == 1.0
    # a worse ranking scores lower
    worse = np.array([0.9, 0.8, 0.2, 0.1])
    assert pr_auc(labels, worse) < 1.0


def test_no_positives_raises():
    try:
        operating_point_at_recall(np.zeros(4), np.arange(4.0), 0.9)
        assert False, "expected ValueError"
    except ValueError:
        pass
