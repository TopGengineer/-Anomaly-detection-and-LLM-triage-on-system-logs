import pandas as pd

from logtriage.triage.agent import TriageAgent
from logtriage.triage.incident import Anomaly, build_incidents, correlate, dedup
from logtriage.triage.llm import RuleBasedExplainer, guess_category


def _a(i, t, score=0.5, entity="", templates=("Deleting block <*>",)):
    return Anomaly(id=f"a{i}", timestamp=pd.Timestamp(t), score=score,
                   entity=entity, templates=templates)


def test_dedup_collapses_identical_signatures():
    anoms = [
        _a(1, "2008-11-11 08:00:00", 0.4, "blk_1"),
        _a(2, "2008-11-11 08:00:01", 0.9, "blk_1"),  # same sig, higher score
        _a(3, "2008-11-11 08:00:02", 0.5, "blk_2"),
    ]
    out = dedup(anoms)
    assert len(out) == 2                       # blk_1 collapsed
    assert max(a.score for a in out) == 0.9    # kept the higher-scoring rep


def test_correlate_groups_by_time_gap():
    anoms = [
        _a(1, "2008-11-11 08:00:00"),
        _a(2, "2008-11-11 08:00:10"),   # within 30s -> same incident
        _a(3, "2008-11-11 08:05:00"),   # 5min later -> new incident
    ]
    incidents = correlate(anoms, gap_seconds=30)
    assert len(incidents) == 2
    assert len(incidents[0].anomalies) == 2


def test_category_from_templates():
    assert guess_category(["Deleting block <*>"]) == "data_loss"
    assert guess_category(["kernel INFO instruction cache parity error"]) in (
        "hardware_fault", "kernel_fault")
    assert guess_category(["something totally generic"]) == "unknown"


def test_rulebased_priority_scales_with_score_and_size():
    small = build_incidents([_a(1, "2008-11-11 08:00:00", 0.1)])[0]
    big = build_incidents([_a(i, f"2008-11-11 08:00:{i:02d}", 0.9) for i in range(25)])[0]
    ex = RuleBasedExplainer()
    assert ex.explain(small)["priority"] == "P4"
    assert ex.explain(big)["priority"] == "P1"


def test_agent_ranks_p1_first_and_emits_records():
    anoms = (
        [_a(i, f"2008-11-11 08:00:{i:02d}", 0.9, "blk_hot") for i in range(25)]  # -> P1
        + [_a(100, "2008-11-11 09:00:00", 0.1, "blk_cold")]                       # -> P4
    )
    agent = TriageAgent(gap_seconds=30)
    triaged = agent.run(anoms)
    assert triaged[0].priority == "P1"          # highest priority first
    records = agent.to_records(triaged)
    assert set(records[0]) >= {"priority", "category", "summary",
                               "recommended_action", "explanation"}
