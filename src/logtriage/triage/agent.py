"""The triage agent: flagged anomalies in, ranked explained incidents out.

Pipeline: dedup + correlate (incident.py) -> explain each incident (llm.py) ->
rank by priority then peak score -> emit structured records. One LLM call per
*incident*, not per raw flag, because dedup/correlation already collapsed the
volume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from logtriage.triage.incident import Anomaly, build_incidents
from logtriage.triage.llm import PRIORITIES, Explainer, RuleBasedExplainer


@dataclass
class TriagedIncident:
    id: str
    priority: str
    category: str
    summary: str
    recommended_action: str
    explanation: str
    n_anomalies: int
    dedup_count: int
    start: str
    end: str
    max_score: float
    entities: list[str]


class TriageAgent:
    def __init__(
        self,
        explainer: Explainer | None = None,
        gap_seconds: float = 30.0,
        same_entity: bool = False,
    ):
        self.explainer = explainer or RuleBasedExplainer()
        self.gap_seconds = gap_seconds
        self.same_entity = same_entity

    def run(self, anomalies: list[Anomaly]) -> list[TriagedIncident]:
        incidents = build_incidents(
            anomalies, gap_seconds=self.gap_seconds, same_entity=self.same_entity
        )
        out: list[TriagedIncident] = []
        for inc in incidents:
            fields = self.explainer.explain(inc)
            out.append(TriagedIncident(
                id=inc.id,
                priority=fields["priority"],
                category=fields["category"],
                summary=fields["summary"],
                recommended_action=fields["recommended_action"],
                explanation=fields["explanation"],
                n_anomalies=len(inc.anomalies),
                dedup_count=inc.dedup_count,
                start=str(inc.start),
                end=str(inc.end),
                max_score=round(inc.max_score, 4),
                entities=inc.entities[:10],
            ))
        # rank: priority (P1 first), then peak score
        prio_rank = {p: i for i, p in enumerate(PRIORITIES)}
        out.sort(key=lambda t: (prio_rank.get(t.priority, 9), -t.max_score))
        return out

    @staticmethod
    def to_records(incidents: list[TriagedIncident]) -> list[dict]:
        return [asdict(i) for i in incidents]
