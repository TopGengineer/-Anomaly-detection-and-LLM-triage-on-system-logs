"""Turn flagged anomalies into correlated incidents.

The detector emits many flagged units (HDFS blocks, BGL windows); paging an
analyst once per flag is unusable. Two reductions happen before any LLM call:

  * DEDUP — flags with the same signature (entity + event-template multiset)
    collapse into one, with a count. Repetition is recorded, not re-alerted.
  * CORRELATE — flags that belong to the same cascade become one incident. This
    is the Hawkes idea made operational: self-exciting events are causally
    linked, so anomalies close in time (and, when available, sharing an entity)
    are grouped rather than reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Anomaly:
    id: str
    timestamp: pd.Timestamp
    score: float
    entity: str = ""                 # block_id (HDFS) or node (BGL)
    templates: tuple[str, ...] = ()  # event-template signature
    context: list[str] = field(default_factory=list)  # surrounding raw log lines

    @property
    def signature(self) -> tuple[str, tuple[str, ...]]:
        return (self.entity, tuple(sorted(self.templates)))


@dataclass
class Incident:
    id: str
    anomalies: list[Anomaly]
    dedup_count: int = 1             # how many raw flags collapsed in

    @property
    def start(self) -> pd.Timestamp:
        return min(a.timestamp for a in self.anomalies)

    @property
    def end(self) -> pd.Timestamp:
        return max(a.timestamp for a in self.anomalies)

    @property
    def max_score(self) -> float:
        return max(a.score for a in self.anomalies)

    @property
    def entities(self) -> list[str]:
        return sorted({a.entity for a in self.anomalies if a.entity})

    @property
    def templates(self) -> list[str]:
        seen: dict[str, None] = {}
        for a in self.anomalies:
            for t in a.templates:
                seen.setdefault(t, None)
        return list(seen)


def dedup(anomalies: list[Anomaly]) -> list[Anomaly]:
    """Collapse identical-signature anomalies; keep the highest-scoring rep."""
    best: dict[tuple, Anomaly] = {}
    counts: dict[tuple, int] = {}
    for a in anomalies:
        k = a.signature
        counts[k] = counts.get(k, 0) + 1
        if k not in best or a.score > best[k].score:
            best[k] = a
    # stash the dedup count on the rep via a parallel dict returned by caller
    dedup.counts = counts  # type: ignore[attr-defined]
    return list(best.values())


def correlate(
    anomalies: list[Anomaly],
    gap_seconds: float = 30.0,
    same_entity: bool = False,
) -> list[Incident]:
    """Group anomalies into incidents by temporal proximity (cascade grouping).

    Sorted by time; a gap larger than `gap_seconds` starts a new incident. With
    `same_entity=True` a new entity also starts a new incident (useful when the
    entity is a stable key like a BGL node; HDFS blocks are per-anomaly so it is
    left off there).
    """
    if not anomalies:
        return []
    ordered = sorted(anomalies, key=lambda a: a.timestamp)
    incidents: list[list[Anomaly]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = (cur.timestamp - prev.timestamp).total_seconds()
        new_group = gap > gap_seconds or (same_entity and cur.entity != prev.entity)
        (incidents.append([cur]) if new_group else incidents[-1].append(cur))
    return [Incident(id=f"INC-{i+1:04d}", anomalies=grp)
            for i, grp in enumerate(incidents)]


def build_incidents(
    anomalies: list[Anomaly], gap_seconds: float = 30.0, same_entity: bool = False
) -> list[Incident]:
    """Full reduction: dedup then correlate. Incidents carry the dedup total."""
    deduped = dedup(anomalies)
    counts = getattr(dedup, "counts", {})
    incidents = correlate(deduped, gap_seconds=gap_seconds, same_entity=same_entity)
    for inc in incidents:
        inc.dedup_count = sum(counts.get(a.signature, 1) for a in inc.anomalies)
    return incidents
