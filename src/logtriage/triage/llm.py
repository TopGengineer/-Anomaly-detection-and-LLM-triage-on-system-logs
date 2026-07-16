"""Incident explainers: turn a correlated Incident into structured triage output.

Two interchangeable backends behind one `explain(incident)` protocol:

  * `RuleBasedExplainer` — deterministic, dependency-free. Derives priority,
    category, summary, and recommended action from the incident's templates,
    scores, and size. Always available, so the pipeline runs self-contained
    (and it is the reference the LLM output is checked against in tests).
  * `AnthropicExplainer` — calls Claude (`claude-opus-4-8`) with a JSON-schema
    structured output, so the model returns exactly the fields we need. Used
    when an API key / credential is configured; falls back to the rule-based
    explainer on any error so a missing key never breaks the demo.

Both emit the same dict shape:
    {priority, category, summary, recommended_action, explanation}
"""

from __future__ import annotations

import json
from typing import Protocol

from logtriage.triage.incident import Incident

PRIORITIES = ("P1", "P2", "P3", "P4")
CATEGORIES = (
    "data_loss", "replication_failure", "network", "resource_exhaustion",
    "hardware_fault", "kernel_fault", "unknown",
)

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": list(PRIORITIES)},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "summary": {"type": "string"},
        "recommended_action": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["priority", "category", "summary", "recommended_action", "explanation"],
    "additionalProperties": False,
}


class Explainer(Protocol):
    def explain(self, incident: Incident) -> dict: ...


# ---- keyword → category heuristics (shared by both backends) ----
_KEYWORDS = {
    "data_loss": ("delete", "deleting", "lost", "corrupt", "missing"),
    "replication_failure": ("replicat", "under-replicat", "addstoredblock", "ask"),
    "network": ("connection", "socket", "timeout", "unreachable", "reset"),
    "resource_exhaustion": ("memory", "disk", "full", "quota", "capacity"),
    "hardware_fault": ("ecc", "parity", "dimm", "temperature", "fan", "card"),
    "kernel_fault": ("kernel", "panic", "trap", "exception", "fatal"),
}


def guess_category(templates: list[str]) -> str:
    blob = " ".join(templates).lower()
    for cat, kws in _KEYWORDS.items():
        if any(kw in blob for kw in kws):
            return cat
    return "unknown"


class RuleBasedExplainer:
    """Deterministic triage — no external calls."""

    def explain(self, incident: Incident) -> dict:
        cat = guess_category(incident.templates)
        n = len(incident.anomalies)
        span = (incident.end - incident.start).total_seconds()
        # priority from score + cascade size
        if incident.max_score >= 0.8 or n >= 20:
            prio = "P1"
        elif incident.max_score >= 0.5 or n >= 5:
            prio = "P2"
        elif incident.max_score >= 0.2:
            prio = "P3"
        else:
            prio = "P4"
        ents = incident.entities
        summary = (
            f"{n} correlated anomal{'y' if n == 1 else 'ies'} over {span:.0f}s"
            f"{f' on {ents[0]}' if len(ents) == 1 else f' across {len(ents)} entities' if ents else ''}"
            f"; dominant category '{cat}'."
        )
        actions = {
            "data_loss": "Verify block replicas and halt further deletions on the affected entities.",
            "replication_failure": "Check DataNode health and trigger re-replication of under-replicated blocks.",
            "network": "Inspect connectivity between the involved nodes; check for packet loss/timeouts.",
            "resource_exhaustion": "Check disk/memory headroom on the affected nodes; free or add capacity.",
            "hardware_fault": "Schedule a hardware diagnostic on the affected node(s).",
            "kernel_fault": "Capture kernel logs and consider draining/rebooting the affected node(s).",
            "unknown": "Review the surrounding log context and correlate with recent changes.",
        }
        return {
            "priority": prio,
            "category": cat,
            "summary": summary,
            "recommended_action": actions[cat],
            "explanation": (
                f"Grouped {incident.dedup_count} raw flag(s) into one incident spanning "
                f"{incident.start} → {incident.end}. Peak anomaly score "
                f"{incident.max_score:.3f}. Templates involved: "
                f"{', '.join(incident.templates[:5])}"
                f"{'…' if len(incident.templates) > 5 else ''}."
            ),
        }


class AnthropicExplainer:
    """Claude-backed triage with structured JSON output; falls back on error."""

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._fallback = RuleBasedExplainer()
        self._client = None
        try:  # construct lazily; a missing key/SDK just means we fall back
            import anthropic

            self._client = anthropic.Anthropic()
        except Exception:
            self._client = None

    def _prompt(self, incident: Incident) -> str:
        lines = [
            "You are a security-operations triage assistant for distributed-system logs.",
            "Classify the incident below and return the structured fields.",
            "",
            f"Incident {incident.id}: {len(incident.anomalies)} correlated anomalies",
            f"time span: {incident.start} -> {incident.end}",
            f"peak anomaly score: {incident.max_score:.3f}",
            f"entities: {', '.join(incident.entities) or 'n/a'}",
            f"event templates: {', '.join(incident.templates)}",
            "",
            "surrounding log context (sampled):",
        ]
        for a in incident.anomalies[:3]:
            lines += [f"  - {c}" for c in a.context[:4]]
        return "\n".join(lines)

    def explain(self, incident: Incident) -> dict:
        if self._client is None:
            return self._fallback.explain(incident)
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"format": {"type": "json_schema", "schema": TRIAGE_SCHEMA}},
                messages=[{"role": "user", "content": self._prompt(incident)}],
            )
            text = next(b.text for b in resp.content if b.type == "text")
            data = json.loads(text)
            # guard the schema even though structured output enforces it
            if not set(TRIAGE_SCHEMA["required"]).issubset(data):
                raise ValueError("missing fields")
            return data
        except Exception:
            return self._fallback.explain(incident)
