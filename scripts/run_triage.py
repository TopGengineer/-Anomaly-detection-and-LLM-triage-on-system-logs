"""Triage the autoencoder's flagged HDFS blocks into ranked incidents.

Takes the top-scoring test blocks, attaches their event templates and a little
surrounding log context, then runs the triage agent (dedup -> correlate ->
explain -> rank) and prints structured JSON incidents.

    python scripts/run_triage.py --top 200

Uses the rule-based explainer by default; pass --llm to use Claude
(claude-opus-4-8) when a credential is configured (falls back automatically).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from logtriage.config import DATA_INTERIM, DATA_PROCESSED
from logtriage.triage.agent import TriageAgent
from logtriage.triage.incident import Anomaly
from logtriage.triage.llm import AnthropicExplainer, RuleBasedExplainer


def build_anomalies(top: int) -> list[Anomaly]:
    scores = np.load(DATA_PROCESSED / "ae_scores_test.npy")
    meta = pd.read_csv(DATA_PROCESSED / "hdfs_meta.csv", index_col=0,
                       parse_dates=["start", "end"])
    test = meta[meta["split"] == "test"].copy()
    test["score"] = scores
    flagged = test.sort_values("score", ascending=False).head(top).copy()
    # Raw reconstruction MSE is unbounded; map to a [0,1] severity by rank
    # percentile within the flagged set so triage priorities differentiate.
    flagged["severity"] = flagged["score"].rank(pct=True)

    # pull each block's event templates (+ a few raw lines) from parsed events
    events = pd.read_csv(DATA_INTERIM / "HDFS.log_parsed.csv",
                         usecols=["event_id", "template", "block_ids", "content"],
                         keep_default_na=False)
    events["block_ids"] = events["block_ids"].astype(str)
    wanted = set(flagged.index)
    # map block_id -> (templates, sample lines)  (single pass, only wanted blocks)
    tmpl: dict[str, list[str]] = {b: [] for b in wanted}
    ctx: dict[str, list[str]] = {b: [] for b in wanted}
    for tmpl_s, content, blocks in zip(events["template"], events["content"],
                                       events["block_ids"].str.split()):
        for b in blocks:
            if b in wanted:
                if tmpl_s not in tmpl[b]:
                    tmpl[b].append(tmpl_s)
                if len(ctx[b]) < 4:
                    ctx[b].append(content)

    anomalies = []
    for bid, row in flagged.iterrows():
        anomalies.append(Anomaly(
            id=bid, timestamp=row["start"], score=float(row["severity"]),
            entity=bid, templates=tuple(tmpl.get(bid, [])[:8]),
            context=ctx.get(bid, []),
        ))
    return anomalies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=200, help="top-scored blocks to triage")
    ap.add_argument("--gap", type=float, default=30.0, help="cascade gap seconds")
    ap.add_argument("--llm", action="store_true", help="use Claude when available")
    ap.add_argument("--out", type=Path, default=DATA_PROCESSED / "incidents.json")
    args = ap.parse_args()

    anomalies = build_anomalies(args.top)
    explainer = AnthropicExplainer() if args.llm else RuleBasedExplainer()
    backend = type(explainer).__name__
    if args.llm and getattr(explainer, "_client", None) is None:
        backend += " (no credential -> rule-based fallback)"

    agent = TriageAgent(explainer=explainer, gap_seconds=args.gap)
    incidents = agent.run(anomalies)

    print(f"triaged {len(anomalies)} flagged blocks -> {len(incidents)} incidents "
          f"| explainer: {backend}\n")
    by_prio = pd.Series([i.priority for i in incidents]).value_counts().to_dict()
    print("incidents by priority:", by_prio, "\n")
    for inc in incidents[:5]:
        print(f"[{inc.priority}] {inc.id}  ({inc.category}, score {inc.max_score})")
        print(f"     {inc.summary}")
        print(f"     action: {inc.recommended_action}")
        print(f"     {inc.n_anomalies} anomalies, {inc.dedup_count} raw flags, "
              f"{inc.start} -> {inc.end}\n")

    records = agent.to_records(incidents)
    args.out.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} incidents to {args.out}")


if __name__ == "__main__":
    main()
