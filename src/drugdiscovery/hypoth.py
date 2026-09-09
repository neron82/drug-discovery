"""Hypothesis engine — drafts mechanism hypotheses grounded ONLY in collected evidence.

Each hypothesis records: mechanism reasoning, predicted effect, supporting and counter
evidence (with statement IDs), an evidence profile from the evidence engine, testable
computational predictions, and a recommended next experiment. Clearly typed HYPOTHESIS.
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from . import repos
from .evidence import score_evidence
from .llm import LLMProvider
from .models import Hypothesis, KnowledgeType, Statement

log = logging.getLogger("dd.hypoth")

_SYSTEM = """You are the Hypotheses lead in an autonomous computational drug-discovery lab working on androgenetic alopecia.
Draft biologically plausible, testable mechanism hypotheses. You must ONLY reason from the supporting and
counter evidence supplied to you — do not introduce new citations or invented facts. Mark everything
clearly. Output STRICT JSON (no markdown):
{
 "title": "short hypothesis title",
 "body": "testable mechanistic hypothesis, 1-3 sentences",
 "mechanism_reasoning": "why this mechanism could matter, based on the given evidence",
 "predicted_effect": "what experimentally/computationally you predict would happen",
 "target": "canonical target name",
 "pathway": "pathway or empty",
 "supporting_evidence_summary": "one line",
 "counter_evidence_summary": "one line or 'none yet'",
 "uncertainty": "main uncertainties",
 "testable_predictions": ["prediction 1", "prediction 2"],
 "next_experiment": "the single highest-value next computational experiment",
 "confidence": 0.0
}
Confidence must be conservative (0-1)."""


def gather_candidate_targets(session, ws_id):
    """Return target nodes (receptor/enzyme/protein) that have >=1 supporting statement."""
    from .models import Node
    targets = session.execute(select(Node).where(
        Node.workspace_id == ws_id,
        Node.kind.in_(["receptor", "enzyme", "protein", "transcription_factor"]),
    )).scalars().all()
    scored = []
    for t in targets:
        sup, contra = statements_for_entity(session, ws_id, t)
        if not sup and not contra:
            continue
        profile = score_evidence(sup, contra)
        scored.append((t, sup, contra, profile))
    scored.sort(key=lambda x: x[3]["overall_confidence"], reverse=True)
    return scored[:8]


def statements_for_entity(session, ws_id, node):
    names = {node.canonical_name.lower(), node.label.lower()} | {s.lower() for s in (node.synonyms or [])}
    sup, contra = [], []
    for st in session.execute(select(Statement).where(Statement.workspace_id == ws_id)).scalars():
        a, b = (st.subject or "").lower(), (st.object or "").lower()
        if a in names or b in names or a and (a.split(" (")[0] in {x.split(" (")[0] for x in names}):
            (contra if st.properties.get("contradicted") else sup).append(st)
    return sup, contra


def generate_hypotheses(session, ws, llm: LLMProvider, limit: int = 3):
    made = []
    candidates = gather_candidate_targets(session, ws.id)
    if not candidates:
        log.info("No candidates for hypothesis generation (empty KG)")
        return made
    for t, sup, contra, profile in candidates[:limit]:
        try:
            h = _draft(session, ws, llm, t, sup, contra, profile)
            if h:
                made.append(h)
        except Exception as e:
            log.warning("hypothesis draft failed for %s: %s", t.canonical_name, e)
    if made:
        repos.add_timeline(session, ws.id, "director", "success",
                           f"Generated {len(made)} evidence-grounded hypotheses", {})
    return made


def _draft(session, ws, llm, target, sup, contra, profile) -> Hypothesis | None:
    sup_text = "\n".join(f"- {s.text}" for s in sup[:8]) or "(none)"
    contra_text = "\n".join(f"- {s.text}" for s in contra[:6]) or "(none yet)"
    user = (
        f"Target: {target.canonical_name} [{target.kind}]\n"
        f"Supporting evidence ({len(sup)} statements):\n{sup_text}\n\n"
        f"Contradicting evidence ({len(contra)}):\n{contra_text}\n\n"
        f"Evidence profile: overall_confidence={profile['overall_confidence']}, "
        f"independent_sources={profile['independent_sources']}, "
        f"contradiction_ratio={profile['contradiction_ratio']}\n\n"
        "Draft one hypothesis focused on this target relevant to androgenetic alopecia pathogenesis and therapy."
    )
    data = llm.chat_json(_SYSTEM, user, temperature=0.4)
    title = (data.get("title") or f"{target.canonical_name} mechanism hypothesis").strip()
    body = data.get("body", "")
    if not body:
        return None
    support_ids = [s.id for s in sup[:10]]
    contra_ids = [c.id for c in contra[:10]]
    h = repos.add_hypothesis(
        session, ws.id, title=title, body=body,
        mechanism_reasoning=data.get("mechanism_reasoning", ""),
        predicted_effect=data.get("predicted_effect", ""),
        supporting_evidence=[{"statement_id": sid} for sid in support_ids],
        counter_evidence=[{"statement_id": sid} for sid in contra_ids],
        uncertainty=data.get("uncertainty", ""),
        testable_predictions=data.get("testable_predictions", []),
        next_experiment=data.get("next_experiment", ""),
        confidence=float(data.get("confidence", 0.3)),
        target_id=target.id, evidence_profile=profile,
    )
    repos.add_timeline(session, ws.id, "director", "info",
                       f"Hypothesis drafted: {title[:80]}", data={"hypothesis_id": h.id})
    return h
