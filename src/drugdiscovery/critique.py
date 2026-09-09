"""Skeptic critique engine (Phase 5, Spec sections 15/30/33).

For each active hypothesis the skeptic:
 1. re-gathers supporting and contradicting statements from the knowledge graph,
 2. recomputes the multi-dimensional evidence profile,
 3. asks the local LLM (skeptic persona) to actively attempt falsification:
    alternative explanations, weaknesses, missing controls,
 4. adjusts confidence conservatively and transitions the hypothesis status,
 5. records an inspectable critique (reasoning summary + alternatives) and a
    timeline entry showing the confidence movement (e.g. 0.72 -> 0.55).

The skeptic is adversarial BY DESIGN: it optimises for disproving, not confirming.
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from . import repos
from .evidence import score_evidence
from .hypoth import statements_for_entity
from .models import Hypothesis, Node

log = logging.getLogger("dd.critique")

_SKEPTIC_SYSTEM = """You are the SKEPTIC in an autonomous computational drug-discovery lab.
Your job is to ATTACK the hypothesis below and try to disprove it. Consider:
- contradictory evidence supplied, and what its strength implies
- alternative mechanistic explanations for the same observations
- model/species relevance gaps (mouse != human scalp, cell culture != follicle)
- missing controls, irreproducibility risk, publication bias
- whether the claimed direction of effect is actually supported
Do NOT invent citations. Judge ONLY from the supplied evidence. Output STRICT JSON (no markdown):
{"verdict": "support|weaken|refute",
 "confidence_adjustment": -0.4 to +0.1 (negative means lower confidence),
 "reasoning_summary": "2-4 sentences, scientifically useful, no chain-of-thought dump",
 "alternative_explanations": ["alt 1", "alt 2"],
 "weaknesses": ["weakness 1", ...],
 "recommended_action": "keep_active|mark_speculative|reject|needs_more_evidence",
 "next_counter_experiment": "one concrete computational experiment that could falsify it"}"""


def critique_hypothesis(session, ws, h: Hypothesis, llm) -> dict:
    """Adversarial review of one hypothesis. Mutates h (confidence/status/critique)."""
    target = session.get(Node, h.target_id) if h.target_id else None
    if target is None:
        sup, contra = [], []
        for entry in (h.supporting_evidence or []):
            sid = entry.get("statement_id") if isinstance(entry, dict) else None
            from .models import Statement
            st = session.get(Statement, sid) if sid else None
            if st is not None:
                sup.append(st)
        for entry in (h.counter_evidence or []):
            sid = entry.get("statement_id") if isinstance(entry, dict) else None
            from .models import Statement
            st = session.get(Statement, sid) if sid else None
            if st is not None:
                contra.append(st)
    else:
        sup, contra = statements_for_entity(session, ws.id, target)
    profile = score_evidence(sup, contra)

    sup_text = "\n".join(f"- {s.text}" for s in sup[:8]) or "(none)"
    contra_text = "\n".join(f"- {s.text}" for s in contra[:8]) or "(none found)"
    user = (
        f"Hypothesis: {h.title}\n{h.body}\n\n"
        f"Mechanism reasoning on record: {h.mechanism_reasoning[:400] or '(none)'}\n\n"
        f"Supporting evidence ({len(sup)} statements):\n{sup_text}\n\n"
        f"Contradicting evidence ({len(contra)}):\n{contra_text}\n\n"
        f"Evidence profile: confidence={profile['overall_confidence']}, "
        f"contradiction_ratio={profile['contradiction_ratio']}, "
        f"independent_sources={profile['independent_sources']}\n\n"
        "Attempt to falsify this hypothesis."
    )
    try:
        data = llm.chat_json(_SKEPTIC_SYSTEM, user, temperature=0.3)
    except Exception as e:
        log.warning("skeptic LLM call failed for %s: %s", h.title[:40], e)
        return {"ok": False, "error": str(e)[:200]}

    adj = float(data.get("confidence_adjustment", -0.1))
    adj = max(-0.4, min(0.1, adj))
    before = round(h.confidence, 2)
    after = round(max(0.0, min(1.0, h.confidence + adj)), 2)
    h.confidence = after

    action = (data.get("recommended_action") or "keep_active").strip()
    if action == "reject" and after < 0.25:
        h.status = "rejected"
    elif action == "mark_speculative" or (after < 0.4 and h.status in ("active", "promising")):
        h.status = "speculative"
    elif after >= 0.7 and not contra:
        h.status = "promising"

    crit = {
        "verdict": data.get("verdict", "weaken"),
        "reasoning_summary": data.get("reasoning_summary", ""),
        "alternative_explanations": data.get("alternative_explanations", [])[:4],
        "weaknesses": data.get("weaknesses", [])[:6],
        "recommended_action": action,
        "next_counter_experiment": data.get("next_counter_experiment", ""),
        "confidence_before": before, "confidence_after": after,
        "evidence_profile": profile,
        "by": "skeptic", "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    props = dict(h.evidence_profile or {})
    props["last_critique"] = crit
    h.evidence_profile = props
    if crit["next_counter_experiment"]:
        h.next_experiment = crit["next_counter_experiment"]

    level = "warn" if crit["verdict"] in ("weaken", "refute") else "info"
    verdict = crit["verdict"]
    repos.add_timeline(
        session, ws.id, "skeptic", level,
        f"Skeptic critique of '{h.title[:60]}': verdict={verdict}, "
        f"confidence {before} -> {after}",
        data={"hypothesis_id": h.id, "action": action})
    return {"ok": True, "before": before, "after": after, "verdict": crit["verdict"],
            "action": action, "critique": crit}


def critique_active(session, ws, llm, limit: int = 3) -> list[dict]:
    """Critique the highest-confidence active/promising hypotheses."""
    hs = list(session.execute(
        select(Hypothesis).where(
            Hypothesis.workspace_id == ws.id,
            Hypothesis.status.in_(["active", "promising", "speculative"]))
        .order_by(Hypothesis.confidence.desc()).limit(limit)).scalars())
    out = []
    for h in hs:
        r = critique_hypothesis(session, ws, h, llm)
        if r.get("ok"):
            out.append({"hypothesis_id": h.id, "title": h.title[:70], **{k: r[k] for k in ("before", "after", "verdict", "action")}})
    if out:
        repos.add_timeline(session, ws.id, "skeptic", "info",
                           f"Skeptic pass complete: {len(out)} hypotheses reviewed", {})
    session.commit()
    return out
