"""Autonomous research loop + experiment executor.

Loop: OBSERVE → PLUCK next prioritized experiment → EXECUTE → UPDATE KG/hypotheses →
PERSIST → SCHEDULE follow-ups → repeat while enabled.
Executes only typed, validated experiments through the dispatcher; no arbitrary shell.
"""
from __future__ import annotations

import logging
import time
import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from . import hypoth, repos
from .config import settings
from .db import get_session
from .extract import process_paper
from .lit.providers import get_provider
from .llm import LLMProvider
from .models import JobStatus, Node, Paper

log = logging.getLogger("dd.research")


def _mk_llm() -> LLMProvider:
    return LLMProvider()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
def execute_experiment(session, ws, ex, llm: LLMProvider):
    """Run one experiment, updating KG/timeline. Returns (summary, result_dict)."""
    kind = ex.kind
    params = ex.params_json or {}

    if kind == "literature_search":
        provider = get_provider(params.get("provider"))
        records = provider.search(params.get("query", ""), int(params.get("size", 10)))
        new = 0
        for rec in records:
            _, created = repos.upsert_paper(session, ws.id, rec)
            if created:
                new += 1
                # queue extraction for the fresh paper so flow continues
                repos.create_experiment(
                    session, ws.id, "paper_extraction",
                    question=f"Extract claims: {rec.title[:60]}…",
                    reason="Ingested paper needs structured extraction",
                    expected_information_gain=0.7, scientific_importance=0.6,
                    confidence_impact=0.4, computational_cost=1.2,
                    priority=0.5, params={"paper_id": rec.pmid and _paper_id_by_pmid(session, ws.id, rec.pmid)},
                    worker="biologist")
        repos.add_timeline(session, ws.id, "researcher", "info",
                           f"Literature search '{params.get('query')}': {new} new of {len(records)} papers ingested",
                           data={"query": params.get("query"), "new": new, "total": len(records)})
        return f"{new} new papers ingested", {"new": new, "seen": len(records)}

    if kind == "paper_extraction":
        pid = params.get("paper_id")
        paper = session.get(Paper, pid) if pid else None
        if paper is None:
            return "paper not found", {}
        process_paper(session, ws, paper, llm)
        n = len(paper.statements)
        got = f"{n} claims extracted" if paper.status == "extracted" else "extraction failed/incomplete"
        return got, {"n_claims": n}

    if kind == "hypothesis_generation":
        limit = int(params.get("limit", 3))
        made = hypoth.generate_hypotheses(session, ws, llm, limit=limit)
        return f"{len(made)} hypotheses generated", {"n": len(made)}

    if kind == "contradiction_search":
        provider = get_provider(params.get("provider"))
        topic = params.get("topic", "androgenetic alopecia")
        q = f"{topic} AND (negative OR \"no effect\" OR \"failed\" OR \"no association\" OR \"did not\")"
        records = provider.search(q, int(params.get("size", 8)))
        new = 0
        for rec in records:
            _, created = repos.upsert_paper(session, ws.id, rec)
            new += int(created)
            if created and rec.abstract:
                repos.create_experiment(session, ws.id, "paper_extraction",
                                        question=f"Extract claims: {rec.title[:60]}…",
                                        reason="Skeptic ingestion", expected_information_gain=0.7,
                                        scientific_importance=0.6, confidence_impact=0.6,
                                        computational_cost=1.2, priority=0.6,
                                        params={"paper_id": _paper_id_by_pmid(session, ws.id, rec.pmid)},
                                        worker="skeptic")
        repos.add_timeline(session, ws.id, "skeptic", "warn" if new else "info",
                           f"Skeptic search for negative evidence: {new} new papers", {})
        return f"Skeptic contradiction search: {new} new papers", {"new": new}

    if kind == "question_generation":
        questions = derive_open_questions(session, ws)
        state = dict(ws.state_json or {})
        state["open_questions"] = [{"q": q} for q in questions[:8]]
        ws.state_json = state
        repos.add_timeline(session, ws.id, "director", "info",
                           f"Proposed {len(questions)} open research questions", {"questions": questions[:8]})
        return f"{len(questions)} open questions defined", {"questions": questions[:8]}

    return f"Unhandled experiment kind: {kind}", {}


def _paper_id_by_pmid(session, ws_id, pmid):
    if not pmid:
        return None
    p = session.execute(select(Paper).where(Paper.workspace_id == ws_id, Paper.pmid == str(pmid))).scalar_one_or_none()
    return p.id if p else None


def derive_open_questions(session, ws) -> list[str]:
    ws_id = ws.id
    questions = []
    proteins = session.execute(select(Node).where(
        Node.workspace_id == ws_id, Node.kind.in_(["receptor", "enzyme", "protein"]))).scalars().all()
    for t in proteins:
        # no drug/compound link yet?
        has_drug = any(e.type in ("INHIBITS", "AGONIST_OF", "ANTAGONIST_OF", "BINDS_TO", "TARGETS")
                       for e in t.out_edges if e.target.kind in ("drug", "compound", "drug_class")) or \
                   any(e.source.kind in ("drug", "compound", "drug_class") for e in t.in_edges)
        if not has_drug:
            questions.append(f"Is {t.canonical_name} therapeutically targetable (known ligands/inhibitors) for {ws.disease or 'this disease'}?")
    questions.append(f"Which follicle-relevant cell type (dermal papilla, keratinocyte) mediates the strongest phenotype effect?")
    questions.append("Which mechanism, if modulated, most robustly extends anagen without driving unwanted androgen/off-target effects?")
    questions.append("Which reported associations fail replication or show negative results in independent cohorts?")
    questions.append("What are the major unresolved unknowns limiting the top-ranked hypothesis?")
    return list(dict.fromkeys(questions[:10]))


# ---------------------------------------------------------------------------
# Cycle + runner
# ---------------------------------------------------------------------------
def run_cycle(ws_id: str, llm: LLMProvider | None = None, force: bool = False) -> dict:
    """One research cycle. Returns a summary dict (or {"noop": true})."""
    from . import planner
    from .db import session_scope
    llm = llm or _mk_llm()
    with session_scope() as session:
        ws = repos.get_workspace(session, ws_id)
        if ws is None:
            return {"error": "workspace not found"}
        if not ws.research_enabled and not force:
            return {"noop": True, "reason": "research disabled"}

        made = planner.ensure_pipeline(session, ws)
        ex = planner.choose_next(session, ws_id)
        if ex is None:
            # nothing to do; park
            log.info("WS %s: queue empty", ws_id)
            return {"noop": True, "reason": "queue empty"}

        repos.mark_experiment(session, ex, JobStatus.RUNNING.value, worker=ex.worker)
        summary, result, error = "", {}, ""
        started = time.time()
        try:
            summary, result = execute_experiment(session, ws, ex, llm)
        except Exception as e:
            error = str(e)
            summary = f"FAILED: {e}"
            log.exception("experiment %s failed", ex.kind)
        finally:
            status = JobStatus.DONE.value if not error else JobStatus.FAILED.value
            repos.mark_experiment(session, ex, status, result_summary=summary, result_json=result, error=error)
            state = dict(ws.state_json or {})
            ctr = state.setdefault("counters", {})
            ctr["cycles"] = ctr.get("cycles", 0) + 1
            state["last_experiment"] = {"kind": ex.kind, "summary": summary, "ran_s": round(time.time() - started, 2)}
            ws.state_json = state
            ws.research_state = "running" if ws.research_enabled else ws.research_state
        return {"kind": ex.kind, "summary": summary, "error": error or None,
                "elapsed_s": round(time.time() - started, 2)}


async def research_loop(ws_id: str, stop_event: asyncio.Event):
    """Continuous loop while workspace enabled and stop not set."""
    llm = _mk_llm()
    sleep_s = settings.research.loop_sleep_seconds
    while not stop_event.is_set():
        try:
            r = run_cycle(ws_id, llm)
            if r.get("noop"):
                await asyncio.sleep(sleep_s)
                continue
            log.info("cycle %s: %s", ws_id, r)
        except Exception:
            log.exception("research cycle error %s", ws_id)
        await asyncio.sleep(sleep_s)
