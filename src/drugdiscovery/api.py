"""FastAPI application — API-first backend + static frontend + research loop manager."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import repos
from .config import settings
from .db import get_session, init_db
from .llm import LLMProvider
from .lit.providers import get_provider
from .models import JobStatus, KnowledgeType
from . import __version__

log = logging.getLogger("dd.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- research loop manager (in-process) -----------------------------------
_running_tasks: dict[str, asyncio.Task] = {}
_stop_events: dict[str, asyncio.Event] = {}


def _start_loop(ws_id: str):
    from .research import research_loop
    if ws_id in _running_tasks and not _running_tasks[ws_id].done():
        return False
    ev = asyncio.Event()
    _stop_events[ws_id] = ev
    _running_tasks[ws_id] = asyncio.create_task(research_loop(ws_id, ev))
    return True


def _stop_loop(ws_id: str):
    ev = _stop_events.get(ws_id)
    if ev:
        ev.set()
    task = _running_tasks.get(ws_id)
    if task and not task.done():
        # do not await hard; let it check the event
        pass
    return True


# --- schemas ---------------------------------------------------------------
class WorkspaceIn(BaseModel):
    name: str
    disease: str = ""
    description: str = ""
    config: dict = Field(default_factory=dict)


class SearchIn(BaseModel):
    query: str
    size: int = 10
    provider: str = "europepmc"


class ResearchAction(BaseModel):
    action: str  # start | pause | stop | cycle


# --- serialization helpers --------------------------------------------------
def _dt(v):
    return v.isoformat() if isinstance(v, datetime) else v


def _node_d(n):
    return {"id": n.id, "kind": n.kind, "canonical_name": n.canonical_name, "label": n.label,
            "synonyms": n.synonyms, "evidence_level": n.evidence_level, "confidence": n.confidence,
            "knowledge_type": n.knowledge_type, "summary": n.summary, "properties": n.properties}


def _stmt_d(s):
    return {"id": s.id, "text": s.text, "claim_type": s.claim_type, "relation": s.relation,
            "subject": s.subject, "object": s.object, "direction": s.direction,
            "evidence_level": s.evidence_level, "confidence": s.confidence, "species": s.species,
            "model_context": s.model_context, "contradicted": bool(s.properties.get("contradicted")),
            "paper_id": s.source_paper_id, "extractor_model": s.extractor_model}


def _paper_d(p):
    return {"id": p.id, "pmid": p.pmid, "doi": p.doi, "title": p.title, "journal": p.journal,
            "year": p.year, "source": p.source, "url": p.url, "status": p.status,
            "authors": p.authors_json[:8], "n_claims": len(p.statements)}


# --- app --------------------------------------------------------------------
app = FastAPI(title="Drug Discovery Platform", version=__version__)
init_db()


@app.get("/api/status")
def status():
    llm = LLMProvider()
    caps = {}
    try:
        caps = llm.capabilities()
    except Exception as e:
        caps = {"error": str(e)}
    sch = get_session()
    try:
        n_ws = len(repos.list_workspaces(sch))
    finally:
        sch.close()
    return {"service": "drug-discovery", "version": __version__,
            "llm": caps, "embeddings": {"enabled": settings.embeddings.enabled},
            "storage": {"db": settings.storage.db_path}, "workspaces": n_ws,
            "active_research_loops": len(_running_tasks)}


@app.post("/api/workspaces", status_code=201)
def create_ws(body: WorkspaceIn):
    with get_session() as s:
        ws = repos.create_workspace(s, body.name, body.disease, body.description, body.config)
        return {"id": ws.id, "slug": ws.slug, "name": ws.name}


@app.get("/api/workspaces")
def list_ws():
    with get_session() as s:
        return [{"id": w.id, "name": w.name, "slug": w.slug, "disease": w.disease,
                 "research_state": w.research_state, "n_papers": repos.paper_count(s, w.id)}
                for w in repos.list_workspaces(s)]


@app.get("/api/workspaces/{ws_id}")
def get_ws(ws_id: str):
    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        if not ws:
            raise HTTPException(404, "workspace not found")
        return {"id": ws.id, "name": ws.name, "disease": ws.disease, "description": ws.description,
                "config": ws.config_json, "research_state": ws.research_state,
                "state": ws.state_json, "created_at": _dt(ws.created_at)}


@app.get("/api/workspaces/{ws_id}/overview")
def overview(ws_id: str):
    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        if not ws:
            raise HTTPException(404, "workspace not found")
        g = repos.graph_projection(s, ws_id)
        nodes, edges = g["nodes"], g["edges"]
        kinds = {}
        for n in nodes:
            kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
        from .models import Paper, Hypothesis, Statement, Experiment, TimelineEntry
        from sqlalchemy import select, func
        n_papers = s.execute(select(func.count()).select_from(Paper).where(Paper.workspace_id == ws_id)).scalar_one()
        n_hyp = s.execute(select(func.count()).select_from(Hypothesis).where(Hypothesis.workspace_id == ws_id)).scalar_one()
        n_stmt = s.execute(select(func.count()).select_from(Statement).where(Statement.workspace_id == ws_id)).scalar_one()
        n_exq = s.execute(select(func.count()).select_from(Experiment).where(
            Experiment.workspace_id == ws_id, Experiment.status == JobStatus.QUEUED.value)).scalar_one()
        hyp = s.execute(select(Hypothesis).where(Hypothesis.workspace_id == ws_id)
                        .order_by(Hypothesis.confidence.desc()).limit(4)).scalars().all()
        tl = s.execute(select(TimelineEntry).where(TimelineEntry.workspace_id == ws_id)
                       .order_by(TimelineEntry.ts.desc()).limit(20)).scalars().all()
        ex_running = s.execute(select(Experiment).where(
            Experiment.workspace_id == ws_id, Experiment.status == JobStatus.RUNNING.value)).scalars().all()
        return {
            "name": ws.name, "disease": ws.disease, "research_state": ws.research_state,
            "counts": {"nodes": len(nodes), "edges": len(edges), "papers": n_papers,
                       "statements": n_stmt, "hypotheses": n_hyp, "queued_experiments": n_exq, "by_kind": kinds},
            "top_hypotheses": [{"id": h.id, "title": h.title, "confidence": h.confidence,
                                "status": h.status, "evidence": h.evidence_profile} for h in hyp],
            "ongoing_experiments": [{"id": e.id, "kind": e.kind, "question": e.question} for e in ex_running],
            "recent_timeline": [{"ts": _dt(t.ts), "agent": t.agent, "level": t.level, "message": t.message} for t in tl],
        }


@app.get("/api/workspaces/{ws_id}/graph")
def graph(ws_id: str):
    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        if not ws:
            raise HTTPException(404, "workspace not found")
        return repos.graph_projection(s, ws_id)


@app.get("/api/workspaces/{ws_id}/papers")
def papers(ws_id: str, status: Optional[str] = Query(None)):
    from .models import Paper
    from sqlalchemy import select
    with get_session() as s:
        q = select(Paper).where(Paper.workspace_id == ws_id)
        if status:
            q = q.where(Paper.status == status)
        rows = s.execute(q.order_by(Paper.fetch_time.desc()).limit(120)).scalars().all()
        return [_paper_d(p) for p in rows]


@app.get("/api/papers/{pid}")
def paper_detail(pid: str):
    from .models import Paper
    with get_session() as s:
        p = s.get(Paper, pid)
        if not p:
            raise HTTPException(404, "paper not found")
        return {**_paper_d(p), "abstract": p.abstract,
                "statements": [_stmt_d(st) for st in p.statements]}


@app.get("/api/workspaces/{ws_id}/statements")
def statements(ws_id: str, entity: Optional[str] = Query(None)):
    from .models import Statement
    from sqlalchemy import select
    with get_session() as s:
        q = select(Statement).where(Statement.workspace_id == ws_id)
        rows = s.execute(q.order_by(Statement.created_at.desc()).limit(300)).scalars().all()
        out = [_stmt_d(st) for st in rows]
        if entity:
            ent = entity.lower()
            out = [d for d in out if ent in (d["subject"] or "").lower() or ent in (d["object"] or "").lower()]
        return out


@app.get("/api/workspaces/{ws_id}/hypotheses")
def hypotheses(ws_id: str):
    from .models import Hypothesis
    from sqlalchemy import select
    with get_session() as s:
        rows = s.execute(select(Hypothesis).where(Hypothesis.workspace_id == ws_id)
                         .order_by(Hypothesis.created_at.desc())).scalars().all()
        return [{"id": h.id, "title": h.title, "body": h.body, "status": h.status,
                 "confidence": h.confidence, "mechanism_reasoning": h.mechanism_reasoning,
                 "predicted_effect": h.predicted_effect, "uncertainty": h.uncertainty,
                 "testable_predictions": h.testable_predictions, "next_experiment": h.next_experiment,
                 "evidence_profile": h.evidence_profile,
                 "n_support": len(h.supporting_evidence), "n_counter": len(h.counter_evidence)}
                for h in rows]


@app.get("/api/workspaces/{ws_id}/targets")
def targets(ws_id: str):
    from .evidence import score_evidence
    from .hypoth import statements_for_entity
    from .models import Node
    from sqlalchemy import select
    with get_session() as s:
        nodes = s.execute(select(Node).where(Node.workspace_id == ws_id,
                                             Node.kind.in_(["receptor", "enzyme", "protein", "transcription_factor"]))).scalars().all()
        cards = []
        for n in nodes:
            sup, contra = statements_for_entity(s, ws_id, n)
            if not sup and not contra:
                continue
            prof = score_evidence(sup, contra)
            ligands = []
            for e in list(n.out_edges) + list(n.in_edges):
                other = e.target if e.target.id != n.id else e.source
                if other.kind in ("drug", "compound", "drug_class", "hormone", "metabolite"):
                    ligands.append({"name": other.canonical_name, "relation": e.type})
            cards.append({"id": n.id, "name": n.canonical_name, "kind": n.kind,
                          "evidence_level": n.evidence_level, "inline_confidence": n.confidence,
                          "profile": prof, "ligands": sorted({l["name"] for l in ligands})[:8]})
        cards.sort(key=lambda c: c["profile"]["overall_confidence"], reverse=True)
        return cards


@app.get("/api/workspaces/{ws_id}/queue")
def queue(ws_id: str):
    from .models import Experiment
    from sqlalchemy import select
    with get_session() as s:
        rows = s.execute(select(Experiment).where(Experiment.workspace_id == ws_id)
                         .order_by(Experiment.priority.desc()).limit(80)).scalars().all()
        return [{"id": e.id, "kind": e.kind, "question": e.question, "reason": e.reason,
                 "priority": e.priority, "expected_information_gain": e.expected_information_gain,
                 "scientific_importance": e.scientific_importance, "confidence_impact": e.confidence_impact,
                 "computational_cost": e.computational_cost, "status": e.status, "worker": e.worker,
                 "result_summary": e.result_summary, "error": e.error} for e in rows]


@app.get("/api/workspaces/{ws_id}/timeline")
def timeline(ws_id: str, limit: int = Query(50)):
    from .models import TimelineEntry
    from sqlalchemy import select
    with get_session() as s:
        rows = s.execute(select(TimelineEntry).where(TimelineEntry.workspace_id == ws_id)
                         .order_by(TimelineEntry.ts.desc()).limit(limit)).scalars().all()
        return [{"ts": _dt(t.ts), "agent": t.agent, "level": t.level, "message": t.message,
                 "data": t.data_json} for t in reversed(rows)]


@app.post("/api/workspaces/{ws_id}/search")
def manual_search(ws_id: str, body: SearchIn):
    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        if not ws:
            raise HTTPException(404, "workspace not found")
        provider = get_provider(body.provider)
        records = provider.search(body.query, body.size)
        new = 0
        for r in records:
            _, created = repos.upsert_paper(s, ws_id, r)
            new += int(created)
        repos.add_timeline(s, ws_id, "researcher", "info",
                           f"Manual search '{body.query}': {new} new papers", {})
        return {"new": new, "seen": len(records)}


@app.post("/api/workspaces/{ws_id}/research")
def research_control(ws_id: str, body: ResearchAction):
    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        if not ws:
            raise HTTPException(404, "workspace not found")
        act = body.action
        if act == "start":
            repos.set_research_enabled(s, ws, True)
            started = _start_loop(ws_id)
            return {"state": "running", "loop_started": started}
        if act == "pause":
            repos.set_research_enabled(s, ws, False)
            return {"state": "paused"}
        if act == "stop":
            repos.set_research_enabled(s, ws, False)
            _stop_loop(ws_id)
            return {"state": "stopped"}
        if act == "cycle":
            from .research import run_cycle
            res = run_cycle(ws_id)
            return {"state": ws.research_state, "cycle": res}
        raise HTTPException(400, "unknown action")


@app.get("/api/workspaces/{ws_id}/activity")
def activity(ws_id: str):
    from .models import Experiment, TimelineEntry
    from sqlalchemy import select
    with get_session() as s:
        running = s.execute(select(Experiment).where(
            Experiment.workspace_id == ws_id, Experiment.status == JobStatus.RUNNING.value)).scalars().all()
        last = s.execute(select(TimelineEntry).where(TimelineEntry.workspace_id == ws_id)
                         .order_by(TimelineEntry.ts.desc()).limit(1)).scalar_one_or_none()
        return {
            "active_experiments": [{"kind": e.kind, "question": e.question, "worker": e.worker} for e in running],
            "last_activity": {"ts": _dt(last.ts), "agent": last.agent, "message": last.message} if last else None,
        }


# static frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
