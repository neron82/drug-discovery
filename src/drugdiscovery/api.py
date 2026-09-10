"""FastAPI application — API-first backend + static frontend + research loop manager."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import repos
from .config import settings
from .db import get_session, session_scope, init_db
from .llm import LLMProvider
from .lit.providers import get_provider
from .lit.sources import acquire_full_text, source_summary, verified_source_path
from .models import JobStatus, KnowledgeType
from . import __version__

log = logging.getLogger("dd.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- research loop manager (in-process) -----------------------------------
_running_tasks: dict[str, asyncio.Task] = {}
_stop_events: dict[str, asyncio.Event] = {}
# The FastAPI event loop, captured at startup. Sync endpoints (start/pause/
# stop) run in a threadpool worker with NO running loop, so loop tasks must
# be scheduled onto this captured loop rather than asyncio.create_task'd here.
_main_loop: asyncio.AbstractEventLoop | None = None


def _spawn_loop(ws_id: str) -> None:
    """Spawn a FRESH research-loop task. MUST run on the event loop.

    Each task gets its OWN stop-event and a UNIQUE drain-lease holder. The
    unique holder makes stop->immediate-start safe: if the old task is still
    finishing an in-flight cycle when a new task starts, the per-workspace
    drain lease lets only ONE of them drain at a time, so they never interleave.
    """
    import uuid
    from .research import research_loop
    ev = asyncio.Event()
    holder = f"api-loop:{ws_id}:{uuid.uuid4().hex[:8]}"
    _stop_events[ws_id] = ev
    _running_tasks[ws_id] = asyncio.create_task(research_loop(ws_id, ev, holder=holder))


def _start_on_loop(ws_id: str) -> None:
    """Check-and-spawn, ATOMIC on the single-threaded event loop.

    Two rapid worker-thread start requests both schedule this callback, but
    they execute sequentially on the loop: the first spawns, the second sees
    the freshly-spawned (active) task and does not spawn. A task whose
    stop-event is set is "exiting" and is replaced; an active task is kept.
    """
    task = _running_tasks.get(ws_id)
    ev = _stop_events.get(ws_id)
    if task is not None and not task.done() and ev is not None and not ev.is_set():
        return  # already actively draining
    _spawn_loop(ws_id)


def _start_loop(ws_id: str) -> bool:
    """Ensure a loop is draining this workspace. Returns True if (re)started.

    The authoritative check-and-spawn runs on the event loop (_start_on_loop);
    the return value is a best-effort worker-thread view for the API response.
    """
    task = _running_tasks.get(ws_id)
    ev = _stop_events.get(ws_id)
    already_active = (task is not None and not task.done()
                      and ev is not None and not ev.is_set())
    try:
        asyncio.get_running_loop()
        _start_on_loop(ws_id)  # already on the event loop (e.g. startup resume)
    except RuntimeError:
        # threadpool worker (sync endpoint): schedule onto the captured loop
        if _main_loop is None or _main_loop.is_closed():
            raise RuntimeError("main event loop not available to start research loop")
        _main_loop.call_soon_threadsafe(_start_on_loop, ws_id)
    return not already_active


def _stop_loop(ws_id: str) -> bool:
    ev = _stop_events.get(ws_id)
    if not ev:
        return True
    # Event.set() is not thread-safe; run it on the loop when called from a
    # threadpool worker (the sync stop endpoint).
    def _do_stop() -> None:
        ev.set()
    try:
        asyncio.get_running_loop()
        _do_stop()
    except RuntimeError:
        if _main_loop is not None and not _main_loop.is_closed():
            _main_loop.call_soon_threadsafe(_do_stop)
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
            "effect_size": s.effect_size, "significance": s.significance,
            "quotation": s.quotation, "dose": s.properties.get("dose", ""),
            "endpoint": s.properties.get("endpoint", ""),
            "support_citations": s.support_citations[:5],
            "contradict_citations": s.contradict_citations[:5],
            "paper_id": s.source_paper_id, "extractor_model": s.extractor_model,
            "extraction_source": s.properties.get("extraction_source") or {
                "kind": "abstract", "legacy": True, "truncated": None}}


def _source_d(source):
    base = f"/api/papers/{source.paper_id}/sources/{source.id}"
    return {**source_summary(source), "text_url": f"{base}?format=text",
            "raw_url": f"{base}?format=raw"}


def _paper_d(p):
    return {"id": p.id, "pmid": p.pmid, "doi": p.doi, "title": p.title, "journal": p.journal,
            "year": p.year, "source": p.source, "url": p.url, "status": p.status,
            "authors": p.authors_json[:8], "n_claims": len(p.statements),
            "sources": [_source_d(source) for source in p.sources]}


# --- app --------------------------------------------------------------------
app = FastAPI(title="Drug Discovery Platform", version=__version__)
init_db()


def _recover_stale_jobs() -> int:
    """Recover interrupted jobs before any persistent loop is resumed."""
    recovered = 0
    with session_scope() as s:
        rows = repos.recover_stale_experiments(
            s, stale_after_minutes=settings.research.stale_running_after_minutes,
        )
        for ex in rows:
            repos.add_timeline(
                s, ex.workspace_id, "system", "warn",
                f"Recovered stale {ex.kind} experiment and returned it to the queue",
                {"experiment_id": ex.id, "lease_minutes": settings.research.stale_running_after_minutes},
            )
        recovered = len(rows)
    if recovered:
        log.warning("Startup recovered %d stale experiment(s)", recovered)
    return recovered


_RECOVERED_AT_STARTUP = _recover_stale_jobs()


@app.on_event("startup")
async def resume_enabled_workspaces() -> None:
    """Resume enabled research workspaces after an API process restart."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()  # capture for threadpool start/stop
    with get_session() as s:
        ids = [w.id for w in repos.list_workspaces(s) if w.research_enabled]
    for ws_id in ids:
        _start_loop(ws_id)
    if ids:
        log.info("Resumed %d enabled research workspace(s)", len(ids))


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
            "active_research_loops": sum(1 for task in _running_tasks.values() if not task.done()),
            "recovered_stale_jobs_at_startup": _RECOVERED_AT_STARTUP,
            "budgets": {"max_lit_requests_per_hour": settings.research.budgets.max_lit_requests_per_hour,
                        "max_simulations_per_hour": settings.research.budgets.max_simulations_per_hour}}


@app.post("/api/workspaces", status_code=201)
def create_ws(body: WorkspaceIn):
    with session_scope() as s:
        ws = repos.create_workspace(s, body.name, body.disease, body.description, body.config)
        return {"id": ws.id, "slug": ws.slug, "name": ws.name}


@app.get("/api/workspaces")
def list_ws():
    with get_session() as s:
        return [{"id": w.id, "name": w.name, "slug": w.slug, "disease": w.disease,
                 "research_state": w.research_state, "research_enabled": w.research_enabled,
                 "n_papers": repos.paper_count(s, w.id)}
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




@app.get("/api/nodes/{node_id}")
def node_card(node_id: str):
    """Node research card (spec 25): fields, typed edges w/ provenance, statements, papers."""
    from .models import Node, Edge, Statement, Paper, Compound, Structure
    from sqlalchemy import select
    with get_session() as s:
        n = s.get(Node, node_id)
        if n is None:
            raise HTTPException(404, "node not found")
        edges = s.execute(select(Edge).where(
            (Edge.source_id == node_id) | (Edge.target_id == node_id))).scalars().all()
        stmts = s.execute(select(Statement).where(
            Statement.workspace_id == n.workspace_id,
            (Statement.subject.ilike(n.canonical_name)) |
            (Statement.object.ilike(n.canonical_name)))).scalars().all()
        paper_ids = sorted({st.source_paper_id for st in stmts if st.source_paper_id})
        papers = []
        if paper_ids:
            for p in s.execute(select(Paper).where(Paper.id.in_(paper_ids))).scalars():
                papers.append({"id": p.id, "title": p.title[:180], "journal": p.journal,
                               "pmid": p.pmid, "doi": p.doi})
        compound = None
        cid = (n.properties or {}).get("compound_id")
        if cid:
            c = s.get(Compound, cid)
            if c:
                compound = {"id": c.id, "name": c.canonical_name,
                            "smiles": c.canonical_smiles, "status": c.research_status}
        sims = []
        st_ids = []
        for st in s.execute(select(Structure).where(
                Structure.protein_node_id == node_id)).scalars():
            st_ids.append(st.id)
        if st_ids:
            from .models import SimulationRun
            for sr in s.execute(select(SimulationRun).where(
                    SimulationRun.structure_id.in_(st_ids))).scalars():
                sims.append({"id": sr.id, "kind": sr.kind, "status": sr.status,
                             "score": sr.score, "compound_id": sr.compound_id,
                             "software": sr.software_version, "seed": sr.seed})
        structures = [{"id": st.id, "source": st.source, "source_id": st.source_id,
                       "resolution": st.resolution}
                      for st in s.execute(select(Structure).where(
                          Structure.protein_node_id == node_id)).scalars()]
        def _nbr(e):
            t = e.target if e.source_id == node_id else e.source
            return {"neighbor_id": t.id, "neighbor": t.label or t.canonical_name,
                    "neighbor_kind": t.kind,
                    "direction": "out" if e.source_id == node_id else "in"}
        return {
            "id": n.id, "kind": n.kind, "name": n.canonical_name,
            "label": n.label or n.canonical_name, "synonyms": n.synonyms,
            "summary": n.summary, "properties": n.properties,
            "knowledge_type": n.knowledge_type, "evidence_level": n.evidence_level,
            "confidence": round(n.confidence, 3),
            "edges": [{**_nbr(e), "id": e.id, "type": e.type,
                       "knowledge_type": e.knowledge_type,
                       "evidence_level": e.evidence_level,
                       "confidence": round(e.confidence, 3),
                       "provenance": e.provenance or []} for e in edges],
            "statements": [_stmt_d(st) for st in stmts[:40]],
            "papers": papers, "compound": compound, "structures": structures,
            "simulations": sims,
        }


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


@app.get("/api/papers/{pid}/sources/{source_id}")
def paper_source(pid: str, source_id: str, format: Literal["text", "raw"] = "text"):
    from .models import PaperSource
    with get_session() as s:
        source = s.get(PaperSource, source_id)
        if source is None or source.paper_id != pid:
            raise HTTPException(404, "paper source not found")
        try:
            path = verified_source_path(source, format=format)
        except FileNotFoundError:
            raise HTTPException(404, "retained source artifact is missing") from None
        except (ValueError, OSError):
            raise HTTPException(409, "retained source artifact failed integrity verification") from None
        raw_xml = format == "raw" and source.kind == "full_text"
        return FileResponse(
            path, media_type="application/xml" if raw_xml else "text/plain",
            filename=f"{source.id}.xml" if raw_xml else f"{source.id}.txt",
            content_disposition_type="attachment" if format == "raw" else "inline",
            headers={"X-Content-Type-Options": "nosniff",
                     "Content-Security-Policy": "default-src 'none'; sandbox"},
        )


@app.post("/api/papers/{pid}/full-text")
def paper_full_text(pid: str):
    from .models import Paper
    with get_session() as s:
        p = s.get(Paper, pid)
        if p is None:
            raise HTTPException(404, "paper not found")
        try:
            source = acquire_full_text(s, p)
            if source is None:
                raise HTTPException(409, "PMC full text is unavailable or could not be retained; existing claims are unchanged")
            if source.paper_id != pid or source.kind != "full_text":
                raise HTTPException(409, "no verified full-text source was acquired")
            verified_source_path(source, format="text")
            verified_source_path(source, format="raw")
        except (ValueError, OSError):
            raise HTTPException(409, "retained full text failed integrity verification; existing claims are unchanged") from None
        s.commit()
        return _source_d(source)


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


# ---------------------------------------------------------------------------
# Compounds (Phase 2)
# ---------------------------------------------------------------------------
@app.get("/api/compounds")
def list_compounds():
    with get_session() as s:
        return repos.list_compounds(s)


@app.get("/api/compounds/{cid}")
def compound_detail(cid: str):
    with get_session() as s:
        d = repos.get_compound(s, cid)
        if d is None:
            raise HTTPException(404, "compound not found")
        # safety screen (spec 12): cheap deterministic screen, computed on read
        if d.get("smiles"):
            from .chem.safety import safety_alerts
            d["safety"] = safety_alerts(d["smiles"])
        return d


@app.get("/api/compounds/{cid}/svg")
def compound_svg(cid: str):
    from .chem import engine
    from fastapi.responses import Response
    with get_session() as s:
        d = repos.get_compound(s, cid)
    if d is None:
        raise HTTPException(404, "compound not found")
    svg = engine.svg_2d(d["smiles"])
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/compounds/{cid}/similar")
def compound_similar(cid: str, limit: int = Query(20), smiles: str = ""):
    from .chem import engine
    with get_session() as s:
        if smiles:
            fp = engine.morgan_fp_hex(smiles)
        else:
            d = repos.get_compound(s, cid)
            fp = (d or {}).get("properties", {}).get("fingerprint_hex")
        if not fp:
            raise HTTPException(400, "no computable structure")
        return repos.similarity_search(s, fp, limit=limit)


@app.get("/api/workspaces/{ws_id}/compounds")
def workspace_compounds(ws_id: str):
    """Catalogue restricted to compounds wired into this workspace + their KG edges."""
    from .chem import engine
    from .models import Node, Edge
    from sqlalchemy import select
    with get_session() as s:
        nodes = s.execute(select(Node).where(Node.workspace_id == ws_id,
                                             Node.kind.in_(["compound", "drug"]))).scalars().all()
        out = []
        for n in nodes:
            cid = (n.properties or {}).get("compound_id")
            c = repos.get_compound(s, cid) if cid else None
            edges = s.execute(select(Edge).where(Edge.workspace_id == ws_id,
                                                 (Edge.source_id == n.id) | (Edge.target_id == n.id))).scalars().all()
            targets = []
            for e in edges:
                other = e.target if e.source_id == n.id else e.source
                if other.kind in ("receptor", "enzyme", "protein", "transcription_factor"):
                    prov = e.provenance if isinstance(e.provenance, dict) else {}
                    targets.append({"id": other.id, "name": other.canonical_name,
                                    "relation": e.type, "pchembl": prov.get("pchembl"),
                                    "evidence_level": e.evidence_level})
            out.append({"node_id": n.id, "compound": c, "targets": targets})
        out.sort(key=lambda x: (x["compound"] or {}).get("name", ""))
        return out


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


@app.get("/api/workspaces/{ws_id}/queue/metrics")
def queue_metrics_endpoint(ws_id: str):
    """Honest queue accounting: created/completed/failed/cancelled/skipped,
    pending by kind, discovery-vs-retry split, newly-scheduled, and the live
    drain-lease holder. Lets queue growth be attributed, not guessed."""
    with get_session() as s:
        return repos.queue_metrics(s, ws_id)


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
    with session_scope() as s:
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
    act = body.action
    if act == "cycle":
        # run_cycle owns its own session/transaction — do NOT nest it inside
        # an outer open transaction (nested-write hazard + stale ws).
        from .research import run_cycle
        res = run_cycle(ws_id, force=True)
        with get_session() as s:
            ws = repos.get_workspace(s, ws_id)
            state = ws.research_state if ws else None
        return {"state": state, "cycle": res}
    # Do the DB mutation in its own transaction and COMMIT it BEFORE touching
    # the loop. The loop's first iteration reads research_enabled; if the loop
    # were spawned before this commit it could read the stale value and exit
    # immediately (the stop->start race). So: mutate+commit, then control.
    with session_scope() as s:
        ws = repos.get_workspace(s, ws_id)
        if not ws:
            raise HTTPException(404, "workspace not found")
        cancelled = 0
        if act == "start":
            repos.set_research_enabled(s, ws, True)
        elif act == "pause":
            repos.set_research_enabled(s, ws, False)
        elif act == "stop":
            # Explicit stop: disable research and cancel all QUEUED work
            # (running work is left to finish — bounded by the stale-running
            # lease). History is preserved, never deleted.
            repos.set_research_enabled(s, ws, False)
            cancelled = repos.cancel_queued_experiments(s, ws.id, "explicit stop")
        else:
            raise HTTPException(400, "unknown action")
    # COMMITTED above. Now control the loop, outside the transaction.
    if act == "start":
        started = _start_loop(ws_id)
        return {"state": "running", "loop_started": started}
    if act == "pause":
        return {"state": "paused"}
    if act == "stop":
        _stop_loop(ws_id)
        return {"state": "stopped", "cancelled_queued": cancelled}


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


# ---------------------------------------------------------------------------
# Structures (Phase 3)
# ---------------------------------------------------------------------------
@app.get("/api/workspaces/{ws_id}/structures")
def list_structures(ws_id: str):
    from .models import Structure, Node
    from sqlalchemy import select
    with get_session() as s:
        rows = s.execute(select(Structure).where(Structure.workspace_id == ws_id)
                         .order_by(Structure.source.asc(), Structure.resolution.is_(None).asc(), Structure.resolution.asc())
                         ).scalars().all()
        node_ids = {r.protein_node_id for r in rows if r.protein_node_id}
        names = {}
        if node_ids:
            ns = s.execute(select(Node).where(Node.id.in_(node_ids))).scalars().all()
            names = {n.id: n.canonical_name for n in ns}
        return [{
            "id": r.id, "source": r.source, "source_id": r.source_id,
            "protein": names.get(r.protein_node_id, "unlinked"),
            "uniprot": r.uniprot_accession, "method": r.method,
            "resolution": r.resolution, "organism": r.organism,
            "ligand_count": len(r.ligands or []),
            "ligands": (r.ligands or [])[:10],
            "limitations": r.limitations, "fetched_at": _dt(r.fetched_at),
        } for r in rows]


@app.get("/api/structures/{sid}")
def structure_detail(sid: str):
    from .models import Structure, Node
    from sqlalchemy import select
    with get_session() as s:
        r = s.get(Structure, sid)
        if r is None:
            raise HTTPException(404, "structure not found")
        node = s.get(Node, r.protein_node_id) if r.protein_node_id else None
        return {
            "id": r.id, "source": r.source, "source_id": r.source_id,
            "protein": node.canonical_name if node else None,
            "protein_node_id": r.protein_node_id,
            "uniprot": r.uniprot_accession, "method": r.method,
            "resolution": r.resolution, "confidence": r.confidence,
            "organism": r.organism, "ligands": r.ligands or [],
            "file_hash": r.file_hash, "limitations": r.limitations,
            "fetched_at": _dt(r.fetched_at),
        }


@app.get("/api/structures/{sid}/file")
def structure_file(sid: str):
    from .models import Structure
    from fastapi.responses import FileResponse
    with get_session() as s:
        r = s.get(Structure, sid)
    if r is None or not r.file_path:
        raise HTTPException(404, "structure file not found")
    return FileResponse(r.file_path, media_type="chemical/x-cif", filename=f"{r.source_id}.cif")


# ---------------------------------------------------------------------------
# Simulation runs (Phase 4 docking)
# ---------------------------------------------------------------------------
@app.get("/api/workspaces/{ws_id}/simulations")
def list_simulations(ws_id: str):
    from .sim.runner import list_runs
    from .models import Compound
    with get_session() as s:
        runs = list_runs(s, ws_id)
        names = {}
        for r in runs:
            if r.compound_id:
                c = s.get(Compound, r.compound_id)
                names[r.compound_id] = c.canonical_name if c else None
        return [
            {
                "id": r.id, "kind": r.kind, "software": r.software,
                "software_version": r.software_version, "score": r.score,
                "rmsd_lb": r.rmsd_lb, "seed": r.seed, "runtime_s": r.runtime_s,
                "hardware": r.hardware, "status": r.status, "error": r.error,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "pose_path": r.pose_path, "params": r.params_json,
                "structure_id": r.structure_id, "compound_id": r.compound_id,
                "receptor_hash": r.receptor_hash, "ligand_hash": r.ligand_hash,
                "compound_name": names.get(r.compound_id),
                "assumptions": r.assumptions,
            }
            for r in runs
        ]


@app.get("/api/simulations/{sid}")
def simulation_detail(sid: str):
    from .models import SimulationRun
    with get_session() as s:
        r = s.get(SimulationRun, sid)
    if r is None:
        raise HTTPException(404, "simulation run not found")
    return {
        "id": r.id, "kind": r.kind, "software": r.software,
        "software_version": r.software_version, "score": r.score, "rmsd_lb": r.rmsd_lb,
        "seed": r.seed, "receptor_hash": r.receptor_hash, "ligand_hash": r.ligand_hash,
        "pose_path": r.pose_path, "runtime_s": r.runtime_s, "hardware": r.hardware,
        "params": r.params_json, "assumptions": r.assumptions, "status": r.status,
        "error": r.error, "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@app.post("/api/simulations/{sid}/reproduce")
def reproduce_simulation(sid: str):
    """Spec 20: re-enqueue an experiment with the exact recorded parameters
    (same structure, compound, seed, box, exhaustiveness). The executor re-runs
    it on the next cycle; hashes let the new run be compared against the old."""
    from .models import SimulationRun, Structure, Compound
    from sqlalchemy import select
    from .sim.engine import DEFAULT_BOX_PADDING
    with session_scope() as s:
        r = s.get(SimulationRun, sid)
        if r is None:
            raise HTTPException(404, "simulation run not found")
        if r.status != "done":
            raise HTTPException(409, "only completed runs can be reproduced")
        if r.kind == "docking" and r.seed == 0:
            raise HTTPException(
                409, "The effective random seed was not recorded for this docking run; "
                     "exact reproduction is unavailable.")
        params = dict(r.params_json or {})
        params["structure_id"] = r.structure_id
        params["seed"] = r.seed
        if r.kind == "docking":
            # Reconstruct the pocket so engine padding restores the recorded final box.
            if params.get("center") and params.get("box_size"):
                params["pocket"] = {"center": params["center"],
                                    "size": float(params["box_size"][0]) - DEFAULT_BOX_PADDING}
            if r.compound_id:
                c = s.get(Compound, r.compound_id)
                if c:
                    params["compound_id"] = c.id
                    params["compound_smiles"] = c.canonical_smiles
        elif r.kind == "md":
            params.setdefault("steps", 2000)
            params.setdefault("temperature", 300.0)
        st = s.get(Structure, r.structure_id) if r.structure_id else None
        ex = repos.create_experiment(
            s, r.workspace_id, r.kind,
            question=f"Reproduce simulation {sid[:8]} ({r.software} {r.software_version})",
            reason="Explicit reproduction request (spec 20): identical recorded "
                   "parameters, seed, and inputs; compare hashes and score afterwards.",
            expected_information_gain=0.3, scientific_importance=0.4,
            confidence_impact=0.2, computational_cost=0.9, priority=0.95,
            params=params, worker="simulation")
        return {"experiment_id": ex.id, "kind": r.kind,
                "original_hashes": {"receptor": r.receptor_hash, "ligand": r.ligand_hash},
                "state": "queued"}


@app.get("/api/simulations/{sid}/pose")
def simulation_pose(sid: str):
    from .models import SimulationRun
    from fastapi.responses import FileResponse
    with get_session() as s:
        r = s.get(SimulationRun, sid)
    if r is None or not r.pose_path:
        raise HTTPException(404, "pose not found")
    import os
    if not os.path.exists(r.pose_path):
        raise HTTPException(404, "pose file missing on disk")
    return FileResponse(r.pose_path, media_type="chemical/x-pdbqt",
                        filename=f"{r.id[:8]}_pose.pdbqt")


# static frontend
@app.get("/api/workspaces/{ws_id}/candidates")
def candidates(ws_id: str, weights: str = ""):
    """Multi-factor candidate ranking (Spec section 31).

    weights: optional JSON object overriding DEFAULT_WEIGHTS per dimension."""
    from .ranking import DEFAULT_WEIGHTS, rank_candidates
    overrides = None
    if weights:
        import json as _json
        try:
            overrides = _json.loads(weights)
        except Exception:
            raise HTTPException(status_code=400, detail="weights must be valid JSON")
    with get_session() as s:
        rows = rank_candidates(s, ws_id, overrides)
    return {"candidates": rows, "default_weights": DEFAULT_WEIGHTS}


@app.get("/api/workspaces/{ws_id}/critiques")
def critiques(ws_id: str):
    """Skeptic critique history: last adversarial review per hypothesis."""
    from sqlalchemy import select as _select
    from .models import Hypothesis as _Hyp
    with get_session() as s:
        hs = list(s.execute(_select(_Hyp).where(
            _Hyp.workspace_id == ws_id)).scalars())
        out = []
        for h in hs:
            last = (h.evidence_profile or {}).get("last_critique")
            if isinstance(last, dict) and last.get("by") == "skeptic":
                out.append({"hypothesis_id": h.id, "title": h.title,
                            "status": h.status, "confidence": h.confidence,
                            "critique": last})
        out.sort(key=lambda r: r["critique"].get("at", ""), reverse=True)
    return {"critiques": out}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
