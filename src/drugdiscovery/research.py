"""Autonomous research loop + experiment executor.

Loop: OBSERVE → PLUCK next prioritized experiment → EXECUTE → UPDATE KG/hypotheses →
PERSIST → SCHEDULE follow-ups → repeat while enabled.
Executes only typed, validated experiments through the dispatcher; no arbitrary shell.
"""
from __future__ import annotations

import logging
import time
import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from . import hypoth, repos
from .config import settings
from .db import get_session
from .extract import process_paper
from .lit.providers import get_provider
from .llm import LLMProvider
from .models import Hypothesis, JobStatus, Node, Paper

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
            paper, created = repos.upsert_paper(session, ws.id, rec)
            if created:
                new += 1
                # queue extraction for the fresh paper so flow continues
                repos.create_experiment(
                    session, ws.id, "paper_extraction",
                    question=f"Extract claims: {rec.title[:60]}…",
                    reason="Ingested paper needs structured extraction",
                    expected_information_gain=0.7, scientific_importance=0.6,
                    confidence_impact=0.4, computational_cost=1.2,
                    priority=0.5, params={"paper_id": paper.id},
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
        topic = params.get("topic", ws.disease or "disease")
        q = f"{topic} AND (negative OR \"no effect\" OR \"failed\" OR \"no association\" OR \"did not\")"
        records = provider.search(q, int(params.get("size", 8)))
        new = 0
        for rec in records:
            paper, created = repos.upsert_paper(session, ws.id, rec)
            new += int(created)
            if created:
                repos.create_experiment(session, ws.id, "paper_extraction",
                                        question=f"Extract claims: {rec.title[:60]}…",
                                        reason="Skeptic ingestion", expected_information_gain=0.7,
                                        scientific_importance=0.6, confidence_impact=0.6,
                                        computational_cost=1.2, priority=0.6,
                                        params={"paper_id": paper.id},
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

    # ---------------- Phase 2: chemical intelligence ----------------
    if kind == "compound_discovery":
        from .chem.catalogue import compounds_for_target_chembl
        tpid = params.get("target_node_id")
        tn = session.get(Node, tpid) if tpid else None
        if tn is None:
            return "target node not found", {"ingested": 0}
        report = compounds_for_target_chembl(tn, session, ws, max_n=int(params.get("max_n", 12)))
        repos.add_timeline(session, ws.id, "chemist", "success" if report.get("ok") else "warn",
                           f"Chemical discovery vs {tn.label or tn.canonical_name}: {report.get('ingested', 0)} "
                           f"catalogued ({report.get('reason', 'ChEMBL measured activity')})", report)
        return (f"{report.get('ingested', 0)} compounds catalogued for {tn.label or tn.canonical_name}",
                report)

    if kind == "compound_catalogue_refresh":
        from .chem.catalogue import fetch_from_pubchem
        name = params.get("name")
        if not name:
            return "no compound name", {}
        rec = fetch_from_pubchem(name)
        if rec is None or not rec.smiles:
            repos.add_timeline(session, ws.id, "chemist", "warn",
                               f"PubChem: no structure for '{name}'", {"name": name})
            return f"PubChem lookup failed for {name}", {"found": False}
        props = {}
        from .chem import engine, catalogue
        cs = engine.canonical_smiles(rec.smiles)
        key = engine.inchi_and_key(cs)[1] if cs else ""
        compound, created = repos.upsert_compound(
            session, name=name, canonical_smiles=cs, inchikey=key,
            synonyms=rec.synonyms,
            sources={"pubchem": {"cid": rec.source_id, "url": rec.source_url}},
            properties=engine.descriptors(cs) if cs else {},
        )
        # wire a compound KG node (no edge without a measured target link)
        repos.find_node(session, ws.id, "compound", name) or repos.upsert_node(
            session, ws.id, "compound", name, label=name,
            evidence_level="computational", confidence=0.1, knowledge_type="source_fact",
            properties={"compound_id": compound.id, "inchikey": key})
        repos.add_timeline(session, ws.id, "chemist", "info", f"PubChem catalogue {name} "
                           f"[{key[:10]}…] {'created' if created else 'updated'}", {"cid": rec.source_id})
        return f"Catalogued {name} [{key[:10]}…]", {"found": True, "created": created}

    if kind == "similarity_expansion":
        from .chem import engine
        from .chem.providers import PubChemProvider
        seed = params.get("seed")
        smi = ""
        if seed:
            rec = PubChemProvider().fetch_by_name(seed)
            smi = rec.smiles if rec else ""
        elif params.get("smiles"):
            smi = params["smiles"]
        fp = engine.morgan_fp_hex(smi) if smi else None
        if fp is None:
            return "no valid structure for similarity", {}
        hits = repos.similarity_search(session, fp, limit=int(params.get("limit", 20)))
        repos.add_timeline(session, ws.id, "chemist", "info",
                           f"Similarity search around {seed or smi}: {len(hits)} analogues found", {"hits": len(hits)})
        return f"{len(hits)} similarity neighbours for {seed or 'query'}", {"n": len(hits),
               "hits": [{h['name']: h['similarity']} for h in hits]}

    if kind == "structure_analysis":
        from .struct import acquisition
        node = None
        if params.get("node_id"):
            node = session.get(Node, params["node_id"])
        if node is None:
            rep = acquisition.acquire_structures(session, ws, limit=int(params.get("limit", 6)))
        else:
            r1 = acquisition.acquire_for_target(session, ws, node)
            rep = {"targets_scanned": 1, "acquired_new": r1.get("new", 0),
                   "reports": [{"node": node.canonical_name, **r1}]}
        n_new = rep.get("acquired_new", 0)
        repos.add_timeline(session, ws.id, "structural_biologist", "discovery",
                           f"Structure acquisition: scanned {rep.get('targets_scanned', 0)} targets, "
                           f"{n_new} new structures stored (PDB/AlphaFold)",
                           {"acquired_new": n_new})
        return (f"structures: {n_new} new from {rep.get('targets_scanned', 0)} targets", rep)

    if kind == "docking":
        from .sim import runner as sim_runner
        after = dict(ex.result_json or {})
        rep = sim_runner.run_docking(session, ex, params)
        after.update({"simulation_run_id": rep["simulation_run_id"],
                      "score": rep["score"]})
        repos.add_timeline(
            session, ws.id, "simulation", "result",
            f"Docking: {rep['score']:.2f} kcal/mol ({ex.params_json.get('compound_id', '?')} "
            f"vs {rep['structure_source'] if 'structure_source' in rep else 'target'})",
            {"score": rep["score"], "simulation_run_id": rep["simulation_run_id"]})
        return (rep["summary"], after)

    if kind == "molecular_dynamics":
        from .sim import runner as sim_runner
        rep = sim_runner.run_md(session, ex, params)
        repos.add_timeline(
            session, ws.id, "simulation", "result",
            f"MD: {rep['summary']}", {"simulation_run_id": rep["simulation_run_id"]})
        after = dict(ex.result_json or {})
        after.update({"simulation_run_id": rep["simulation_run_id"]})
        return (rep["summary"], after)

    if kind == "hypothesis_critique":
        from . import critique
        if llm is None:
            return "hypothesis_critique requires an LLM", {}
        if params.get("hypothesis_id"):
            h = session.get(Hypothesis, params["hypothesis_id"])
            if h is None:
                return "Hypothesis not found", {}
            res = critique.critique_hypothesis(session, ws, h, llm)
            summary = f"Critique: verdict={res.get('verdict')} {res.get('before')}->{res.get('after')}"
        else:
            res = critique.critique_active(session, ws, llm, limit=int(params.get("limit", 3)))
            summary = f"Skeptic pass: {len(res)} reviewed"
            res = res[:3]   # list branch: cap payload; dict branch passes through
        return (summary, {"results": res})

    return f"Unhandled experiment kind: {kind}", {}


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
    cfg = ws.config_json or {}
    cells = ", ".join(cfg.get("relevant_cell_types") or ["the relevant cell type"])
    questions.append(f"Which disease-relevant cell type ({cells}) mediates the strongest phenotype effect?")
    questions.append(f"Which mechanism, if modulated, most robustly reverses the phenotype ({', '.join(cfg.get('phenotype') or ['progression'])}) without driving unwanted off-target effects?")
    questions.append("Which reported associations fail replication or show negative results in independent cohorts?")
    questions.append("What are the major unresolved unknowns limiting the top-ranked hypothesis?")
    return list(dict.fromkeys(questions[:10]))


# ---------------------------------------------------------------------------
# Cycle + runner
# ---------------------------------------------------------------------------
# --- compute budgets (spec 29) ---------------------------------------------
_LIT_KINDS = {"literature_search", "paper_extraction", "contradiction_search"}
_SIM_KINDS = {"docking", "molecular_dynamics", "structure_analysis"}


def _budget_gate(ws, kind: str) -> str | None:
    """Rolling 1-hour budget check against ws.state_json counters.
    Returns None if allowed, else the reason string."""
    b = settings.research.budgets
    state = dict(ws.state_json or {})
    bud = state.setdefault("budget", {"window_start": _utc_iso(), "lit": 0, "sim": 0})
    from datetime import datetime, timezone, timedelta
    try:
        start = datetime.fromisoformat(bud.get("window_start", ""))
    except Exception:
        start = datetime.now(timezone.utc); bud["window_start"] = _utc_iso(); bud["lit"] = 0; bud["sim"] = 0
    if datetime.now(timezone.utc) - start > timedelta(hours=1):
        bud["window_start"] = _utc_iso(); bud["lit"] = 0; bud["sim"] = 0
    ws.state_json = state
    if kind in _LIT_KINDS and b.max_lit_requests_per_hour and bud["lit"] >= b.max_lit_requests_per_hour:
        return f"literature budget exhausted ({b.max_lit_requests_per_hour}/h)"
    if kind in _SIM_KINDS and b.max_simulations_per_hour and bud["sim"] >= b.max_simulations_per_hour:
        return f"simulation budget exhausted ({b.max_simulations_per_hour}/h)"
    return None


def _budget_record(ws, kind: str) -> None:
    state = dict(ws.state_json or {})
    bud = state.get("budget") or {"window_start": _utc_iso(), "lit": 0, "sim": 0}
    if kind in _LIT_KINDS:
        bud["lit"] = bud.get("lit", 0) + 1
    if kind in _SIM_KINDS:
        bud["sim"] = bud.get("sim", 0) + 1
    state["budget"] = bud
    ws.state_json = state


def _utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _entity_counts(session, ws_id: str) -> dict:
    """Workspace-scoped entity tallies used for information-gain detection."""
    from .models import Compound, Hypothesis, Paper, SimulationRun, Statement, Structure
    from sqlalchemy import func

    def _cnt(model, scoped: bool = True) -> int:
        q = select(func.count()).select_from(model)
        if scoped:
            q = q.where(model.workspace_id == ws_id)
        return session.execute(q).scalar_one()

    return {
        "papers": _cnt(Paper),
        "statements": _cnt(Statement),
        "structures": _cnt(Structure),
        "simulation_runs": _cnt(SimulationRun),
        "hypotheses": _cnt(Hypothesis),
        "compounds": _cnt(Compound, scoped=False),  # compound catalogue is global
    }


def run_cycle(ws_id: str, llm: LLMProvider | None = None, force: bool = False,
              holder: str | None = None) -> dict:
    """One research cycle. Returns a summary dict (or {"noop": true}).

    Bounded-autonomy guarantees:
      * single-drain lease — at most one holder runs a cycle at a time;
      * stagnation stop — consecutive no-gain cycles pause the workspace.

    ``holder`` identifies the drainer for the lease. When omitted (manual /
    CLI / API-endpoint callers) a UNIQUE id is minted per invocation so two
    concurrent manual drains cannot share a holder and both win the lease.
    The persistent loop passes a stable per-workspace holder instead.
    """
    from . import planner
    from .db import session_scope
    if holder is None:
        holder = f"manual:{uuid.uuid4().hex[:12]}"
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

        # compute budget (spec 29): skip over-budget kinds, keep them queued
        blocked = _budget_gate(ws, ex.kind)
        if blocked and not force:
            return {"noop": True, "reason": blocked}

        # single-drain lease: refuse to interleave with another live drainer
        if not repos.acquire_drain_lease(session, ws, holder,
                                         int(settings.research.drain_lease_seconds)):
            other = repos.drain_lease_holder(session, ws_id)
            log.info("WS %s: drain lease held by %s, skipping cycle", ws_id, other)
            return {"noop": True, "reason": f"drain in progress (holder: {other})"}

        _budget_record(ws, ex.kind)
        before = _entity_counts(session, ws_id)
        repos.mark_experiment(session, ex, JobStatus.RUNNING.value, worker=ex.worker)
        # Commit NOW to release the SQLite write lock before the heavy work.
        # Holding a transaction across a ~27s docking (or minutes of MD) would
        # block every other writer (control endpoints, the drain counter) for
        # the whole computation -> "database is locked" storms. The result is
        # persisted in a fresh short transaction afterwards.
        session.commit()
        summary, result, error = "", {}, ""
        # Workers may replace result_json; retain interruption/recovery history
        # independently so finishing a recovered job cannot erase its provenance.
        prior_result = dict(ex.result_json or {})
        started = time.time()
        try:
            summary, result = execute_experiment(session, ws, ex, llm)
        except Exception as e:
            error = str(e)
            summary = f"FAILED: {e}"
            log.exception("experiment %s failed", ex.kind)
        finally:
            status = JobStatus.DONE.value if not error else JobStatus.FAILED.value
            result = {**prior_result, **(ex.result_json or {}), **result}
            repos.mark_experiment(session, ex, status, result_summary=summary, result_json=result, error=error)
            # information gain: did this cycle add ANY new evidence/matter?
            after = _entity_counts(session, ws_id)
            gained = any(after[k] > before[k] for k in after)
            state = dict(ws.state_json or {})
            ctr = state.setdefault("counters", {})
            ctr["cycles"] = ctr.get("cycles", 0) + 1
            state["consecutive_no_gain"] = 0 if gained else state.get("consecutive_no_gain", 0) + 1
            state["last_experiment"] = {"kind": ex.kind, "summary": summary, "ran_s": round(time.time() - started, 2)}
            ws.state_json = state
            ws.research_state = "running" if ws.research_enabled else ws.research_state
            # stagnation stop: the loop is done learning — pause honestly
            limit = int(settings.research.stagnation_limit)
            if limit > 0 and state["consecutive_no_gain"] >= limit:
                # set_research_enabled transitions research_state to "paused"
                # (a bare flag flip leaves state "running" while disabled).
                repos.set_research_enabled(session, ws, False)
                repos.add_timeline(session, ws.id, "director", "warn",
                                   f"Research paused: {state['consecutive_no_gain']} consecutive "
                                   f"cycles without new evidence (stagnation stop). "
                                   f"Re-enable research to resume.",
                                   {"consecutive_no_gain": state["consecutive_no_gain"],
                                    "stagnation_limit": limit})
                summary = (f"{summary} | STAGNATION: paused after "
                           f"{state['consecutive_no_gain']} no-gain cycles")
            repos.release_drain_lease(session, ws, holder)
        import gc  # docking loads gemmi/RDKit structures; free them between cycles
        gc.collect()  # (API lived in a ~4GB cgroup scope and was OOM-killed twice)
        return {"kind": ex.kind, "summary": summary, "error": error or None,
                "elapsed_s": round(time.time() - started, 2)}


def _workspace_enabled(ws_id: str) -> bool:
    """Off-loop helper: True iff the workspace exists and research is enabled.

    Runs in a worker thread (via asyncio.to_thread) so the SQLite read never
    blocks the event loop — a contention window here would stall status and
    control requests for the whole cycle.
    """
    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        return ws is not None and ws.research_enabled


async def research_loop(ws_id: str, stop_event: asyncio.Event,
                        holder: str | None = None):
    """Continuous loop while workspace enabled and stop not set.

    Exits cleanly when the workspace is paused (user pause/stop or stagnation
    stop) so the asyncio task is freed; a later "start" spawns a fresh loop.
    The enabled flag is checked at the TOP of every iteration (off the event
    loop), so a loop spawned after a stop (race) exits before draining anything.
    All blocking work (DB reads, run_cycle) runs in worker threads; the event
    loop only awaits, so status/control requests stay responsive.
    """
    llm = _mk_llm()
    sleep_s = settings.research.loop_sleep_seconds
    loop_holder = holder or f"api-loop:{ws_id}"
    while not stop_event.is_set():
        # Honor the enabled flag every iteration (stop/pause/stagnation set it
        # False). OFF the event loop; checked BEFORE draining so a raced spawn
        # exits cleanly.
        if not await asyncio.to_thread(_workspace_enabled, ws_id):
            log.info("WS %s: research disabled, loop exiting", ws_id)
            return
        try:
            # run_cycle is synchronous and blocks ~27s (docking) to minutes
            # (MD). Offload to a worker thread so the event loop stays
            # responsive (API startup, control endpoints, status).
            r = await asyncio.to_thread(run_cycle, ws_id, llm, holder=loop_holder)
            if r.get("noop"):
                log.info("WS %s: cycle noop (%s)", ws_id, r.get("reason"))
            else:
                log.info("WS %s: cycle done: %s", ws_id, r.get("summary"))
        except Exception:
            log.exception("research cycle error %s", ws_id)
        # Interruptible sleep: a stop frees the task promptly instead of
        # lingering for the full interval.
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_s)
        except asyncio.TimeoutError:
            pass
