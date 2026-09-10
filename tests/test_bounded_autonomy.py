"""Bounded-autonomy tests: the guarantees that make overnight runs converge.

Covers the deterministic, network-free pieces:
  * planner skips oversized structures before CPU MD (explicit failure, not a crash)
  * planner does not duplicate jobs across repeated planning passes
  * per-(kind,params) retry ceiling stops failure loops
  * single-drain lease refuses concurrent drainers (cross-process atomic)
  * stagnation stop pauses a workspace that stops producing new evidence
  * explicit stop cancels queued (not running) work with provenance
  * queue metrics attribute created/completed/failed/cancelled + discovery/retry

The real OpenMM/Vina/LLM integration is exercised as a live smoke run, not here.
"""
import pytest
from sqlalchemy import select

from drugdiscovery import planner, repos, research
from drugdiscovery.db import get_session, init_db, session_scope
from drugdiscovery.models import Experiment, JobStatus, SimulationRun, Structure


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    from drugdiscovery import config as cfg
    monkeypatch.setattr(cfg.settings.storage, "db_path", str(tmp_path / "t.db"))
    monkeypatch.setattr(cfg.settings.storage, "data_dir", str(tmp_path))
    import drugdiscovery.db as _db
    monkeypatch.setattr(_db, "_engine", None)
    monkeypatch.setattr(_db, "_SessionLocal", None)
    init_db()
    yield
    # drop the global engine so the next test (new tmp_path) rebuilds cleanly
    import drugdiscovery.db as _db2
    _db2._engine = None
    _db2._SessionLocal = None


def _mk_ws(s, name="BA WS", disease="mpb", config=None):
    return repos.create_workspace(s, name, disease, config=config or {})


def _mk_experiment(s, ws_id, kind="question_generation", params=None, priority=0.1):
    return repos.create_experiment(
        s, ws_id, kind, question=f"{kind} q", reason="test",
        expected_information_gain=0.1, scientific_importance=0.1,
        confidence_impact=0.1, computational_cost=0.1, priority=priority,
        params=params or {}, worker="director")


def _kinds(s, ws_id):
    rows = s.execute(select(Experiment.kind)
                     .where(Experiment.workspace_id == ws_id)).all()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Planner: oversized-MD skip + queue non-duplication
# ---------------------------------------------------------------------------
def _large_pdb(tmp_path, n_lines=20_001):
    p = tmp_path / "large.pdb"
    with p.open("w") as f:
        for i in range(n_lines):
            f.write(f"ATOM  {i:5d}  N   ALA A   1      11.004  13.524  10.204  1.00  0.00           N\n")
    return str(p)


def _mk_docked_structure(s, ws, file_path, score=-6.0):
    """Target + compound + structure + a good docking run (<= -5 kcal/mol)."""
    tgt, _ = repos.upsert_node(s, ws.id, "enzyme", "SRD5A2", confidence=0.8,
                               properties={"structural": {"acquired": True}})
    st = Structure(id="st_big", workspace_id=ws.id, source="pdb", source_id="BIG1",
                   protein_node_id=tgt.id, method="x-ray", resolution=2.1,
                   file_path=file_path)
    s.add(st)
    s.flush()  # persist the structure before the FK-referencing run
    s.add(SimulationRun(id="sr_big", workspace_id=ws.id, kind="docking",
                        structure_id=st.id, software="autodock-vina",
                        score=score, status="done"))
    s.commit()
    return ws, st


def test_planner_skips_oversized_md_and_records_reason(tmp_path):
    with get_session() as s:
        ws = _mk_ws(s)
        pdb = _large_pdb(tmp_path)
        _mk_docked_structure(s, ws, pdb, score=-6.0)
        planner.ensure_pipeline(s, ws)
        s.commit()
        kinds = _kinds(s, ws.id)
        assert "molecular_dynamics" not in kinds, \
            "oversized structure must not be queued for CPU MD"
        skipped = (ws.state_json or {}).get("md_skipped", {})
        assert "st_big" in skipped, "skip reason must be persisted as provenance"
        assert skipped["st_big"]["atom_count"] > 20_000


def test_planner_queues_md_for_small_structure(tmp_path):
    with get_session() as s:
        ws = _mk_ws(s)
        # small file: well under the 20k budget
        small = tmp_path / "small.pdb"
        small.write_text("ATOM      1  N   ALA A   1      11.004  13.524  10.204  1.00  0.00           N\n")
        _mk_docked_structure(s, ws, str(small), score=-6.0)
        planner.ensure_pipeline(s, ws)
        s.commit()
        assert "molecular_dynamics" in _kinds(s, ws.id), \
            "a good docking on a small structure should schedule MD"


def test_planner_does_not_duplicate_across_passes():
    with get_session() as s:
        ws = _mk_ws(s)
        # a fetched paper triggers a paper_extraction job
        from drugdiscovery.models import Paper
        p = Paper(id="p1", workspace_id=ws.id, pmid="123", title="T", status="fetched")
        s.add(p)
        s.commit()
        planner.ensure_pipeline(s, ws)
        s.commit()
        first = s.execute(select(Experiment).where(Experiment.workspace_id == ws.id)).scalars().all()
        n_first = len(first)
        assert n_first >= 1
        # a second planning pass must not add duplicates
        planner.ensure_pipeline(s, ws)
        s.commit()
        second = s.execute(select(Experiment).where(Experiment.workspace_id == ws.id)).scalars().all()
        assert len(second) == n_first, "repeated planning passes must not duplicate jobs"


# ---------------------------------------------------------------------------
# Retry ceiling
# ---------------------------------------------------------------------------
def _fail_latest(s, ws_id, kind, params):
    """Mark the most recent queued experiment for (kind, params) as FAILED."""
    ex = s.execute(select(Experiment).where(
        Experiment.workspace_id == ws_id, Experiment.kind == kind,
        Experiment.status == JobStatus.QUEUED.value)).scalars().first()
    assert ex is not None, "expected a queued experiment to fail"
    repos.mark_experiment(s, ex, JobStatus.FAILED.value, error="boom")
    s.commit()
    return ex


def test_retry_ceiling_stops_failure_loop():
    from drugdiscovery.models import Paper
    with get_session() as s:
        ws = _mk_ws(s)
        p = Paper(id="p1", workspace_id=ws.id, pmid="123", title="T", status="fetched")
        s.add(p)
        s.commit()
        wid, pid = ws.id, p.id

    # 0 failures -> planner schedules the extraction (discovery)
    with session_scope() as s:
        ws = repos.get_workspace(s, wid)
        planner.ensure_pipeline(s, ws)
    with get_session() as s:
        n = s.execute(select(Experiment).where(
            Experiment.workspace_id == wid, Experiment.kind == "paper_extraction")).scalars().all()
        assert len(n) == 1

    # 1 failure (< max_retries=2) -> planner retries once
    with session_scope() as s:
        _fail_latest(s, wid, "paper_extraction", {"paper_id": pid})
        ws = repos.get_workspace(s, wid)
        planner.ensure_pipeline(s, ws)
    with get_session() as s:
        n = s.execute(select(Experiment).where(
            Experiment.workspace_id == wid, Experiment.kind == "paper_extraction")).scalars().all()
        assert len(n) == 2, "one retry is allowed under the ceiling"

    # 2 failures (>= max_retries=2) -> planner stops re-scheduling
    with session_scope() as s:
        _fail_latest(s, wid, "paper_extraction", {"paper_id": pid})
        ws = repos.get_workspace(s, wid)
        planner.ensure_pipeline(s, ws)
    with get_session() as s:
        n = s.execute(select(Experiment).where(
            Experiment.workspace_id == wid, Experiment.kind == "paper_extraction")).scalars().all()
        assert len(n) == 2, "retry ceiling must stop the failure loop"


# ---------------------------------------------------------------------------
# Single-drain lease
# ---------------------------------------------------------------------------
def test_acquire_release_and_expiry():
    with session_scope() as s:
        ws = _mk_ws(s)
        wid = ws.id
    with session_scope() as s:
        ws = repos.get_workspace(s, wid)
        assert repos.acquire_drain_lease(s, ws, "A", 300) is True
        assert repos.drain_lease_holder(s, wid) == "A"
        # a different holder cannot take a live lease
        assert repos.acquire_drain_lease(s, ws, "B", 300) is False
        assert repos.drain_lease_holder(s, wid) == "A"
        # same-holder renewal is allowed
        assert repos.acquire_drain_lease(s, ws, "A", 300) is True
        # release frees it
        repos.release_drain_lease(s, ws, "A")
        assert repos.drain_lease_holder(s, wid) is None
        assert repos.acquire_drain_lease(s, ws, "B", 300) is True


def test_expired_lease_is_taken_over():
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import text
    with session_scope() as s:
        ws = _mk_ws(s)
        wid = ws.id
    with session_scope() as s:
        ws = repos.get_workspace(s, wid)
        repos.acquire_drain_lease(s, ws, "dead-holder", 30)
    # simulate the holder dying: backdate the expiry into the past
    with session_scope() as s:
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        s.execute(text("UPDATE drain_leases SET expires_at = :e WHERE workspace_id = :w"),
                  {"e": past, "w": wid})
        assert repos.drain_lease_holder(s, wid) is None, "expired lease must read as free"
        ws = repos.get_workspace(s, wid)
        assert repos.acquire_drain_lease(s, ws, "new-holder", 300) is True
        assert repos.drain_lease_holder(s, wid) == "new-holder"


def test_run_cycle_refuses_when_lease_held_by_other():
    with get_session() as s:
        ws = _mk_ws(s)
        _mk_experiment(s, ws.id)
        s.commit()
        wid = ws.id
    # holder A takes the lease in a committed transaction
    with session_scope() as s:
        ws = repos.get_workspace(s, wid)
        assert repos.acquire_drain_lease(s, ws, "A", 300)
    # a cycle as holder B must refuse, leaving the experiment queued
    r = research.run_cycle(wid, force=True, holder="B")
    assert r.get("noop") is True
    assert "drain in progress" in r.get("reason", "")
    with get_session() as s:
        ex = s.execute(select(Experiment).where(Experiment.workspace_id == wid)
                       ).scalars().first()
        assert ex.status == JobStatus.QUEUED.value, "refused cycle must not execute work"


# ---------------------------------------------------------------------------
# Stagnation stop
# ---------------------------------------------------------------------------
def test_stagnation_pauses_workspace(monkeypatch):
    from drugdiscovery import config as cfg
    monkeypatch.setattr(cfg.settings.research, "stagnation_limit", 3)

    def fake_execute(session, ws, ex, llm):
        # replenish the queue so the loop never parks, but create NO entities
        # (no information gain) — this is the stagnation condition.
        _mk_experiment(session, ws.id, priority=0.05)
        return "noop", {}

    monkeypatch.setattr(research, "execute_experiment", fake_execute)
    with get_session() as s:
        ws = _mk_ws(s)
        repos.set_research_enabled(s, ws, True)
        _mk_experiment(s, ws.id)
        s.commit()
        wid = ws.id

    for _ in range(3):
        r = research.run_cycle(wid, force=True, holder="stagnation-test")
        assert not r.get("noop"), f"cycle should have run, got {r}"

    with get_session() as s:
        ws = repos.get_workspace(s, wid)
        assert ws.research_enabled is False, \
            "workspace must pause after consecutive no-gain cycles"
        assert (ws.state_json or {}).get("consecutive_no_gain", 0) >= 3


def test_gain_resets_stagnation_counter(monkeypatch):
    from drugdiscovery import config as cfg
    monkeypatch.setattr(cfg.settings.research, "stagnation_limit", 3)
    calls = {"n": 0}

    def fake_execute(session, ws, ex, llm):
        calls["n"] += 1
        _mk_experiment(session, ws.id, priority=0.05)
        if calls["n"] == 2:
            # second cycle produces a NEW entity (information gain)
            from drugdiscovery.models import Paper
            session.add(Paper(id=f"gain{calls['n']}", workspace_id=ws.id,
                              pmid=f"9{calls['n']}", title="G", status="fetched"))
        return "noop", {}

    monkeypatch.setattr(research, "execute_experiment", fake_execute)
    with get_session() as s:
        ws = _mk_ws(s)
        repos.set_research_enabled(s, ws, True)
        _mk_experiment(s, ws.id)
        s.commit()
        wid = ws.id

    # 3 cycles, but the 2nd produces a gain -> counter resets, no pause
    for _ in range(3):
        research.run_cycle(wid, force=True, holder="gain-test")
    with get_session() as s:
        ws = repos.get_workspace(s, wid)
        assert ws.research_enabled is True, \
            "an information-gain cycle must reset the stagnation counter"


# ---------------------------------------------------------------------------
# Stop cancels queued (not running)
# ---------------------------------------------------------------------------
def test_stop_cancels_queued_but_not_running():
    with get_session() as s:
        ws = _mk_ws(s)
        e1 = _mk_experiment(s, ws.id, priority=0.1)
        e2 = _mk_experiment(s, ws.id, params={"x": 2}, priority=0.2)
        repos.mark_experiment(s, e2, JobStatus.RUNNING.value, worker="sim")
        s.commit()
        wid, eid1, eid2 = ws.id, e1.id, e2.id
    with session_scope() as s:
        n = repos.cancel_queued_experiments(s, wid, "explicit stop")
        assert n == 1, "only the queued experiment is cancelled"
    with get_session() as s:
        e1 = s.get(Experiment, eid1)
        e2 = s.get(Experiment, eid2)
        assert e1.status == JobStatus.CANCELLED.value
        assert "cancellation" in (e1.result_json or {})
        assert e1.result_json["cancellation"]["reason"] == "explicit stop"
        assert e1.finished_at is not None, "cancellation must be timestamped"
        assert e2.status == JobStatus.RUNNING.value, "running work is left to finish"


# ---------------------------------------------------------------------------
# Queue metrics
# ---------------------------------------------------------------------------
def test_queue_metrics_attribution():
    with get_session() as s:
        ws = _mk_ws(s)
        wid = ws.id
        # a FAILED docking (identity s1)
        f = _mk_experiment(s, wid, kind="docking",
                           params={"structure_id": "s1", "compound_id": "c1"})
        repos.mark_experiment(s, f, JobStatus.FAILED.value, error="boom")
        # a DONE docking (identity s3)
        d = _mk_experiment(s, wid, kind="docking",
                           params={"structure_id": "s3", "compound_id": "c3"})
        repos.mark_experiment(s, d, JobStatus.DONE.value, result_summary="ok")
        # a QUEUED docking matching the failed identity s1 -> RETRY
        _mk_experiment(s, wid, kind="docking",
                       params={"structure_id": "s1", "compound_id": "c1"})
        # a QUEUED docking with a fresh identity s2 -> DISCOVERY
        _mk_experiment(s, wid, kind="docking",
                       params={"structure_id": "s2", "compound_id": "c2"})
        s.commit()
    with get_session() as s:
        m = repos.queue_metrics(s, wid)
    assert m["completed"] == 1
    assert m["failed"] == 1
    assert m["pending"] == 2
    assert m["pending_discovery"] == 1, "fresh identity is new discovery"
    assert m["pending_retry"] == 1, "identity with a prior failure is a retry"
    assert m["lease_holder"] is None
    assert m["pending_by_kind"] == {"docking": 2}
def test_stop_then_immediate_start_respawns(monkeypatch):
    """Lifecycle must be deterministic across stop->immediate-start.

    The fake loop finishes an "in-flight cycle" (100ms) BEFORE checking the
    stop-event, so after a stop there is a real window where the task is
    NON-DONE but exiting (stop-event set) -- the branch that must trigger a
    respawn. Verifies:
      * start while a loop is actively running -> no new task
      * stop (task non-done, exiting) -> immediate start -> FRESH task
      * two rapid starts -> exactly one task (no double-spawn)
      * each spawned task gets a unique drain-lease holder
    """
    import asyncio
    from drugdiscovery import api

    holders = []

    async def fake_loop(ws_id, stop_event, holder=None):
        holders.append(holder)
        while True:
            await asyncio.sleep(0.1)  # in-flight cycle; non-done while running
            if stop_event.is_set():
                return  # exit only after the in-flight cycle completes

    monkeypatch.setattr("drugdiscovery.research.research_loop", fake_loop)

    async def scenario():
        api._running_tasks.clear()
        api._stop_events.clear()
        # 1. start -> spawns T1 (enters its 100ms in-flight cycle)
        api._start_on_loop("wsA")
        await asyncio.sleep(0.03)
        t1 = api._running_tasks["wsA"]
        assert not t1.done()
        # 2. start again while active (stop-event clear) -> same task, no respawn
        api._start_on_loop("wsA")
        await asyncio.sleep(0.03)
        assert api._running_tasks["wsA"] is t1, "active loop must not respawn"
        # 3. stop -> sets T1's stop-event; T1 is STILL non-done (in-flight)
        api._stop_loop("wsA")
        await asyncio.sleep(0.02)
        assert not t1.done(), "T1 must still be finishing its in-flight cycle"
        # 4. IMMEDIATE start while T1 is non-done-but-exiting -> FRESH task
        api._start_on_loop("wsA")
        await asyncio.sleep(0.03)
        t2 = api._running_tasks["wsA"]
        assert t2 is not t1, "stop->start must replace the exiting (non-done) task"
        assert not t2.done()
        # 5. two rapid starts -> still exactly one task (no double-spawn)
        api._start_on_loop("wsA")
        api._start_on_loop("wsA")
        await asyncio.sleep(0.03)
        assert api._running_tasks["wsA"] is t2, "rapid starts must not double-spawn"
        # cleanup: stop the fresh task; it exits after its in-flight cycle
        api._stop_loop("wsA")
        await asyncio.sleep(0.2)
        assert t2.done()

    asyncio.run(scenario())
    # exactly two tasks were ever spawned (T1 and T2), with unique holders
    assert len(holders) == 2, f"expected 2 spawned tasks, got {len(holders)}"
    assert holders[0] != holders[1], "holders must be unique per task"
    assert all(h.startswith("api-loop:wsA:") for h in holders)
def test_docking_box_guard_refuses_oversized_blind_box(tmp_path):
    """A whole-protein blind box (edge > MAX_BOX_EDGE) must be refused with
    explicit provenance BEFORE any Vina map allocation — this is the guard
    against the ~40 GiB RSS runaway that forced the API to be killed.

    * oversized blind box (size 80 A -> edge 84 A > 40 A) -> ValueError
    * normal pocket-targeted box (size 18 A -> edge 22 A < 40 A) -> must NOT
      trip the guard (it proceeds to receptor prep, which fails for a fake CIF
      with a different, expected error)
    """
    from drugdiscovery.sim.engine import DockingEngine, MAX_BOX_EDGE
    assert MAX_BOX_EDGE == 40.0
    eng = DockingEngine(bin_dir=str(tmp_path / "bin"), cpu=2, seed=0)
    fake_cif = str(tmp_path / "fake.cif")

    # 1. oversized blind box -> refused with the box-guard error
    big = {"center": [0.0, 0.0, 0.0], "size": 80.0, "source": "blind"}
    with pytest.raises(ValueError, match="exceeds the"):
        eng.dock(cif_path=fake_cif, pocket=big, smiles="CCO",
                 work_dir=str(tmp_path / "work_big"))

    # 2. normal pocket-targeted box -> must NOT trip the box guard
    ok = {"center": [0.0, 0.0, 0.0], "size": 18.0, "source": "ligand"}
    try:
        eng.dock(cif_path=fake_cif, pocket=ok, smiles="CCO",
                 work_dir=str(tmp_path / "work_ok"))
    except ValueError as e:
        assert "exceeds the" not in str(e), "normal box must not trip the guard"
    except Exception:
        pass  # receptor/ligand prep failure is expected for a fake CIF