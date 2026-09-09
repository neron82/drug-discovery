"""Phase 6 tests: molecular-dynamics planner gating + analysis maths.

The real OpenMM integration is exercised as a live smoke run (scripts, not in
this unit suite) — here we verify the deterministic, network-free pieces:
the planner only queues MD for completed dockings ≤ -5 kcal/mol (spec §11:
expensive, gated secondary validation) and the analysis maths are sane.
"""
import numpy as np
import pytest

from drugdiscovery import repos
from drugdiscovery.db import get_session, init_db
from drugdiscovery.models import SimulationRun, Structure
from drugdiscovery.sim import md as md_eng
from drugdiscovery import planner


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    from drugdiscovery import config as cfg
    monkeypatch.setattr(cfg.settings.storage, "db_path", str(tmp_path / "t.db"))
    monkeypatch.setattr(cfg.settings.storage, "data_dir", str(tmp_path))
    # give each test its own fresh engine/session (global engine caches the
    # FIRST db_path; without this, tests across modules share one database
    # and collide on UNIQUE columns).
    import drugdiscovery.db as _db
    monkeypatch.setattr(_db, "_engine", None)
    monkeypatch.setattr(_db, "_SessionLocal", None)
    init_db()


def _mk_docked_ws(s, score=-6.28):
    ws = repos.create_workspace(s, "MD WS", "mpb")
    tgt, _ = repos.upsert_node(s, ws.id, "enzyme", "SRD5A2", confidence=0.8)
    from drugdiscovery import chem
    rec = chem.validate("CC(C)(C)NC(=O)C1CCC2C3CCC4NC(=O)C=CC4(C)C3CCC12C")
    c, _ = repos.upsert_compound(
        s, name="Finasteride", canonical_smiles=rec["canonical_smiles"],
        inchikey=rec["inchikey"],
        properties={"molmass": 361, "logp": 3.3, "hbd": 1, "hba": 3},
        sources={"pubchem": {"cid": "33571"}})
    cnode, _ = repos.upsert_node(s, ws.id, "drug", "Finasteride",
                                 properties={"compound_id": c.id})
    repos.add_edge(s, ws.id, "INHIBITS", cnode, tgt, evidence_level="established",
                   confidence=0.9, provenance={"source": "test"})
    s.commit()
    st = Structure(id="st1", workspace_id=ws.id, source="pdb", source_id="7BW1",
                   protein_node_id=tgt.id, method="x-ray", resolution=2.1)
    s.add(st); s.flush()
    s.add(SimulationRun(id="sr1", workspace_id=ws.id, kind="docking",
                        compound_id=c.id, structure_id=st.id,
                        software="autodock-vina", score=score, status="done"))
    s.commit()
    return ws, st


def _kinds(s, ws_id):
    from sqlalchemy import select
    rows = s.execute(select(planner.Experiment.kind)
                     .where(planner.Experiment.workspace_id == ws_id)).all()
    return {r[0] for r in rows}


def test_md_queued_for_good_docking():
    with get_session() as s:
        ws, st = _mk_docked_ws(s, score=-6.28)
        planner.ensure_pipeline(s, ws)
        kinds = _kinds(s, ws.id)
        assert "molecular_dynamics" in kinds
        from sqlalchemy import select
        rows = s.query(planner.Experiment).filter_by(
            workspace_id=ws.id, kind="molecular_dynamics").all()
        assert rows, "planner must queue MD for a ≤ -5 kcal/mol docking"
        assert rows[0].params_json.get("structure_id") == st.id


def test_md_not_queued_for_weak_docking():
    with get_session() as s:
        ws, _ = _mk_docked_ws(s, score=-3.0)
        planner.ensure_pipeline(s, ws)
        rows = s.query(planner.Experiment).filter_by(
            workspace_id=ws.id, kind="molecular_dynamics").all()
        assert not rows, "MD is expensive — must not be scheduled for weak dockings"


def test_md_analysis_maths_sane():
    rng = np.random.default_rng(7)
    T, N = 30, 50
    # nearly-static protein: small motion around a stable fold
    frames = np.zeros((T, N, 3))
    base = np.tile(np.linspace(0, 2, N).reshape(-1, 1), (1, 3))
    for t in range(T):
        frames[t] = base + rng.normal(0, 0.03, (N, 3))
    ana = md_eng._analyse(frames, base, None, 100)
    assert 0.0 <= ana["backbone_rmsd_nm"] <= 0.2
    assert ana["rmsf_mean_nm"] < 0.06
    assert ana["radius_of_gyration_nm"] > 0
    assert ana["steps"] == 100 and ana["frames"] == T
    # wildly drifting system must show larger drift than the stable one
    frames2 = np.zeros((T, N, 3))
    for t in range(T):
        frames2[t] = base + t * 0.05 + rng.normal(0, 0.03, (N, 3))
    ana2 = md_eng._analyse(frames2, base, None, 100)
    assert ana2["final_drift_nm"] > ana["final_drift_nm"]