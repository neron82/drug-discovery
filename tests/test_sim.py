"""Phase 4 docking tests: ligand prep, pocket derivation, reproducibility record shape."""
import json
import pathlib
import tempfile

import pytest

from drugdiscovery.sim import prepare_ligand, prepare_receptor, pocket_from_ligand

FINASTERIDE = "CC(C)(C)NC(=O)C1CCC2C3CCC4NC(=O)C=CC4(C)C3CCC12C"


def test_prepare_ligand_pdbqt():
    """SMILES -> PDBQT must succeed and produce atom records."""
    out = pathlib.Path(tempfile.mkdtemp()) / "lig.pdbqt"
    prepare_ligand(FINASTERIDE, out, seed=0)
    txt = out.read_text()
    assert "REMARK" in txt
    assert any(l.startswith("ATOM") for l in txt.splitlines()), "no PDBQT atoms"
    # deterministic: same seed -> same output
    out2 = pathlib.Path(tempfile.mkdtemp()) / "lig2.pdbqt"
    prepare_ligand(FINASTERIDE, out2, seed=0)
    assert out.read_text() == out2.read_text()


def test_prepare_ligand_bad_smiles():
    with pytest.raises(ValueError):
        prepare_ligand("garbage!!not-a-smiles", pathlib.Path("/tmp/x.pdbqt"), seed=0)


@pytest.mark.parametrize("smiles", [
    "N=c1nc(N2CCCCC2)cc(N)n1O",   # minoxidil
    "CC(C)(C)NC(=O)C1CCC2C3CCC4NC(=O)C=CC4(C)C3CCC12C",  # finasteride
])
def test_prepare_ligand_multiple(smiles):
    out = pathlib.Path(tempfile.mkdtemp()) / f"m.pdbqt"
    prepare_ligand(smiles, out, seed=7)
    assert out.stat().st_size > 200


def _any_structure():
    from drugdiscovery.db import session_scope, init_db
    from drugdiscovery.models import Structure
    import sqlalchemy as sa
    init_db()
    with session_scope() as s:
        for st in s.execute(sa.select(Structure)).scalars():
            ligs = st.ligands if isinstance(st.ligands, list) else json.loads(st.ligands or "[]")
            if ligs and st.file_path and pathlib.Path(st.file_path).exists():
                return st
    return None


def test_pocket_from_real_ligand():
    """pocket derivation against a real acquired structure (skip if none)."""
    st = _any_structure()
    if st is None or not (st.file_path and pathlib.Path(st.file_path).exists()):
        pytest.skip("no acquired ligand-bearing structure")
    ligands = st.ligands if isinstance(st.ligands, list) else json.loads(st.ligands or "[]")
    if not ligands:
        pytest.skip("no ligands")
    pocket = pocket_from_ligand(st.file_path, ligands[0]["comp_id"])
    assert len(pocket["center"]) == 3
    assert all(isinstance(c, float) for c in pocket["center"])
    assert pocket["size"] > 0


def test_pocket_blind_any_structure():
    """Blind box works WITHOUT ligands (generalisation path); protein-only."""
    from drugdiscovery.sim import pocket_blind
    st = _any_structure()
    if st is None:
        # blind doesn't need ligands — take any structure with a file
        from drugdiscovery.db import session_scope, init_db
        from drugdiscovery.models import Structure
        import sqlalchemy as sa
        init_db()
        with session_scope() as s:
            for x in s.execute(sa.select(Structure)).scalars():
                if x.file_path and pathlib.Path(x.file_path).exists():
                    st = x; break
    if st is None:
        pytest.skip("no acquired structure on disk")
    pocket = pocket_blind(st.file_path)
    assert pocket["source"] == "blind"
    assert len(pocket["center"]) == 3
    assert pocket["size"] > 10.0  # whole protein, not a ligand pocket


def test_prepare_receptor_strips():
    """Receptor prep must yield a non-empty protein PDB from a real cif."""
    st = _any_structure()
    if st is None or not (st.file_path and pathlib.Path(st.file_path).exists()):
        pytest.skip("no acquired structure")
    out = pathlib.Path(tempfile.mkdtemp()) / "rec.pdb"
    prepare_receptor(st.file_path, out)
    txt = out.read_text()
    assert any(l.startswith("ATOM") for l in txt.splitlines())
    assert not any(l.startswith("HETATM") for l in txt.splitlines()), "HETATM leaked"


@pytest.fixture(autouse=True)
def isolated_sim_db(tmp_path, monkeypatch):
    """Keep every simulation test, including structure lookups, off live storage."""
    from drugdiscovery import config, db

    db_path = str(tmp_path / "simulation.db")
    monkeypatch.setattr(config.settings.storage, "db_path", db_path)
    monkeypatch.setattr(config.settings.storage, "data_dir", str(tmp_path))
    monkeypatch.setattr(config.settings.storage, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(config.settings.storage, "log_dir", str(tmp_path / "logs"))
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    connect = db._connect

    def isolated_connect(path):
        if path != db_path:
            raise AssertionError("simulation tests must not connect to a live database")
        return connect(path)

    monkeypatch.setattr(db, "_connect", isolated_connect)
    db.init_db()
    try:
        yield
    finally:
        db.get_engine().dispose()
        db._engine = None
        db._SessionLocal = None


@pytest.fixture
def reproduction_client(isolated_sim_db):
    # api initializes its DB at import; isolated_sim_db must already be active.
    from fastapi.testclient import TestClient
    from drugdiscovery import api

    # Do not enter the application lifespan: it resumes research-loop workers.
    client = TestClient(api.app)
    try:
        yield client
    finally:
        client.close()


def _record_replay_run(seed, kind="docking"):
    from drugdiscovery import repos
    from drugdiscovery.db import session_scope
    from drugdiscovery.models import Compound, SimulationRun, Structure

    with session_scope() as s:
        ws = repos.create_workspace(s, "Replay regression", "test")
        compound = Compound(canonical_name="finasteride", canonical_smiles=FINASTERIDE)
        structure = Structure(workspace_id=ws.id, source="pdb", source_id="TEST")
        s.add_all([compound, structure])
        s.flush()
        params = (
            {"center": [1.25, -3.5, 7.0], "box_size": [24.5, 24.5, 24.5],
             "exhaustiveness": 8, "n_poses": 3}
            if kind == "docking" else {"steps": 120, "temperature": 310.0}
        )
        run = SimulationRun(
            workspace_id=ws.id, kind=kind, structure_id=structure.id,
            compound_id=compound.id if kind == "docking" else None,
            software="autodock-vina" if kind == "docking" else "openmm",
            seed=seed, params_json=params, status="done")
        s.add(run)
        s.flush()
    return run


def test_reproduce_legacy_zero_seed_rejects_without_queueing(reproduction_client):
    from sqlalchemy import select
    from drugdiscovery.db import get_session
    from drugdiscovery.models import Experiment, SimulationRun

    run = _record_replay_run(seed=0)
    response = reproduction_client.post(f"/api/simulations/{run.id}/reproduce")

    assert response.status_code == 409
    with get_session() as s:
        assert s.execute(select(Experiment)).scalars().all() == []
        original = s.get(SimulationRun, run.id)
        assert original.seed == 0
        assert original.params_json == run.params_json


def test_reproduce_recorded_seed_preserves_docking_inputs(reproduction_client):
    from drugdiscovery.db import get_session
    from drugdiscovery.models import Experiment, JobStatus
    from drugdiscovery.sim.engine import DEFAULT_BOX_PADDING

    run = _record_replay_run(seed=173205)
    response = reproduction_client.post(f"/api/simulations/{run.id}/reproduce")

    assert response.status_code == 200
    with get_session() as s:
        queued = s.get(Experiment, response.json()["experiment_id"])
        assert queued.status == JobStatus.QUEUED.value
        assert queued.kind == "docking"
        params = queued.params_json
        assert params["seed"] == run.seed
        assert params["structure_id"] == run.structure_id
        assert params["compound_id"] == run.compound_id
        assert params["compound_smiles"] == FINASTERIDE
        assert params["pocket"]["center"] == run.params_json["center"]
        final_box = [params["pocket"]["size"] + DEFAULT_BOX_PADDING] * 3
        assert final_box == run.params_json["box_size"]
        assert params["exhaustiveness"] == run.params_json["exhaustiveness"]
        assert params["n_poses"] == run.params_json["n_poses"]


def test_reproduce_md_zero_seed_remains_replayable(reproduction_client):
    from drugdiscovery.db import get_session
    from drugdiscovery.models import Experiment, JobStatus

    run = _record_replay_run(seed=0, kind="md")
    response = reproduction_client.post(f"/api/simulations/{run.id}/reproduce")

    assert response.status_code == 200
    with get_session() as s:
        queued = s.get(Experiment, response.json()["experiment_id"])
        assert queued.status == JobStatus.QUEUED.value
        assert queued.kind == "md"
        assert queued.params_json == {
            **run.params_json, "structure_id": run.structure_id, "seed": 0,
        }