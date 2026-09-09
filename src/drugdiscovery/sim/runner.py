"""Docking job runner: orchestrates a docking experiment and persists the
SimulationRun record with full reproducibility fields."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from sqlalchemy import select

from .. import repos
from ..db import session_scope
from ..models import (  # noqa: F401
    Compound,
    JobStatus,
    Node,
    SimulationRun,
    Structure,
)
from ..config import settings
from . import engine as eng

log = logging.getLogger(__name__)


def _ensure_run_columns(session):
    """Create the simulation_runs table if the schema predates Phase 4."""
    SimulationRun.__table__.create(session.get_bind(), checkfirst=True)


def run_docking(session, ex, params: dict) -> dict:
    """Execute a docking experiment. params: structure_id, compound_id(optional),
    compound_smiles, pocket (optional explicit box), exhaustiveness, n_poses.
    Returns the SimulationRun record id and a short summary."""
    structure_id = params.get("structure_id")
    structure = session.get(Structure, structure_id)
    if structure is None or not structure.file_path or not Path(structure.file_path).exists():
        raise ValueError(f"structure not usable for docking: {structure_id}")

    lig_pdbqt_bytes = None
    compound = None
    compound_id = params.get("compound_id")
    smiles = params.get("compound_smiles")
    if compound_id:
        compound = session.get(Compound, compound_id)
        if compound is not None:
            smiles = compound.canonical_smiles or smiles
    if not smiles:
        raise ValueError("docking requires compound_smiles (or a linked compound)")

    # pocket: explicit box OR derive from the structure's co-crystallised ligand
    pocket = params.get("pocket")
    if not pocket:
        ligands = structure.ligands or []
        comp_ids = [l.get("comp_id") for l in ligands if isinstance(l, dict)]
        comp_ids = [c for c in comp_ids if c]
        if not comp_ids:
            raise ValueError(f"no ligand pocket available for structure {structure.source_id}")
        from . import pocket_from_ligand
        pocket = pocket_from_ligand(structure.file_path, comp_ids[0])

    # artifact dir
    artifact_root = Path(settings.storage.artifact_dir) / "docking"
    run_dir = artifact_root / f"{structure.source_id}_{ex.id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dock = eng.DockingEngine(
        bin_dir=settings.storage.venv_bin, cpu=int(params.get("cpu", 8)),
        seed=int(params.get("seed", 0)),
    )
    record = dock.dock(
        cif_path=structure.file_path, pocket=pocket, smiles=smiles,
        work_dir=str(run_dir),
        exhaustiveness=int(params.get("exhaustiveness", 24)),
        n_poses=int(params.get("n_poses", 9)),
    )

    # persist SimulationRun
    run = SimulationRun(
        workspace_id=ex.workspace_id, kind="docking", experiment_id=ex.id,
        structure_id=structure_id, compound_id=compound_id,
        software=record["engine"], software_version=record["engine_version"],
        params_json={
            "exhaustiveness": record["exhaustiveness"],
            "n_poses": record["n_poses"],
            "center": record["center"], "box_size": record["box_size"],
            "cpu": record["cpu"],
        },
        seed=record["seed"],
        receptor_hash=record["receptor_hash"], ligand_hash=record["ligand_hash"],
        score=record["score"],
        rmsd_lb=record.get("rmsd_lb"),
        pose_path=record["pose_path"], runtime_s=record["runtime_s"],
        hardware=record["hardware"],
        assumptions="Ligand protons added at pH 7.4 by meeko; receptor prepared "
                    "with Gasteiger charges via meeko (default_altloc=A, bad residues "
                    "deleted). Docking is a computational prediction, not proof of "
                    "biological activity.",
    )
    session.add(run)
    session.flush()

    summary = (f"docked {compound.canonical_name if compound else smiles} "
               f"vs {structure.source_id}: score {record['score']:.2f} kcal/mol "
               f"({record['runtime_s']:.1f}s, seed {record['seed']})")
    # record on the experiment
    ex.result_summary = summary
    ex.result_json = {
        "score": record["score"], "simulation_run_id": run.id,
        "pose_path": record["pose_path"], "structure_id": structure_id,
        "box_center": record["center"], "exhaustiveness": record["exhaustiveness"],
        "n_poses": record["n_poses"],
    }
    session.flush()

    return {"simulation_run_id": run.id, "score": record["score"],
            "summary": summary, "pose_path": record["pose_path"]}


def list_runs(session, workspace_id: str):
    from ..models import SimulationRun
    _ensure_run_columns(session)
    return list(session.execute(
        select(SimulationRun).where(SimulationRun.workspace_id == workspace_id)
        .order_by(SimulationRun.created_at.desc())
    ).scalars())