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

    # pocket: explicit box OR co-crystallised ligand OR blind whole-protein box
    pocket = params.get("pocket")
    if not pocket:
        ligands = structure.ligands or []
        comp_ids = [l.get("comp_id") for l in ligands if isinstance(l, dict)]
        comp_ids = [c for c in comp_ids if c]
        if comp_ids:
            from . import pocket_from_ligand
            pocket = pocket_from_ligand(structure.file_path, comp_ids[0])
        else:
            # generalisation path: no co-crystal ligand -> blind box over the
            # whole polymer (labelled honestly in assumptions below)
            from . import pocket_blind
            pocket = pocket_blind(structure.file_path)

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
        assumptions=("Ligand protons added at pH 7.4 by meeko; receptor prepared with "
                    "Gasteiger charges via meeko (default_altloc=A, bad residues "
                    "deleted). Docking is a computational prediction, not proof of "
                    "biological activity."
                    + (" Box was BLIND (whole-polymer bounding box, no co-crystal "
                       "ligand): scores are site-agnostic and weaker evidence than "
                       "pocket-targeted docking."
                       if pocket.get("source") == "blind" else "")),
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


def run_md(session, ex, params: dict) -> dict:
    """Execute a protein-stability MD experiment. params: structure_id,
    steps, temperature, platform(optional), cpu_threads, seed.
    Returns a summary + SimulationRun record (kind='md')."""
    structure_id = params.get("structure_id")
    structure = session.get(Structure, structure_id)
    if structure is None or not structure.file_path or not Path(structure.file_path).exists():
        raise ValueError(f"structure not usable for MD: {structure_id}")

    run_dir = (Path(settings.storage.artifact_dir) / "md" /
               f"{structure.source_id}_{ex.id[:8]}")
    run_dir.mkdir(parents=True, exist_ok=True)

    from . import prepare_receptor
    protein_pdb = run_dir / "protein.pdb"
    prepare_receptor(structure.file_path, protein_pdb)

    from . import md as md_eng
    rec = md_eng.run_md(
        protein_path=protein_pdb, work_dir=str(run_dir),
        steps=int(params.get("steps", 1000)),
        temperature=float(params.get("temperature", 310.0)),
        platform=params.get("platform"),
        cpu_threads=int(params.get("cpu_threads", 8)),
        seed=int(params.get("seed", 7)),
    )

    run = SimulationRun(
        workspace_id=ex.workspace_id, kind="md", experiment_id=ex.id,
        structure_id=structure_id, compound_id=None,
        software=rec["engine"], software_version=rec.get("engine_version"),
        params_json={
            "steps": rec["steps_planned"], "temperature_K": rec["temperature_K"],
            "platform": rec["platform"], "box_padding_nm": rec["box_padding_nm"],
            "force_field": rec["force_field"], "cpu_threads": params.get("cpu_threads", 8),
        },
        seed=rec["seed"],
        receptor_hash=rec["protein_hash"], ligand_hash=None,
        score=rec.get("backbone_rmsd_nm"),
        pose_path=rec.get("trajectory_path"), runtime_s=rec["runtime_s"],
        hardware=rec["hardware"], assumptions=rec.get("limitations", ""),
    )
    session.add(run); session.flush()

    summary = (f"MD {structure.source_id}: RMSD {rec['backbone_rmsd_nm']} nm "
               f"drift {rec['final_drift_nm']} nm, RMSF p95 {rec['rmsf_p95_nm']} nm "
               f"({rec['runtime_s']:.0f}s, {rec['platform']})")
    ex.result_summary = summary
    ex.result_json = {"simulation_run_id": run.id, "score": rec.get("backbone_rmsd_nm"),
                      "analysis": rec.get("analysis"), "structure_id": structure_id}
    session.flush()
    return {"simulation_run_id": run.id, "summary": summary,
            "score": rec.get("backbone_rmsd_nm")}


def list_runs(session, workspace_id: str):
    from ..models import SimulationRun
    _ensure_run_columns(session)
    return list(session.execute(
        select(SimulationRun).where(SimulationRun.workspace_id == workspace_id)
        .order_by(SimulationRun.created_at.desc())
    ).scalars())