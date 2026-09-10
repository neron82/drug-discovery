"""Docking engine adapter (Phase 4).

Wraps AutoDock Vina (Python API) + meeko preparation as a pluggable DockingEngine.
Every run records full reproducibility fields (software/version/params/seed/hashes/
pose artifact/runtime/hardware) per Specification section 20.

A docking score is a COMPUTATIONAL PREDICTION, never biological proof.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from . import (  # noqa: F401
    prepare_receptor,
    pocket_from_ligand,
    prepare_ligand,
)

log = logging.getLogger(__name__)

ENGINE = "autodock-vina"
ENGINE_VERSION = "1.2.7"
DEFAULT_EXHAUSTIVENESS = 24
DEFAULT_N_POSES = 9
# Maximum docking-box edge (Angstrom) before compute_vina_maps is refused.
# A whole-protein blind box on a large structure makes the box edge huge; the
# Vina map grid scales with box volume x atom-type maps, so an unbounded box
# exhausts memory (observed: ~40 GiB RSS, process killed). Pocket-targeted
# boxes (pocket_from_ligand) are ~15-25 A and pass comfortably. 40 A allows
# small/medium whole-protein blind docking while bounding the map grid.
MAX_BOX_EDGE = 40.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hardware() -> str:
    try:
        return f"{platform.machine()} {platform.processor() or 'unknown'}"
    except Exception:
        return "unknown"


class DockingEngine:
    """Adapter seam around AutoDock Vina (Spec section 36)."""

    def __init__(self, bin_dir: str, cpu: int = 8, seed: int = 0):
        self.bin_dir = Path(bin_dir)
        self.cpu = cpu
        self.seed = seed

    # -- one-shot, fully reproducible --------------------------------------
    def dock(self, *, cif_path, pocket, smiles, work_dir, exhaustiveness=24,
             n_poses=9, center=None, box_padding=4.0) -> dict:
        """Perform a docking run and return a reproducibility record dict.

        pocket: dict from pocket_from_ligand() OR an explicit box:
                {"center": [x,y,z], "size": float}
        """
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # HARD BOX GUARD (fail fast, before any prep/allocation): a whole-protein
        # blind box (pocket_blind on a large structure) makes the box edge huge;
        # compute_vina_maps allocates a grid proportional to box volume x
        # atom-type maps, so an unbounded box exhausts memory (observed: ~40 GiB
        # RSS, process killed). Refuse explicitly with provenance. Pocket-targeted
        # boxes (pocket_from_ligand) are ~15-25 A and pass comfortably.
        box_edge = float(pocket.get("size", 15.0)) + box_padding
        if box_edge > MAX_BOX_EDGE:
            raise ValueError(
                f"docking box edge {box_edge:.1f} A exceeds the "
                f"{MAX_BOX_EDGE:.0f} A safety limit (source="
                f"{pocket.get('source', '?')}); a whole-protein blind box this "
                f"large would allocate an unbounded Vina map grid and exhaust "
                f"memory. Use a co-crystal-ligand pocket or a smaller target "
                f"region.")

        # receptor preparation (cached by content hash)
        rec_base = work / "receptor"
        rec_pdbqt = work / "receptor.pdbqt"
        if not rec_pdbqt.exists():
            raw = work / "protein_raw.pdb"
            prepare_receptor(cif_path, raw)
            r = subprocess.run(
                [str(self.bin_dir / "mk_prepare_receptor.py"), "-i", str(raw),
                 "-o", str(rec_base), "-p", "-x", "--default_altloc", "A"],
                capture_output=True, text=True, timeout=600)
            # meeko appends ".pdbqt" to the -o basename
            if not rec_pdbqt.exists() and (work / "receptor.pdbqt.pdbqt").exists():
                rec_pdbqt = work / "receptor.pdbqt.pdbqt"
            if r.returncode != 0 or not rec_pdbqt.exists() or rec_pdbqt.stat().st_size == 0:
                raise RuntimeError(f"receptor preparation failed: {r.stderr[-400:]}")

        # ligand preparation
        lig_pdbqt = work / "ligand.pdbqt"
        prepare_ligand(smiles, lig_pdbqt, seed=self.seed)
        if not lig_pdbqt.exists() or lig_pdbqt.stat().st_size == 0:
            raise RuntimeError("ligand preparation produced no PDBQT")

        if center is None:
            center = pocket["center"]

        box_size = [box_edge] * 3
        from vina import Vina
        v = Vina(sf_name="vina", seed=self.seed, cpu=self.cpu)
        v.set_receptor(str(rec_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(center=[float(c) for c in center], box_size=box_size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        score = float(v.score()[0])   # [0] = total affinity (kcal/mol)

        pose = work / "pose.pdbqt"
        v.write_poses(str(pose), n_poses=n_poses, overwrite=True)
        runtime_s = float(time.time() - t0)

        return {
            "engine": ENGINE,
            "engine_version": ENGINE_VERSION,
            "score": score,                      # kcal/mol, best pose
            "seed": self.seed,
            "exhaustiveness": exhaustiveness,
            "n_poses": n_poses,
            "center": [float(c) for c in center],
            "box_size": box_size,
            "receptor_hash": _sha256(rec_pdbqt),
            "ligand_hash": _sha256(lig_pdbqt),
            "pose_path": str(pose),
            "pose_hash": _sha256(pose),
            "runtime_s": runtime_s,
            "cpu": self.cpu,
            "hardware": _hardware(),
        }