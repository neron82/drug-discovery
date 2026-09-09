"""Structure preparation helpers for docking (Phase 4).

- pocket_from_ligand: derive a docking box from a co-crystallised ligand's atoms
- prepare_receptor: strip a PDB/mmCIF to protein-only PDB ready for meeko
- prepare_ligand: SMILES -> 3D conformer -> PDBQT via meeko
"""
from __future__ import annotations

import logging
from pathlib import Path

import gemmi
import numpy as np

log = logging.getLogger(__name__)

# residues to discard from receptor (waters, common crystallisation artifacts)
_SKIP = {"HOH", "WAT", "H2O"}


def pocket_from_ligand(cif_path, comp_id: str, padding: float = 4.0) -> dict:
    """Centroid + extent of a ligand residue in a PDB/mmCIF structure.

    Returns {"center":[x,y,z], "size":float, "comp_id":str}. size is the largest
    axis span of the ligand; the engine adds `padding` when building the box.
    """
    stru = gemmi.read_structure(str(cif_path))
    xs, ys, zs = [], [], []
    found = False
    for model in stru:
        for chain in model:
            for res in chain:
                if res.name == comp_id:
                    found = True
                    for a in res:
                        xs.append(a.pos.x); ys.append(a.pos.y); zs.append(a.pos.z)
    if not found:
        raise ValueError(f"ligand {comp_id!r} not found in {cif_path}")
    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
    return {
        "center": [float(xs.mean()), float(ys.mean()), float(zs.mean())],
        "size": float(max(np.ptp(xs), np.ptp(ys), np.ptp(zs))),
        "comp_id": comp_id,
    }


def pocket_blind(cif_path, margin: float = 5.0) -> dict:
    """Blind-docking box: bounding box of all polymer atoms + margin.

    Used when a structure has NO co-crystallised ligand (generalisation path —
    any target, including AlphaFold models, becomes dockable). Returns the same
    shape as pocket_from_ligand, tagged with source='blind'.
    """
    stru = gemmi.read_structure(str(cif_path))
    stru.remove_ligands_and_waters()
    xs, ys, zs = [], [], []
    for model in stru:
        for chain in model:
            for res in chain:
                ri = gemmi.find_tabulated_residue(res.name)
                if ri is None or not ri.is_amino_acid():
                    continue  # protein-only receptor convention (matches prepare_receptor)
                for a in res:
                    xs.append(a.pos.x); ys.append(a.pos.y); zs.append(a.pos.z)
    if not xs:
        raise ValueError(f"no polymer atoms found in {cif_path}")
    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
    return {
        "center": [float(xs.mean()), float(ys.mean()), float(zs.mean())],
        "size": float(max(np.ptp(xs), np.ptp(ys), np.ptp(zs)) + margin),
        "comp_id": None, "source": "blind",
    }


def prepare_receptor(cif_path, out_pdb) -> None:
    """Strip non-polymer entities -> protein-only PDB (minimal format)."""
    stru = gemmi.read_structure(str(cif_path))
    stru.remove_ligands_and_waters()
    if out_pdb.exists():
        out_pdb.unlink()
    # write to temp then move so partial writes never leak usable files
    tmp = out_pdb.with_suffix(".pdb.tmp")
    stru.write_pdb(str(tmp), gemmi.PdbWriteOptions(True))
    tmp.replace(out_pdb)
    if out_pdb.stat().st_size == 0:
        raise ValueError(f"no protein atoms in {cif_path}")


def prepare_ligand(smiles: str, out_pdbqt, seed: int = 42) -> None:
    """SMILES -> protonated 3D conformer -> PDBQT (deterministic via seed)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    molh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(molh, randomSeed=seed) != 0:
        # retry with a different seed before giving up
        if AllChem.EmbedMolecule(molh, randomSeed=seed + 1) != 0:
            raise ValueError(f"conformer generation failed for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(molh, maxIters=300)
    molh.SetProp("_Name", "ligand")

    from meeko import MoleculePreparation, PDBQTWriterLegacy
    prep = MoleculePreparation()
    setup = prep.prepare(molh)
    if not setup:
        raise ValueError(f"meeko preparation failed for {smiles!r}")
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setup[0])
    if not ok:
        raise ValueError(f"PDBQT write failed: {err}")
    Path(out_pdbqt).write_text(pdbqt)