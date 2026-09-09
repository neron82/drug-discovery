"""Cheminformatics engine (RDKit).

Pure-python helpers for parsing/validating/canonicalizing SMILES, computing
descriptors, fingerprints, similarity, Murcko scaffolds, and rendering 2D SVG.
All external calls go through providers.py; this module never touches the network.
"""
from __future__ import annotations

import hashlib
import re

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# Silence RDKit's noisy stderr logs (only in this module import scope).
RDLogger.DisableLog("rdApp.*")

_FLOAT_KEYS = {"logp", "tpsa"}
_INT_KEYS = {"hbd", "hba", "rotb", "heavy_atoms", "aromatic_rings"}


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Parse and sanitize a SMILES string. Returns None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return mol


def canonical_smiles(smiles: str) -> str | None:
    """Return the canonical (isomeric) SMILES for an input; None if invalid."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def inchi_and_key(smiles: str) -> tuple[str, str]:
    mol = parse_smiles(smiles)
    if mol is None:
        return "", ""
    try:
        inchi = Chem.MolToInchi(mol)
        key = Chem.MolToInchiKey(mol)
        return inchi or "", key or ""
    except Exception:
        return "", ""


def identity_hash(smiles: str) -> str | None:
    """Structure-aware identity hash: canonical SMILES digest (fallback for no InChIKey)."""
    cs = canonical_smiles(smiles)
    if cs is None:
        return None
    return "csha1:" + hashlib.sha1(cs.encode("utf-8")).hexdigest()


def descriptors(smiles: str) -> dict:
    """Compute molecular descriptors. Returns {k: float|int|str}; empty dict on failure."""
    mol = parse_smiles(smiles)
    if mol is None:
        return {}
    try:
        return {
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "molmass": round(Descriptors.MolWt(mol), 3),
            "logp": round(Descriptors.MolLogP(mol), 3),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "charge": Chem.GetFormalCharge(mol),
        }
    except Exception:
        return {}


def sanitize_descriptors(d: dict) -> dict:
    """JSON-friendly descriptor dict (only known scalar keys, rounded)."""
    out = {}
    for k, v in (d or {}).items():
        if k in _INT_KEYS:
            out[k] = int(v)
        elif k in _FLOAT_KEYS:
            out[k] = round(float(v), 3)
        elif k in ("formula", "charge"):
            out[k] = v
    return out


def morgan_fp_hex(smiles: str, radius: int = 2, bits: int = 2048) -> str | None:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    return bytes(fp.ToBitString().encode("utf-8")).hex()


def tanimoto(fp_a_hex: str, fp_b_hex: str) -> float:
    """Tanimoto similarity between two hex-encoded RDKit bit vectors."""
    try:
        a = bytes.fromhex(fp_a_hex).decode("utf-8")
        b = bytes.fromhex(fp_b_hex).decode("utf-8")
    except Exception:
        return 0.0
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    on_both = sum(1 for x, y in zip(a, b) if x == y == "1")
    on_any = sum(1 for x, y in zip(a, b) if x == "1" or y == "1")
    return round(on_both / on_any, 4) if on_any else 0.0


def murcko_scaffold(smiles: str) -> str | None:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc, canonical=True) if sc else None
    except Exception:
        return None


def sdf_with_conformer(smiles: str, num_confs: int = 1) -> bytes | None:
    """Generate 3D conformers and embed them into an SDF block (used by docking/MD later)."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE) != 0:
        return None
    if num_confs > 1:
        AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, randomSeed=0xC0FFEE)
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=300)
    return Chem.MolToMolBlock(mol_h).encode("utf-8")


def svg_2d(smiles: str, width: int = 280, height: int = 220) -> str:
    """Render a 2D structure as SVG markup (safe to serve/embed).

    Uses rdkit.Chem.Draw.MolToSVG (the SVG builder shipped on modern RDKit
    wheels); plain PIL SVG save is usually compiled out.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                '<rect width="100%" height="100%" fill="#0d1117"/>'
                f'<text x="{width//2}" y="{height//2}" fill="#8b949e" text-anchor="middle" '
                f'font-family="monospace" font-size="12">no structure</text></svg>')
    try:
        from rdkit.Chem import Draw as _Draw
        svg = _Draw.MolToSVG(mol, width=width, height=height)
    except Exception:  # noqa: BLE001
        return ""
    if not svg or "<svg" not in svg:
        return ""
    # white background so dark UI keeps bonds readable
    svg = svg.replace("<svg ", f'<svg style="background:#ffffff" ', 1)
    return svg


def validate(smiles: str) -> dict:
    """Full validation + enrichment tuple used by the compound catalogue."""
    cs = canonical_smiles(smiles)
    if cs is None:
        return {"valid": False}
    inchi, key = inchi_and_key(cs)
    d = descriptors(cs)
    return {
        "valid": True,
        "canonical_smiles": cs,
        "inchi": inchi,
        "inchikey": key,
        "identity_hash": identity_hash(cs),
        "fingerprint_hex": morgan_fp_hex(cs),
        "scaffold_smiles": murcko_scaffold(cs),
        **d,
    }