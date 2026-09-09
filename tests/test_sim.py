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