"""Phase 2 tests: cheminformatics engine (RDKit) + structure-aware identity."""
import pytest

from drugdiscovery.chem import (
    canonical_smiles, descriptors, inchi_and_key, identity_hash,
    morgan_fp_hex, tanimoto, validate,
)
from drugdiscovery.chem import engine


def test_canonicalization_and_identity():
    # same molecule written two ways must collapse to one identity
    a = canonical_smiles("CCO")
    b = canonical_smiles("OCC")
    assert a == b == "CCO"
    assert identity_hash("CCO") == identity_hash("OCC")
    k1 = inchi_and_key("CCO")[1]
    k2 = inchi_and_key("OCC")[1]
    assert k1 == k2 and k1  # ASNXHKRYGQEPRM-UHFFFAOYSA-N (ethanol)


def test_invalid_smiles_returns_none():
    assert canonical_smiles("not-a-molecule!!") is None
    assert validate("garbage")["valid"] is False


def test_descriptors_aspirin():
    d = descriptors("CC(=O)Oc1ccccc1C(=O)O")
    assert d["formula"] == "C9H8O4"
    assert d["molmass"] > 170 and d["molmass"] < 190
    assert d["hbd"] >= 1
    assert 3 <= d["hba"] <= 6


def test_tanimoto_same_vs_null():
    fp = morgan_fp_hex("CCO")
    assert fp
    assert tanimoto(fp, fp) == 1.0
    assert tanimoto(fp, "") == 0.0
    # ethanol vs water share O but differ heavily
    assert tanimoto(fp, morgan_fp_hex("O")) < 0.5


def test_similar_small_molecules_closer_than_dissimilar():
    from drugdiscovery.chem import morgan_fp_hex as fp
    phenol = fp("Oc1ccccc1")
    benzyl = fp("Cc1ccccc1")
    benzene = fp("c1ccccc1")
    far = fp("C1CCN(CC1)C(=O)c2ccc3c(c2)OCO3")
    assert tanimoto(phenol, benzyl) > tanimoto(phenol, far)
    assert tanimoto(phenol, benzene) < tanimoto(phenol, benzyl)


def test_svg_render():
    svg = engine.svg_2d("CCO")
    assert "<svg" in svg
    assert "no structure" not in svg
    empty = engine.svg_2d("@@@")
    assert "no structure" in empty


def test_murcko_scaffold():
    assert engine.murcko_scaffold("COc1ccc2cc(ccc2c1)c1ccc(cc1)C(C)(C)C") is not None


def test_sdf_with_conformer_returns_bytes():
    sdf = engine.sdf_with_conformer("CCO", num_confs=1)
    assert sdf and isinstance(sdf, bytes) and sdf.startswith(b"\n")