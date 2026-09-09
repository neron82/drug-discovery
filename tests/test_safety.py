"""Safety screening (spec 12): deterministic structural alerts."""
import pytest

from drugdiscovery.chem.safety import safety_alerts


def test_clean_molecule_no_alerts():
    r = safety_alerts("c1ccccc1")
    assert r["prediction"] is True
    assert r["pains"] == [] and r["reactive_groups"] == []
    assert r["aggregation_risk"] == "low"
    assert r["liability_score"] == 0.0


def test_reactive_groups_flagged():
    # nitrobenzaldehyde: aldehyde + aryl nitro
    r = safety_alerts("Cc1ccc(N(=O)=O)cc1C=O")
    assert "aldehyde" in r["reactive_groups"]
    assert "aryl_nitro" in r["reactive_groups"]
    assert r["liability_score"] > 0


def test_nitro_not_double_flagged_as_nitroso():
    r = safety_alerts("Nc1ccc([N+](=O)[O-])cc1")
    assert r["reactive_groups"] == ["aryl_nitro"]


def test_unparseable_smiles_is_error_not_crash():
    r = safety_alerts("not_a_smiles((")
    assert r["prediction"] is True and "error" in r