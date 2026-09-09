"""Phase 3 structural biology tests: gene resolution, parsing, artifact dedup."""
import hashlib
import pytest

from drugdiscovery.struct.providers import sha256, alphafold, pdb_cross_refs
from drugdiscovery.struct.acquisition import gene_symbol
from drugdiscovery.models import Node


def test_gene_symbol_from_paren():
    n = Node(kind="protein", canonical_name="Androgen receptor (AR)")
    assert gene_symbol(n) == "AR"


def test_gene_symbol_from_synonym():
    n = Node(kind="enzyme", canonical_name="5-alpha-reductase", synonyms=["SRD5A2 (SRD5A2)"])
    assert gene_symbol(n) == "SRD5A2"


def test_gene_symbol_fallback_first_word():
    n = Node(kind="protein", canonical_name="DKK3")
    assert gene_symbol(n) == "DKK3"


def test_sha256_stable():
    data = b"dummy cif bytes"
    assert sha256(data) == hashlib.sha256(data).hexdigest()


def test_alphafold_parses_list(monkeypatch):
    fake = [{"modelEntityId": "AF-P10275-F1", "cifUrl": "https://x/p.cif",
             "fractionPlddtConfident": 0.92, "organismScientificName": "Homo sapiens"}]
    monkeypatch.setattr("drugdiscovery.struct.providers._get", lambda url, **kw: fake)
    rec = alphafold("P10275")
    assert rec["source_id"] == "AF-P10275-F1"
    assert rec["confidence"] == 0.92
    assert rec["method"] == "predicted"


def test_pdb_cross_refs_parse(monkeypatch):
    fake = {"uniProtKBCrossReferences": [
        {"database": "PDB", "id": "1E3G", "properties": [
            {"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "2.40 A"}]},
        {"database": "PubMed", "id": "123", "properties": []},
        {"database": "PDB", "id": "7XYZ", "properties": [{"key": "Method", "value": "NMR"}]},
    ]}
    monkeypatch.setattr("drugdiscovery.struct.providers._get", lambda url, **kw: fake)
    refs = pdb_cross_refs("P10275")
    assert len(refs) == 2
    assert refs[0]["pdb_id"] == "1E3G" and refs[0]["resolution"] == 2.40
    assert refs[1]["resolution"] is None