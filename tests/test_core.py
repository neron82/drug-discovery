"""Unit tests for repository layer + entity normalization + evidence scoring."""
import pytest

from drugdiscovery import repos
from drugdiscovery.db import init_db, get_session
from drugdiscovery.entity import normalize_entity
from drugdiscovery.evidence import score_evidence


_SEED = {"n": 0}
def _fake_stmt(level="probable", ctx="cell_culture", species="human", conf=0.7):
    _SEED["n"] += 1
    class P:
        id = f"p{_SEED['n']}"
        year = 2021
    class S:
        pass
    S.evidence_level = level
    S.model_context = ctx
    S.species = species
    S.confidence = conf
    S.paper = P()
    return S()


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    from drugdiscovery import config as cfg
    monkeypatch.setattr(cfg.settings.storage, "db_path", str(tmp_path / "t.db"))
    monkeypatch.setattr(cfg.settings.storage, "data_dir", str(tmp_path))
    init_db()


def test_entity_normalization():
    assert normalize_entity("DHT", "hormone")["canonical_name"] == "DHT (dihydrotestosterone)"
    assert normalize_entity("SRD5A2", "enzyme")["canonical_name"] == "5-alpha-reductase type 2 (SRD5A2)"
    assert normalize_entity("wnt/beta-catenin signalling", "pathway")["canonical_name"] == "Wnt/beta-catenin signalling"
    # unmapped generic entity keeps name, hinted kind
    r = normalize_entity("SomeNovelProtein", "protein")
    assert r["canonical_name"] == "SomeNovelProtein" and r["kind"] == "protein"


def test_workspace_and_graph_upsert():
    with get_session() as s:
        ws = repos.create_workspace(s, "TestWS", "androgenetic_alopecia")
        n1, _ = repos.upsert_node(s, ws.id, "enzyme", "5-alpha-reductase type 2 (SRD5A2)")
        n2, _ = repos.upsert_node(s, ws.id, "drug", "Finasteride")
        repos.add_edge(s, ws.id, "INHIBITS", n1, n2)
        # duplicate upsert does not duplicate
        n1b, created = repos.upsert_node(s, ws.id, "enzyme", "5-alpha-reductase type 2 (SRD5A2)")
        assert n1b.id == n1.id and created is False
        g = repos.graph_projection(s, ws.id)
        assert len(g["nodes"]) == 2 and len(g["edges"]) == 1


def test_evidence_scoring_transparent():
    sup = [_fake_stmt() for _ in range(3)]
    contra = [_fake_stmt(level="speculative", ctx="in_silico", conf=0.3)]
    prof = score_evidence(sup, contra)
    assert prof["independent_sources"] == 3          # three distinct papers (each fake has own P)
    assert prof["overall_confidence"] > 0.0
    # contradiction lowers confidence vs pure support
    prof2 = score_evidence(sup, [])
    assert prof2["overall_confidence"] >= prof["overall_confidence"]


def test_planning_pipeline_empty_ws():
    from drugdiscovery import planner
    with get_session() as s:
        ws = repos.create_workspace(s, "P0", "x", config={"seed_queries": ["dht pathway"]})
        created = planner.ensure_pipeline(s, ws)
        kinds = {e.kind for e in created}
        assert "literature_search" in kinds
