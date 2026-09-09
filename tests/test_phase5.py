"""Phase 5 tests: skeptic critique (stub LLM) + multi-factor candidate ranking."""
import pytest

from drugdiscovery import repos
from drugdiscovery.db import init_db, get_session
from drugdiscovery import critique as crit
from drugdiscovery import ranking


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    from drugdiscovery import config as cfg
    monkeypatch.setattr(cfg.settings.storage, "db_path", str(tmp_path / "t.db"))
    monkeypatch.setattr(cfg.settings.storage, "data_dir", str(tmp_path))
    init_db()


class StubLLM:
    """Deterministic skeptic verdict matching _SKEPTIC_SYSTEM contract."""
    def chat_json(self, system, user, temperature=None):
        assert "SKEPTIC" in system
        return {
            "verdict": "weaken",
            "confidence_adjustment": -0.2,
            "reasoning_summary": "Mouse-only data; link is correlative.",
            "alternative_explanations": ["stress artefact", "small n"],
            "weaknesses": ["no positive control"],
            "recommended_action": "keep_active",
            "next_counter_experiment": "human follicle explant assay",
        }


def _mk_ws_hyp(s):
    ws = repos.create_workspace(s, "Test MPB", "androgenetic alopecia")
    tgt, _ = repos.upsert_node(s, ws.id, "enzyme", "DKK3", confidence=0.6)
    h = repos.add_hypothesis(s, ws.id, title="DKK3 suppresses regeneration",
                             body="DKK3 inhibits Wnt in dermal papilla.",
                             mechanism_reasoning="", predicted_effect="",
                             status="active", confidence=0.85, target_id=tgt.id)
    return ws, h


def test_skeptic_weakens_and_records():
    with get_session() as s:
        ws, h = _mk_ws_hyp(s)
        r = crit.critique_hypothesis(s, ws, h, StubLLM())
        assert r["ok"] and r["verdict"] == "weaken"
        assert r["before"] == 0.85 and r["after"] == pytest.approx(0.65)
        prof = h.evidence_profile or {}
        lc = prof.get("last_critique")
        assert lc and lc["by"] == "skeptic"
        assert len(lc["alternative_explanations"]) == 2
        assert h.next_experiment == "human follicle explant assay"


def test_skeptic_demotes_when_low():
    with get_session() as s:
        ws, h = _mk_ws_hyp(s)
        h.confidence = 0.5
        r = crit.critique_hypothesis(s, ws, h, StubLLM())
        assert r["after"] == pytest.approx(0.3)
        assert h.status == "speculative"


def test_skeptic_llm_failure_returns_error_not_crash():
    class BrokenLLM:
        def chat_json(self, *a, **k):
            raise RuntimeError("model offline")
    with get_session() as s:
        ws, h = _mk_ws_hyp(s)
        r = crit.critique_hypothesis(s, ws, h, BrokenLLM())
        assert not r["ok"] and "model offline" in r["error"]
        assert h.confidence == 0.85  # untouched on failure


def test_score_to_binding_clamped():
    assert ranking.score_to_binding(-3.0) == 0.0
    assert ranking.score_to_binding(-12.0) == 1.0
    assert ranking.score_to_binding(-15.0) == 1.0
    assert ranking.score_to_binding(2.0) == 0.0
    assert ranking.score_to_binding(None) == 0.0


def test_drug_likeness_bounds():
    class Good:
        properties_json = {"molmass": 327, "logp": 3.1, "hbd": 1, "hba": 3}
        molmass = None; logp = None; hbd = None; hba = None
    assert ranking._drug_likeness(Good()) == pytest.approx(1.0)
    class Bad:
        properties_json = {"molmass": 800, "logp": 8, "hbd": 9, "hba": 14}
        molmass = None; logp = None; hbd = None; hba = None
    assert ranking._drug_likeness(Bad()) == 0.0


def test_rank_candidates_dimensions_and_order():
    from drugdiscovery.models import SimulationRun, Structure
    with get_session() as s:
        ws = repos.create_workspace(s, "Rank WS", "mpb")
        tgt, _ = repos.upsert_node(s, ws.id, "enzyme", "SRD5A2", confidence=0.8)
        from drugdiscovery import chem
        rec = chem.validate("CC(C)(C)NC(=O)C1CCC2C3CCC4NC(=O)C=CC4(C)C3CCC12C")
        c, _ = repos.upsert_compound(
            s, name="Finasteride", canonical_smiles=rec["canonical_smiles"],
            inchikey=rec["inchikey"],
            properties={"molmass": 361, "logp": 3.3, "hbd": 1, "hba": 3},
            sources={"pubchem": {"cid": "33571"}})
        cnode, _ = repos.upsert_node(s, ws.id, "drug", "Finasteride",
                                     properties={"compound_id": c.id})
        repos.add_edge(s, ws.id, "INHIBITS", cnode, tgt,
                       evidence_level="established", confidence=0.9,
                       provenance={"source": "test"})
        s.commit()  # populate uuid defaults before FK refs
        st = Structure(id="st1", workspace_id=ws.id, source="pdb",
                       source_id="7BW1", protein_node_id=tgt.id,
                       method="x-ray", resolution=2.1)
        s.add(st); s.flush()
        s.add(SimulationRun(id="sr1", workspace_id=ws.id, kind="docking",
                            compound_id=c.id, structure_id=st.id,
                            software="autodock-vina", score=-6.28))
        s.commit()
        R = ranking.rank_candidates(s, ws.id)
        assert len(R) == 1
        r0 = R[0]
        assert r0["compound_name"] == "Finasteride"
        assert r0["target"] == "SRD5A2"
        assert r0["best_docking"] == pytest.approx(-6.28)
        d = r0["dimensions"]
        assert d["mechanism_relevance"] == pytest.approx(0.9)
        assert d["target_confidence"] == pytest.approx(0.8)
        assert d["drug_likeness"] == pytest.approx(1.0)
        assert 0.0 <= r0["composite"] <= 1.0


def test_rank_weights_override_changes_composite():
    with get_session() as s:
        ws = repos.create_workspace(s, "Weight WS", "mpb")
        from drugdiscovery import chem
        rec = chem.validate("CCO")
        c, _ = repos.upsert_compound(
            s, name="Ethanol", canonical_smiles=rec["canonical_smiles"],
            inchikey=rec["inchikey"],
            properties={"molmass": 46}, sources={"pubchem": {"cid": "70"}})
        s.commit()
        base = ranking.rank_candidates(s, ws.id)[0]
        over = ranking.rank_candidates(s, ws.id, weights={"uncertainty": 1.0})[0]
        assert base["composite"] != over["composite"]
        # override MERGES with defaults: verify weighted mean exactly
        d, w = over["dimensions"], over["weights_used"]
        assert w["uncertainty"] == 1.0 and w["drug_likeness"] == 0.10
        expected = sum(d[k] * w[k] for k in d) / sum(w.values())
        assert over["composite"] == pytest.approx(expected, abs=1e-3)
