"""Source-aware extraction regressions with isolated storage and no network/model calls."""
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from drugdiscovery import extract, repos
from drugdiscovery.config import settings
from drugdiscovery.db import Base
from drugdiscovery.lit.sources import retain_abstract
from drugdiscovery.models import Edge, Node, Paper, PaperSource, Statement


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(extract, "acquire_full_text", lambda session, paper: None)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


class ClaimModel:
    model = "synthetic-extraction-model"

    def __init__(self, claims):
        self.claims = claims
        self.inputs = []

    def chat_json(self, system, user, **kwargs):
        self.inputs.append(user)
        return {"claims": self.claims}


def _paper(session, abstract):
    ws = repos.create_workspace(session, "Extraction fixtures", "synthetic")
    paper = Paper(
        workspace_id=ws.id, title="Synthetic study", abstract=abstract,
        source="synthetic", url="https://example.org/study", content_hash=uuid4().hex,
    )
    session.add(paper)
    session.flush()
    return ws, paper


def _full_text(session, paper, text):
    directory = Path(settings.storage.artifact_dir) / uuid4().hex
    directory.mkdir(parents=True)
    raw = b"<article><body><p>Synthetic retained source fixture</p></body></article>"
    raw_path, text_path = directory / "source.xml", directory / "text.txt"
    raw_path.write_bytes(raw)
    text_path.write_bytes(text.encode("utf-8"))
    source = PaperSource(
        paper=paper, kind="full_text", source_url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fullTextXML",
        pmcid="PMC123456", raw_path=str(raw_path), raw_sha256=sha256(raw).hexdigest(),
        media_type="application/xml", text_path=str(text_path),
        text_sha256=sha256(text.encode("utf-8")).hexdigest(), text_chars=len(text),
        metadata_json={"provider": "synthetic"},
    )
    session.add(source)
    session.flush()
    return source


def _claim(quotation):
    return {
        "text": "DHT activates AR in the synthetic experiment.",
        "subject": "DHT", "subject_type": "hormone", "relation": "ACTIVATES",
        "object": "AR", "object_type": "receptor", "quotation": quotation,
        "confidence": 0.7, "evidence_level": "probable",
        "significance": "p<0.05", "contradicted": True,
    }


def _assert_no_claim_graph(session):
    for model in (Statement, Node, Edge):
        assert session.scalars(select(model)).all() == []


@pytest.mark.parametrize("quotation", [None, "   ", "DHT activates AR."])
def test_unsupported_or_empty_model_quotation_creates_no_source_facts(session, quotation):
    ws, paper = _paper(session, "DHT did not activate AR in this experiment.")
    llm = ClaimModel([_claim(quotation)])

    extract.process_paper(session, ws, paper, llm)

    _assert_no_claim_graph(session)
    assert paper.status == "failed"


def test_full_text_only_excerpt_records_exact_scope_and_rejects_omitted_quote(session, monkeypatch):
    ws, paper = _paper(session, "")
    body = "DHT\n\tactivates AR.\n" + "α background information. " * 400 + "Omitted finding."
    source = _full_text(session, paper, body)
    monkeypatch.setattr(extract, "acquire_full_text", lambda session, paper: source)
    llm = ClaimModel([_claim("DHT activates AR."), _claim("Omitted finding.")])

    extract.process_paper(session, ws, paper, llm)
    extract.process_paper(session, ws, paper, llm)

    statement = session.scalars(select(Statement)).one()
    edge = session.scalars(select(Edge)).one()
    expected = {
        "source_id": source.id, "kind": "full_text", "source_url": source.source_url,
        "text_sha256": sha256(body.encode("utf-8")).hexdigest(),
        "input_sha256": sha256(body[:8500].encode("utf-8")).hexdigest(),
        "start_char": 0, "end_char": 8500, "total_chars": len(body), "truncated": True,
    }
    assert statement.properties == {
        "contradicted": True, "significance": "p<0.05", "extraction_source": expected,
    }
    assert edge.provenance == [{
        "paper_id": paper.id, "statement_id": statement.id, "kind": "supports",
        "extraction_source": expected,
    }]
    assert statement.quotation == "DHT activates AR."
    assert len(llm.inputs) == 1
    assert llm.inputs[0].endswith(body[:8500])
    assert "Omitted finding." not in llm.inputs[0]
    assert paper.status == "extracted"


@pytest.mark.parametrize("retained_only", [False, True])
def test_abstract_fallback_preserves_verifiable_input_identity(session, retained_only):
    abstract = "DHT activates AR.\nSynthetic abstract."
    ws, paper = _paper(session, abstract)
    if retained_only:
        retain_abstract(session, paper)
        paper.abstract = ""
    llm = ClaimModel([_claim("DHT activates AR.")])

    extract.process_paper(session, ws, paper, llm)

    source = session.scalars(select(PaperSource).where(PaperSource.kind == "abstract")).one()
    statement = session.scalars(select(Statement)).one()
    expected = {
        "source_id": source.id, "kind": "abstract", "source_url": source.source_url,
        "text_sha256": sha256(abstract.encode("utf-8")).hexdigest(),
        "input_sha256": sha256(abstract.encode("utf-8")).hexdigest(),
        "start_char": 0, "end_char": len(abstract), "total_chars": len(abstract), "truncated": False,
    }
    assert statement.properties["extraction_source"] == expected
    assert session.scalars(select(Edge)).one().provenance[0]["extraction_source"] == expected
    assert llm.inputs[0].endswith(abstract)


@pytest.mark.parametrize("status", ["extracted", "failed"])
def test_terminal_historical_claims_unchanged_after_full_text_retained(session, monkeypatch, status):
    ws, paper = _paper(session, "Historical abstract.")
    paper.status = status
    paper.extractor = "historical-model"
    old_properties = {"significance": "old result", "custom": {"preserve": True}}
    statement = repos.add_statement(
        session, ws.id, "Historical claim", paper_id=paper.id, quotation="Old quotation",
        properties=old_properties, extractor_model="historical-model",
    )
    subject, _ = repos.upsert_node(session, ws.id, "protein", "Historical subject")
    target, _ = repos.upsert_node(session, ws.id, "protein", "Historical target")
    old_provenance = {"paper_id": paper.id, "statement_id": statement.id, "kind": "supports"}
    edge = repos.add_edge(session, ws.id, "ACTIVATES", subject, target, provenance=old_provenance)
    _full_text(session, paper, "Newly retained full text must not relabel old claims.")
    session.commit()

    def unexpected_acquisition(*args):
        pytest.fail("Terminal papers must not acquire or extract sources again")

    monkeypatch.setattr(extract, "acquire_full_text", unexpected_acquisition)
    llm = ClaimModel([_claim("Newly retained full text")])
    extract.process_paper(session, ws, paper, llm)
    session.commit()
    session.expire_all()

    assert session.scalars(select(Statement)).one().id == statement.id
    assert statement.properties == old_properties
    assert statement.quotation == "Old quotation"
    assert statement.extractor_model == "historical-model"
    assert edge.provenance == [old_provenance]
    assert paper.status == status
    assert paper.extractor == "historical-model"
    assert llm.inputs == []


def test_corrupt_selected_source_is_not_replaced_with_unverified_text(session, monkeypatch):
    ws, paper = _paper(session, "DHT activates AR.")
    source = _full_text(session, paper, "DHT activates AR.")
    Path(source.text_path).write_text("Altered source text", encoding="utf-8")
    monkeypatch.setattr(extract, "acquire_full_text", lambda session, paper: source)
    llm = ClaimModel([_claim("DHT activates AR.")])

    extract.process_paper(session, ws, paper, llm)

    assert paper.status == "failed"
    assert llm.inputs == []
    _assert_no_claim_graph(session)
