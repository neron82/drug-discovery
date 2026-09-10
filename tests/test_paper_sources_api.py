"""Retained-source API access, integrity, and historical attribution regressions."""
from hashlib import sha256
from pathlib import Path

import pytest

from drugdiscovery import db, repos
from drugdiscovery.config import settings
from drugdiscovery.models import Paper, PaperSource, Statement


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage, "db_path", str(tmp_path / "sources.db"))
    monkeypatch.setattr(settings.storage, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings.storage, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings.storage, "log_dir", str(tmp_path / "logs"))
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    db.init_db()
    # Import only after storage isolation. Do not enter lifespan: it starts workers.
    from fastapi.testclient import TestClient
    from drugdiscovery.api import app

    http = TestClient(app)
    try:
        yield http
    finally:
        http.close()
        db.get_engine().dispose()
        db._engine = None
        db._SessionLocal = None


def make_papers():
    with db.get_session() as session:
        workspace = repos.create_workspace(session, "Source API fixture", "synthetic")
        papers = [Paper(workspace_id=workspace.id, content_hash=str(index),
                        title=f"Synthetic paper {index}", abstract="Original abstract claim.",
                        status="extracted") for index in range(2)]
        session.add_all(papers)
        session.flush()
        result = workspace.id, papers[0].id, papers[1].id
        session.commit()
        return result


def retained_source(session, paper):
    raw = b'<?xml version="1.0"?><article><body><p>Retained \xce\xb2 claim.</p></body></article>\n'
    text = "Retained β claim.\nFurther body text."
    root = Path(settings.storage.artifact_dir) / "papers" / paper.id
    root.mkdir(parents=True, exist_ok=True)
    raw_path, text_path = root / "raw.xml", root / "text.txt"
    raw_path.write_bytes(raw)
    text_path.write_bytes(text.encode("utf-8"))
    source = PaperSource(paper_id=paper.id, kind="full_text", pmcid="PMC1234567",
                         source_url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1234567/fullTextXML",
                         raw_path=str(raw_path), text_path=str(text_path),
                         raw_sha256=sha256(raw).hexdigest(),
                         text_sha256=sha256(text.encode("utf-8")).hexdigest(),
                         text_chars=len(text), media_type="application/xml",
                         metadata_json={"provider": "synthetic-api-fixture"})
    session.add(source)
    session.flush()
    return source, raw, text


def test_retained_artifacts_are_exact_safe_and_owned_by_paper(client):
    _, paper_id, other_id = make_papers()
    with db.get_session() as session:
        source, raw, text = retained_source(session, session.get(Paper, paper_id))
        source_id = source.id
        session.commit()

    detail = client.get(f"/api/papers/{paper_id}")
    assert detail.status_code == 200
    summary = detail.json()["sources"][0]
    assert "raw_path" not in summary and "text_path" not in summary
    assert settings.storage.artifact_dir not in detail.text
    text_response = client.get(summary["text_url"])
    assert text_response.status_code == 200
    assert text_response.content == text.encode("utf-8")
    assert text_response.headers["content-type"].startswith("text/plain")
    raw_response = client.get(summary["raw_url"])
    assert raw_response.content == raw
    assert raw_response.headers["content-type"].startswith("application/xml")
    assert raw_response.headers["content-disposition"].startswith("attachment;")
    assert raw_response.headers["x-content-type-options"] == "nosniff"
    for format in ("raw", "text"):
        refused = client.get(f"/api/papers/{other_id}/sources/{source_id}?format={format}")
        assert refused.status_code == 404
        assert "Retained" not in refused.text


@pytest.mark.parametrize("damage,expected", [("corrupt", 409), ("missing", 404), ("escape", 409)])
@pytest.mark.parametrize("format", ["raw", "text"])
def test_source_artifact_integrity_and_confinement(client, tmp_path, damage, expected, format):
    _, paper_id, _ = make_papers()
    with db.get_session() as session:
        source, _, _ = retained_source(session, session.get(Paper, paper_id))
        source_id = source.id
        path = Path(getattr(source, f"{format}_path"))
        if damage == "corrupt":
            path.write_bytes(b"CORRUPTED SECRET")
        elif damage == "missing":
            path.unlink()
        else:
            outside = tmp_path / "outside-artifact-root"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
        session.commit()
    response = client.get(f"/api/papers/{paper_id}/sources/{source_id}?format={format}")
    assert response.status_code == expected
    assert "CORRUPTED SECRET" not in response.text
    assert str(tmp_path) not in response.text


def test_manual_acquisition_preserves_legacy_claims_and_records_new_scope(client, monkeypatch):
    from drugdiscovery import api

    workspace_id, paper_id, _ = make_papers()
    with db.get_session() as session:
        node, _ = repos.upsert_node(session, workspace_id, "protein", "Synthetic target")
        legacy = Statement(workspace_id=workspace_id, source_paper_id=paper_id,
                           text="Historical claim", subject=node.canonical_name,
                           quotation="Original abstract claim.", properties={"dose": "historical dose"})
        session.add(legacy)
        session.flush()
        legacy_id, node_id = legacy.id, node.id
        session.commit()
    before = client.get(f"/api/papers/{paper_id}").json()
    legacy_scope = {"kind": "abstract", "legacy": True, "truncated": None}
    assert before["statements"][0]["extraction_source"] == legacy_scope

    def acquire(session, paper):
        source, _, _ = retained_source(session, paper)
        return source

    monkeypatch.setattr(api, "acquire_full_text", acquire)
    fetched = client.post(f"/api/papers/{paper_id}/full-text")
    assert fetched.status_code == 200
    assert client.get(fetched.json()["text_url"]).status_code == 200
    after = client.get(f"/api/papers/{paper_id}").json()
    assert after["status"] == "extracted"
    assert after["abstract"] == before["abstract"]
    assert after["sources"][0]["kind"] == "full_text"
    assert after["statements"] == before["statements"]
    card = client.get(f"/api/nodes/{node_id}").json()
    assert card["statements"][0]["extraction_source"] == legacy_scope
    with db.get_session() as session:
        assert session.get(Statement, legacy_id).properties == {"dose": "historical dose"}
        source = session.get(PaperSource, fetched.json()["id"])
        text = Path(source.text_path).read_text(encoding="utf-8")
        excerpt = text[:17]
        scope = {"source_id": source.id, "kind": "full_text", "source_url": source.source_url,
                 "text_sha256": source.text_sha256,
                 "input_sha256": sha256(excerpt.encode("utf-8")).hexdigest(),
                 "start_char": 0, "end_char": len(excerpt), "total_chars": len(text), "truncated": True}
        fresh = Statement(workspace_id=workspace_id, source_paper_id=paper_id,
                          text="Retained body claim", quotation="Retained β claim.",
                          properties={"extraction_source": scope})
        session.add(fresh)
        session.flush()
        fresh_id = fresh.id
        session.commit()
    statements = client.get(f"/api/papers/{paper_id}").json()["statements"]
    fresh_result = next(statement for statement in statements if statement["id"] == fresh_id)
    assert fresh_result["extraction_source"] == scope
    retained = client.get(fetched.json()["text_url"]).text
    selected = retained[scope["start_char"]:scope["end_char"]]
    assert fresh_result["quotation"] in selected
    assert sha256(selected.encode("utf-8")).hexdigest() == scope["input_sha256"]


def test_unavailable_manual_full_text_is_an_error_not_an_abstract_success(client, monkeypatch):
    from drugdiscovery import api

    _, paper_id, _ = make_papers()
    monkeypatch.setattr(api, "acquire_full_text", lambda session, paper: None)
    response = client.post(f"/api/papers/{paper_id}/full-text")
    assert response.status_code == 409
    detail = client.get(f"/api/papers/{paper_id}").json()
    assert detail["status"] == "extracted"
    assert detail["sources"] == []
    assert client.post("/api/papers/missing-paper/full-text").status_code == 404
