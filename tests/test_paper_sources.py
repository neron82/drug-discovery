"""Local-only regressions for retained primary source identity and artifact integrity."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from drugdiscovery import repos
from drugdiscovery.config import settings
from drugdiscovery.db import Base
from drugdiscovery.lit import sources
from drugdiscovery.lit.providers import EuropePMCProvider, PaperRecord, _parse_pubmed_xml
from drugdiscovery.models import PaperSource


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage, "artifact_dir", str(tmp_path / "artifacts"))
    engine = create_engine(f"sqlite:///{tmp_path / 'sources.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def network(monkeypatch):
    clients = []
    def install(handler):
        requests = []
        def respond(request):
            requests.append(request)
            return handler(request)
        client = httpx.Client(transport=httpx.MockTransport(respond))
        clients.append(client)
        monkeypatch.setattr(sources.httpx, "stream", client.stream)
        return requests
    yield install
    for client in clients:
        client.close()


def response(data, status=200, headers=None):
    return httpx.Response(status, stream=httpx.ByteStream(data), headers=headers)


def record(**changes):
    base = PaperRecord(pmid="123456", doi="10.0000/primary", title="Synthetic source",
                       abstract="Provider abstract, not article body.", authors=[], journal="",
                       year=None, source="synthetic", url="https://europepmc.org/article/MED/123456",
                       extra={"pmcid": "PMC123456"})
    return replace(base, **changes)


def paper_in(session, rec=None):
    ws = repos.create_workspace(session, "Retained source fixture")
    paper, _ = repos.upsert_paper(session, ws.id, rec or record())
    return paper


def xml(body="<sec><title>Results</title><p>Primary <italic>body</italic> &amp; H<sub>2</sub>O.</p></sec>",
        pmcid="PMC123456", extra=""):
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS//EN" "https://invalid.example/never-fetch.dtd">
<article><front><article-meta>
<article-id pub-id-type="pmcid">{pmcid}</article-id>
<article-id pub-id-type="pmid">123456</article-id>
<article-id pub-id-type="doi">10.0000/primary</article-id>
<abstract><p>Publisher abstract must not become body evidence.</p></abstract>
</article-meta></front><body>{body}</body>{extra}</article>'''.encode()


def test_retains_exact_original_and_derived_hashes_without_reacquiring(session, network):
    paper = paper_in(session)
    raw = xml(extra="<back><ref-list><ref><p>Reference text is excluded.</p></ref></ref-list></back>")
    calls = network(lambda request: response(raw))
    source = sources.acquire_full_text(session, paper)
    assert source is not None and source.kind == "full_text"
    assert Path(source.raw_path).read_bytes() == raw
    expected = "Results\n\nPrimary body & H2O."
    assert sources.read_source_text(source) == expected
    assert source.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert source.text_sha256 == hashlib.sha256(expected.encode()).hexdigest()
    assert source.text_chars == len(expected)
    source_id = source.id
    session.commit()
    session.expire_all()
    assert sources.acquire_full_text(session, paper).id == source_id
    assert len(calls) == 1
    assert len([s for s in paper.sources if s.kind == "full_text"]) == 1
    summary = sources.source_summary(source)
    assert summary["raw_sha256"] == source.raw_sha256
    assert "raw_path" not in summary and "text_path" not in summary


@pytest.mark.parametrize("raw", [
    xml(pmcid="PMC999999", extra='<back><ref><article-id pub-id-type="pmcid">PMC123456</article-id></ref></back>'),
    xml().replace(b"10.0000/primary", b"10.0000/wrong"),
    xml().replace(b"<body>", b"<sub-article><body>").replace(b"</body>", b"</body></sub-article>"),
    xml(body="<title>Body title only</title>"),
    xml(body="<sub-article><body><p>Other article prose.</p></body></sub-article>"),
    xml().replace(b"<body>", b"<abstract>").replace(b"</body>", b"</abstract>"),
    b'<!DOCTYPE article [<!ENTITY x "unsafe">]><article>&x;</article>',
])
def test_invalid_or_wrong_primary_article_never_becomes_full_text(session, network, raw):
    paper = paper_in(session)
    network(lambda request: response(raw))
    assert sources.acquire_full_text(session, paper) is None
    assert all(s.kind == "abstract" for s in paper.sources)
    assert sources.read_source_text(paper.sources[0]) == paper.abstract


def test_primary_body_excludes_nested_articles_and_preserves_table_text():
    text, _ = sources._parse_pmc_xml(xml(body='''<p>Primary sentence.</p>
      <sub-article><body><p>Nested foreign evidence.</p></body></sub-article>
      <table-wrap><caption><p>Primary table.</p></caption><table><tr><td>Group A</td><td>2 mg</td></tr></table></table-wrap>'''), "PMC123456")
    assert "Primary sentence." in text and "Group A 2 mg" in text
    assert "Nested foreign evidence" not in text


def test_existing_paper_enrichment_keeps_terminal_status_and_source_identity(session):
    paper = paper_in(session, record(abstract="", extra={}))
    paper.status = "extracted"
    enriched, created = repos.upsert_paper(session, paper.workspace_id, record())
    assert not created and enriched.id == paper.id
    assert paper.status == "extracted"
    assert paper.abstract == record().abstract
    retained = paper.sources[0]
    source_id, fetched_at = retained.id, retained.fetched_at
    for _ in range(2):
        same, created = repos.upsert_paper(session, paper.workspace_id, record())
        assert not created and same.id == paper.id
    assert [(s.id, s.fetched_at) for s in paper.sources] == [(source_id, fetched_at)]
    assert sources.read_source_text(retained) == record().abstract
    assert retained.metadata_json["publisher_xml"] is False
    assert retained.pmcid == "PMC123456"


def test_existing_abstract_gains_pmcid_without_a_duplicate_snapshot(session):
    paper = paper_in(session, record(extra={}))
    original = paper.sources[0]
    paper.status = "failed"
    same, created = repos.upsert_paper(session, paper.workspace_id, record())
    assert not created and same.id == paper.id
    assert paper.status == "failed"
    assert [(s.id, s.pmcid) for s in paper.sources] == [(original.id, "PMC123456")]


def test_pmc_only_provider_metadata_survives_empty_abstract_and_deduplicates(session, monkeypatch, network):
    hit = {"id": "PMC123456", "source": "PMC", "title": "PMC-only synthetic article", "isOpenAccess": "Y"}
    monkeypatch.setattr("drugdiscovery.lit.providers.httpx.get", lambda *args, **kwargs:
                        httpx.Response(200, json={"resultList": {"result": [hit]}},
                                       request=httpx.Request("GET", args[0])))
    rec, = EuropePMCProvider().search("synthetic")
    assert rec.pmid == "" and rec.doi == "" and rec.abstract == ""
    assert rec.url == "https://europepmc.org/article/PMC/PMC123456"
    paper = paper_in(session, rec)
    assert paper.sources[0].pmcid == "PMC123456"
    assert paper.sources[0].text_chars == 0
    same, created = repos.upsert_paper(session, paper.workspace_id, replace(rec, title="Corrected title"))
    assert not created and same.id == paper.id
    assert len(paper.sources) == 1
    calls = network(lambda request: response(xml()))
    source = sources.acquire_full_text(session, paper)
    assert source.kind == "full_text" and sources.read_source_text(source).startswith("Results")
    assert str(calls[0].url) == f"{sources.EPMC_BASE}/PMC123456/fullTextXML"
    assert len(calls) == 1


def test_pubmed_pmcid_is_primary_not_cited_reference():
    rec, = _parse_pubmed_xml('''<PubmedArticleSet><PubmedArticle>
      <MedlineCitation><PMID>1</PMID><Article><ArticleTitle>Primary</ArticleTitle></Article></MedlineCitation>
      <PubmedData><ReferenceList><Reference><ArticleIdList><ArticleId IdType="pmc">PMC999</ArticleId></ArticleIdList></Reference></ReferenceList>
      <ArticleIdList><ArticleId IdType="pmc">PMC123456</ArticleId></ArticleIdList></PubmedData>
      </PubmedArticle></PubmedArticleSet>''')
    assert rec.extra["pmcid"] == "PMC123456"


def test_missing_corrupt_and_escaped_artifacts_fail_explicitly(session, tmp_path):
    source = paper_in(session).sources[0]
    artifact = Path(source.text_path)
    original = artifact.read_bytes()
    artifact.unlink()
    with pytest.raises(FileNotFoundError):
        sources.read_source_text(source)
    artifact.write_bytes(b"Changed source bytes")
    with pytest.raises(ValueError, match="checksum"):
        sources.read_source_text(source)
    with pytest.raises(ValueError, match="checksum"):
        sources.verified_source_path(source, "raw")
    with pytest.raises(ValueError, match="checksum"):
        sources._artifact(original, "txt")
    assert artifact.read_bytes() == b"Changed source bytes"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(original)
    source.text_path = str(outside)
    with pytest.raises(ValueError, match="outside"):
        sources.read_source_text(source)


def test_corrupt_retained_full_text_is_not_replaced_or_refetched(session, network):
    paper = paper_in(session)
    calls = network(lambda request: response(xml()))
    source = sources.acquire_full_text(session, paper)
    Path(source.raw_path).write_bytes(b"corrupt")
    assert sources.acquire_full_text(session, paper) is None
    assert len(calls) == 1
    assert Path(source.raw_path).read_bytes() == b"corrupt"


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.consumed = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk


def test_retrieval_rejects_unannounced_oversize_body_without_consuming_remainder(network):
    stream = Chunks([b"12345", b"67890", b"not consumed"])
    network(lambda request: httpx.Response(200, stream=stream))
    with pytest.raises(ValueError, match="byte limit"):
        sources._download(f"{sources.EPMC_BASE}/PMC123456/fullTextXML", limit=8)
    assert stream.consumed == 2


@pytest.mark.parametrize("headers", [{"Content-Length": "999"}, {"Content-Encoding": "gzip"}])
def test_retrieval_rejects_oversize_or_compressed_headers_before_consumption(network, headers):
    stream = Chunks([b"never consumed"])
    network(lambda request: httpx.Response(200, stream=stream, headers=headers))
    with pytest.raises(ValueError):
        sources._download(f"{sources.EPMC_BASE}/PMC123456/fullTextXML", limit=8)
    assert stream.consumed == 0


def test_retrieval_stops_slow_stream_at_elapsed_bound(network, monkeypatch):
    stream = Chunks([b"first", b"late", b"never consumed"])
    network(lambda request: httpx.Response(200, stream=stream))
    ticks = iter([0, 1, sources.RETRIEVAL_DEADLINE + 1])
    monkeypatch.setattr(sources, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ValueError, match="elapsed-time"):
        sources._download(f"{sources.EPMC_BASE}/PMC123456/fullTextXML", limit=100)
    assert stream.consumed == 2


def test_lookup_checks_primary_identifiers_before_fetching_xml(session, network):
    paper = paper_in(session, record(extra={}))
    payload = {"resultList": {"result": [
        {"pmid": "999999", "doi": "10.0000/wrong", "pmcid": "PMC999999"},
        {"pmid": "123456", "doi": "10.0000/primary", "pmcid": "PMC123456"},
    ]}}
    calls = network(lambda request: response(json.dumps(payload).encode()) if request.url.path.endswith("/search")
                    else response(xml()))
    assert sources.acquire_full_text(session, paper).pmcid == "PMC123456"
    assert len(calls) == 2
    assert calls[0].url.params["query"] == "EXT_ID:123456 AND SRC:MED"
    assert calls[1].url.path.endswith("/PMC123456/fullTextXML")


def test_unavailable_source_returns_no_full_text_and_does_not_retry(session, network):
    paper = paper_in(session)
    calls = network(lambda request: response(b"Not available", status=404))
    assert sources.acquire_full_text(session, paper) is None
    assert len(calls) == 1
    assert session.execute(select(PaperSource).where(PaperSource.kind == "full_text")).scalars().all() == []
