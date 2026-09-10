"""Retained, hash-verified abstract snapshots and accessible primary PMC article bodies."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Paper, PaperSource
from .providers import EPMC_BASE, normalize_pmcid

log = logging.getLogger("dd.lit.sources")
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_LOOKUP_BYTES = 512 * 1024
REQUEST_TIMEOUT = 10.0
RETRIEVAL_DEADLINE = 30.0
PARSER_VERSION = "pmc-primary-body-1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _confined_path(path: str) -> Path:
    root = Path(settings.storage.artifact_dir).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Source artifact is outside the artifact root")
    return target


def verified_source_path(source: PaperSource, format: str = "text") -> Path:
    """Return a confined artifact path only after checking its retained byte hash."""
    if format not in {"text", "raw"}:
        raise ValueError("Unknown source format")
    path = _confined_path(source.text_path if format == "text" else source.raw_path)
    if path.exists() and not path.is_file():
        raise ValueError("Source artifact is not a regular file")
    expected = source.text_sha256 if format == "text" else source.raw_sha256
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Source artifact checksum mismatch")
    return path


def read_source_text(source: PaperSource) -> str:
    path = _confined_path(source.text_path)
    if path.exists() and not path.is_file():
        raise ValueError("Source artifact is not a regular file")
    data = path.read_bytes()
    if _sha(data) != source.text_sha256:
        raise ValueError("Source artifact checksum mismatch")
    return data.decode("utf-8")


def source_summary(source: PaperSource) -> dict:
    return {
        "id": source.id, "kind": source.kind, "source_url": source.source_url,
        "pmcid": source.pmcid, "raw_sha256": source.raw_sha256,
        "text_sha256": source.text_sha256, "text_chars": source.text_chars,
        "media_type": source.media_type, "metadata": source.metadata_json or {},
        "fetched_at": source.fetched_at.isoformat() if source.fetched_at else None,
    }


def _artifact(data: bytes, suffix: str) -> tuple[str, str]:
    """Publish by hash without overwriting even an already-corrupt retained artifact."""
    digest = _sha(data)
    directory = Path(settings.storage.artifact_dir).resolve() / "papers"
    directory.mkdir(parents=True, exist_ok=True)
    target = _confined_path(str(directory / f"{digest}.{suffix}"))
    if target.exists():
        if _sha(target.read_bytes()) != digest:
            raise ValueError("Existing retained artifact checksum mismatch")
        return str(target), digest
    fd, temporary = tempfile.mkstemp(prefix=".source-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _sha(target.read_bytes()) != digest:
                raise ValueError("Existing retained artifact checksum mismatch")
    finally:
        os.unlink(temporary)
    return str(target), digest


def _retain(session: Session, paper: Paper, *, kind: str, source_url: str,
            pmcid: str, raw: bytes, text: str, media_type: str, metadata: dict) -> PaperSource:
    text_bytes = text.encode("utf-8")
    digest = _sha(text_bytes)
    existing = session.execute(select(PaperSource).where(
        PaperSource.paper_id == paper.id, PaperSource.kind == kind,
        PaperSource.text_sha256 == digest,
    )).scalar_one_or_none()
    if existing is not None:
        # Metadata-only enrichment must not replace the original snapshot or evidence identity.
        if pmcid and not existing.pmcid:
            existing.pmcid = pmcid
        existing.metadata_json = {**(existing.metadata_json or {}),
                                  **{k: v for k, v in metadata.items() if v not in (None, "")}}
        return existing
    raw_path, raw_hash = _artifact(raw, "xml" if kind == "full_text" else "txt")
    text_path, text_hash = _artifact(text_bytes, "txt")
    source = PaperSource(
        paper=paper, kind=kind, source_url=source_url, pmcid=pmcid,
        raw_path=raw_path, raw_sha256=raw_hash, media_type=media_type,
        text_path=text_path, text_sha256=text_hash, text_chars=len(text), metadata_json=metadata,
    )
    session.add(source)
    session.flush()
    return source


def retain_abstract(session: Session, paper: Paper, extra: dict | None = None) -> PaperSource | None:
    extra = dict(extra or {})
    pmcid = normalize_pmcid(extra.get("pmcid"))
    if not pmcid:
        pmcid = next((s.pmcid for s in paper.sources if s.pmcid), "")
    text = paper.abstract or ""
    if not text and not pmcid:
        return None
    metadata = {**extra, "provider": paper.source, "pmcid": pmcid,
                "representation": "provider-derived abstract", "publisher_xml": False}
    return _retain(session, paper, kind="abstract", source_url=paper.url or "", pmcid=pmcid,
                   raw=text.encode("utf-8"), text=text, media_type="text/plain", metadata=metadata)


def _download(url: str, *, params: dict | None = None, limit: int) -> bytes:
    """One fixed-endpoint request, with bounded decoded bytes and elapsed time; no retries."""
    start = time.monotonic()
    with httpx.stream("GET", url, params=params, timeout=REQUEST_TIMEOUT,
                      follow_redirects=False, headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError("Compressed source response is not accepted")
        length = response.headers.get("content-length")
        if length and int(length) > limit:
            raise ValueError("Source response exceeds byte limit")
        data = bytearray()
        for chunk in response.iter_raw():
            if time.monotonic() - start > RETRIEVAL_DEADLINE:
                raise ValueError("Source retrieval exceeded elapsed-time limit")
            if len(data) + len(chunk) > limit:
                raise ValueError("Source response exceeds byte limit")
            data.extend(chunk)
        return bytes(data)


def _doi(value: str) -> str:
    return str(value or "").strip().lower()


def _resolve_pmcid(paper: Paper) -> str:
    retained = {s.pmcid for s in paper.sources if normalize_pmcid(s.pmcid)}
    if len(retained) > 1:
        raise ValueError("Conflicting retained primary PMC identifiers")
    if retained:
        return retained.pop()
    pmid = str(paper.pmid or "").strip()
    doi = _doi(paper.doi)
    if pmid and re.fullmatch(r"[0-9]+", pmid):
        query = f"EXT_ID:{pmid} AND SRC:MED"
    elif doi and re.fullmatch(r"10\.[0-9]{4,9}/[^\s\"\\]+", doi):
        query = f'DOI:"{doi}"'
    else:
        return ""
    payload = json.loads(_download(f"{EPMC_BASE}/search", params={
        "query": query, "format": "json", "resultType": "core", "pageSize": "10",
    }, limit=MAX_LOOKUP_BYTES))
    if not isinstance(payload, dict) or not isinstance(payload.get("resultList", {}), dict):
        raise ValueError("Malformed PMC lookup response")
    hits = (payload.get("resultList") or {}).get("result") or []
    if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
        raise ValueError("Malformed PMC lookup results")
    candidates = set()
    for hit in hits:
        hit_pmid = str(hit.get("pmid") or (hit.get("id") if hit.get("source") == "MED" else "") or "")
        hit_doi = _doi(hit.get("doi"))
        if pmid and hit_pmid != pmid:
            continue
        if not pmid and hit_doi != doi:
            continue
        if doi and hit_doi and hit_doi != doi:
            continue
        pmcid = normalize_pmcid(hit.get("pmcid") or (hit.get("id") if hit.get("source") == "PMC" else ""))
        if pmcid:
            candidates.add(pmcid)
    if len(candidates) > 1:
        raise ValueError("Ambiguous PMC lookup for primary article")
    return next(iter(candidates), "")


_SKIP = {"sub-article", "response", "ref-list", "abstract", "back", "front"}
_BLOCK = {"body", "sec", "title", "p", "list", "list-item", "table", "tr", "caption",
          "fig", "table-wrap", "disp-quote", "boxed-text", "disp-formula"}


def _body_text(node: ET.Element) -> str:
    parts = []
    def visit(element):
        if element.tag in _SKIP:
            return
        if element.tag in _BLOCK:
            parts.append("\n\n")
        if element.text:
            parts.append(element.text)
        for child in element:
            visit(child)
            if child.tail:
                parts.append(child.tail)
        if element.tag in {"td", "th"}:
            parts.append("\t")
        if element.tag in _BLOCK:
            parts.append("\n\n")
    visit(node)
    return "\n\n".join(" ".join(part.split()) for part in "".join(parts).split("\n\n") if part.strip())


def _parse_pmc_xml(raw: bytes, pmcid: str, *, pmid: str = "", doi: str = "") -> tuple[str, dict]:
    # External JATS DOCTYPEs are common; ElementTree does not fetch them. Internal
    # entities are unnecessary for the retained representation and are rejected.
    if b"<!ENTITY" in raw.replace(b"\x00", b"").upper():
        raise ValueError("PMC XML contains an entity declaration")
    root = ET.fromstring(raw)
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag != "article":
        raise ValueError("PMC XML does not contain a primary article root")
    meta = root.find("front/article-meta")
    if meta is None:
        raise ValueError("PMC XML lacks primary article metadata")
    ids: dict[str, set[str]] = {}
    for identifier in meta.findall("article-id"):
        kind = (identifier.get("pub-id-type") or "").lower()
        value = "".join(identifier.itertext()).strip()
        if kind in {"pmc", "pmcid"}:
            kind = "pmcid"
            value = normalize_pmcid(value if value.upper().startswith("PMC") else f"PMC{value}")
        elif kind == "doi":
            value = _doi(value)
        ids.setdefault(kind, set()).add(value)
    if ids.get("pmcid") != {pmcid}:
        raise ValueError("PMC XML primary article identifier does not match requested PMCID")
    for kind, expected in (("pmid", str(pmid or "").strip()), ("doi", _doi(doi))):
        if expected and ids.get(kind) and ids[kind] != {expected}:
            raise ValueError(f"PMC XML primary article {kind} does not match publication")
    body = root.find("body")
    if body is None:
        raise ValueError("PMC XML primary article has no body")
    # Exclude nested article/reference prose from both eligibility and derived text.
    def has_prose(element):
        if element.tag in _SKIP:
            return False
        if element.tag in {"p", "td", "th"} and _body_text(element):
            return True
        return any(has_prose(child) for child in element)
    text = _body_text(body)
    if not text or not has_prose(body):
        raise ValueError("PMC XML primary article body has no usable prose")
    permissions = meta.find("permissions")
    return text, {"provider": "europepmc", "parser_version": PARSER_VERSION,
                  "representation": "primary article body", "publisher_xml": True,
                  "article_ids": {k: sorted(v) for k, v in ids.items()},
                  "license": " ".join("".join(permissions.itertext()).split()) if permissions is not None else ""}


def acquire_full_text(session: Session, paper: Paper) -> PaperSource | None:
    """Acquire accessible PMC text, or return None for an explicitly labelled fallback."""
    try:
        retained = [s for s in paper.sources if s.kind == "full_text"]
        if retained:
            source = max(retained, key=lambda s: (s.fetched_at.isoformat(), s.id))
            verified_source_path(source, "raw")
            read_source_text(source)
            return source
        pmcid = _resolve_pmcid(paper)
        if not pmcid:
            log.info("Full text unavailable for paper %s: no unambiguous PMC identifier", paper.id)
            return None
        url = f"{EPMC_BASE}/{pmcid}/fullTextXML"
        raw = _download(url, limit=MAX_XML_BYTES)
        text, metadata = _parse_pmc_xml(raw, pmcid, pmid=paper.pmid, doi=paper.doi)
        return _retain(session, paper, kind="full_text", source_url=url, pmcid=pmcid,
                       raw=raw, text=text, media_type="application/xml", metadata=metadata)
    except (httpx.HTTPError, OSError, ValueError, ET.ParseError, RecursionError) as exc:
        log.warning("Full text unavailable for paper %s: %s: %s; abstract fallback only", paper.id,
                    type(exc).__name__, exc)
        return None
