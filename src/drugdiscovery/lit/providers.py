"""Literature provider adapters: Europe PMC (default) + PubMed/NCBI E-utilities.

Common record: PaperRecord. Each provider implements SearchProvider.search(query, size) -> list.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from ..config import settings

log = logging.getLogger("dd.lit")

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class PaperRecord:
    pmid: str
    doi: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    year: int | None
    source: str
    url: str
    extra: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        blob = "|".join([self.pmid, self.title, self.abstract, self.doi]).encode()
        return hashlib.sha256(blob).hexdigest()



def normalize_pmcid(value) -> str:
    value = str(value or "").strip().upper()
    return value if re.fullmatch(r"PMC[1-9][0-9]*", value) else ""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, size: int) -> list[PaperRecord]: ...


class EuropePMCProvider:
    name = "europepmc"

    def search(self, query: str, size: int = 12) -> list[PaperRecord]:
        params = {
            "query": _emc_query(query),
            "format": "json",
            "pageSize": str(size),
            "resultType": "core",
        }
        r = httpx.get(f"{EPMC_BASE}/search", params=params, timeout=60)
        r.raise_for_status()
        hits = (r.json().get("resultList") or {}).get("result") or []
        out = []
        for h in hits:
            if not (h.get("title")):
                continue
            authors = []
            if h.get("authorString"):
                authors = [a.strip() for a in h["authorString"].split(",") if a.strip()][:20]
            jinfo = h.get("journalInfo") or {}
            jtitle = (jinfo.get("journal") or {}).get("title", "")
            year = h.get("pubYear") or ((jinfo.get("yearOfPublication") or 0) or None)
            try:
                year = int(year) if year else None
            except Exception:
                year = None
            doi = h.get("doi") or ""
            pmid = h.get("pmid") or (h.get("id") if h.get("source") == "MED" else "") or ""
            pmcid = normalize_pmcid(h.get("pmcid") or (h.get("id") if h.get("source") == "PMC" else ""))
            url = (f"https://europepmc.org/article/MED/{pmid}" if pmid else
                   f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else "")
            out.append(PaperRecord(
                pmid=str(pmid), doi=doi, title=h["title"],
                abstract=h.get("abstractText") or "", authors=authors,
                journal=jtitle, year=year, source="europepmc", url=url,
                extra={"id": h.get("id"), "source_attribution": h.get("source"), "pmcid": pmcid},
            ))
        return out


def _emc_query(q: str) -> str:
    """Include abstracts and open-access articles, including full-text-only records."""
    return f"({q}) AND (HAS_ABSTRACT:Y OR OPEN_ACCESS:Y)"


class PubMedProvider:
    name = "pubmed"

    def search(self, query: str, size: int = 12) -> list[PaperRecord]:
        api_key = settings.literature.get("api_key", "")
        params = {
            "db": "pubmed", "term": query, "retmax": str(size), "retmode": "json", "sort": "relevance",
        }
        if api_key:
            params["api_key"] = api_key
        else:
            time.sleep(0.34)  # ~3 req/s courtesy
        r = httpx.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=60)
        r.raise_for_status()
        ids = ((r.json().get("esearchresult") or {}).get("idlist")) or []
        if not ids:
            return []
        return self._fetch(ids, api_key=api_key)

    def _fetch(self, ids: list[str], api_key: str = "") -> list[PaperRecord]:
        if api_key:
            pass
        else:
            time.sleep(0.34)
        r = httpx.get(f"{NCBI_BASE}/efetch.fcgi",
                      params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml",
                              **({"api_key": api_key} if api_key else {})}, timeout=90)
        r.raise_for_status()
        return _parse_pubmed_xml(r.text)


def _parse_pubmed_xml(xml: str) -> list[PaperRecord]:
    import xml.etree.ElementTree as ET
    out = []
    root = ET.fromstring(xml)
    for art in root.iter("PubmedArticle"):
        article = art.find("MedlineCitation/Article")
        if article is None:
            continue
        pmid = (art.findtext("MedlineCitation/PMID") or "").strip()
        title_element = article.find("ArticleTitle")
        title = "".join(title_element.itertext()).strip() if title_element is not None else ""
        if not title:
            continue
        sections = []
        for section in article.findall("Abstract/AbstractText"):
            text = "".join(section.itertext()).strip()
            if not text:
                continue
            label = (section.get("Label") or "").strip()
            sections.append(f"{label}: {text}" if label else text)
        abstract = "\n\n".join(sections)
        authors = [f"{a.findtext('LastName','')} {a.findtext('Initials','')}".strip()
                   for a in article.findall("AuthorList/Author")][:20]
        journal = (article.findtext("Journal/Title") or "").strip()
        year_s = article.findtext("Journal/JournalIssue/PubDate/Year") or article.findtext("Journal/JournalIssue/PubDate/MedlineDate")
        year = None
        try:
            year = int(year_s[:4]) if year_s else None
        except Exception:
            year = None
        doi = (art.findtext("PubmedData/ArticleIdList/ArticleId[@IdType='doi']") or "").strip()
        pmcid = normalize_pmcid(art.findtext("PubmedData/ArticleIdList/ArticleId[@IdType='pmc']"))
        out.append(PaperRecord(pmid=pmid, doi=doi, title=title, abstract=abstract, authors=authors,
                               journal=journal, year=year, source="pubmed",
                               url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", extra={"pmcid": pmcid}))
    return out


PROVIDERS: dict[str, SearchProvider] = {
    "europepmc": EuropePMCProvider(),
    "pubmed": PubMedProvider(),
}


def get_provider(name: str | None = None) -> SearchProvider:
    name = name or settings.literature.get("default_provider", "europepmc")
    if name not in PROVIDERS:
        raise KeyError(f"Unknown literature provider: {name} (have {list(PROVIDERS)})")
    return PROVIDERS[name]
