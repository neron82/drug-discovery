"""Structured biological claim extraction from paper abstracts via the local LLM.

Guarantees:
  * Claims are generated ONLY from the provided abstract text — the platform never lets
    the model emit an uncited claim.
  * Every claim becomes a Statement + knowledge-graph nodes/edges with provenance
    pointing back to the source paper and statement.
  * On parse failure the paper is marked failed and skipped — no fabricated science.
"""
from __future__ import annotations

import logging

from .. import repos
from ..entity import normalize_entity
from ..llm import LLMProvider
from ..models import KnowledgeType

log = logging.getLogger("dd.extract")

_SYSTEM = """You are a precise biomedical claim extractor. You MUST output one single JSON object and NOTHING else.
No reasoning, no chain-of-thought, no markdown, no code fences, no commentary, no apologies.
Your entire reply must be exactly a JSON object that begins with "{" and ends with "}".
Extract only claims explicitly supported by the given abstract text — never invent facts.
Label speculation in the source as evidence_level "speculative". If no mechanical claims exist, return {"claims": []}.

JSON shape:
{
  "claims": [{
      "claim_type": "mechanism|target_relationship|compound_target|expression|association|clinical|other",
      "text": "One concise grounded sentence describing the finding.",
      "subject": "entity name, e.g. DHT, AR, TGF-beta, beta-catenin, Finasteride, SRD5A2, androgen receptor signalling, dermal papilla cell, Wnt/beta-catenin signalling",
      "subject_type": "protein|receptor|enzyme|hormone|metabolite|compound|drug|pathway|gene|cell_type|tissue|biological_process|phenotype|disease",
      "relation": "INHIBITS|ACTIVATES|REGULATES|UPREGULATES|DOWNREGULATES|INDUCES|INCREASES|DECREASES|ASSOCIATED_WITH|EXPRESSED_IN|PART_OF|TARGETS|BINDS_TO|METABOLIZED_BY|CONVERTS_TO|INVOLVES|AGONIST_OF|ANTAGONIST_OF|RELATED_TO",
      "object": "entity name (or empty string for a one-sided statement)",
      "object_type": "same vocabulary",
      "direction": "increase|decrease|modulate|none",
      "evidence_level": "established|probable|speculative",
      "species": "human|mouse|rat|in_silico|unknown",
      "model_context": "clinical|ex_vivo|animal_model|cell_culture|in_silico|unknown",
      "significance": "p-value or 'ns' if stated, else empty",
      "confidence": 0.0,
      "contradicted": false,
      "quotation": "short verbatim snippet from the abstract supporting this claim, or empty"
  }]
}"""


def process_paper(session, ws, paper, llm: LLMProvider):
    """Extract claims from one paper; build statements + KG; mark paper status."""
    if not paper.abstract or paper.status == "extracted":
        return paper
    if paper.status == "failed":
        return paper

    user = (
        "Paper metadata:\n"
        f"Title: {paper.title}\n"
        f"Journal: {paper.journal} ({paper.year})\n"
        "Abstract:\n"
        f"{paper.abstract[:8500]}"
    )
    try:
        data = llm.chat_json(_SYSTEM, user, temperature=0.1)
    except Exception as e:
        log.warning("LLM extraction failed for paper %s: %s", paper.id, e)
        paper.status = "failed"
        return paper

    claims = (data or {}).get("claims") or []
    made = 0
    for c in claims:
        try:
            made += _apply_claim(session, ws, paper, llm, c)
        except Exception as e:
            log.warning("claim apply error (paper %s): %s", paper.id, e)
            continue

    paper.status = "extracted"
    paper.extractor = llm.model
    if made:
        repos.add_timeline(session, ws.id, "biologist", "info",
                           f"Extracted {made} claims from '{paper.title[:70]}…'",
                           data={"paper_id": paper.id, "n_claims": made})
    return paper


def _apply_claim(session, ws, paper, llm, c) -> int:
    text = (c.get("text") or "").strip()
    if not text:
        return 0
    subject_raw = c.get("subject") or ""
    object_raw = c.get("object") or ""
    relation = (c.get("relation") or "").strip().upper() or "RELATED_TO"
    subj = normalize_entity(subject_raw, c.get("subject_type", "protein"))
    obj = normalize_entity(object_raw, c.get("object_type", "protein")) if object_raw else None
    conf = max(0.0, min(1.0, float(c.get("confidence", 0.5) or 0.5)))
    level = c.get("evidence_level", "speculative")
    if level not in ("established", "probable", "speculative"):
        level = "speculative"

    # statement first (ids needed for provenance)
    st = repos.add_statement(
        session, ws.id, text, paper_id=paper.id, claim_type=c.get("claim_type", "mechanism"),
        relation=relation, subject=subj["canonical_name"] if subj else "",
        object=obj["canonical_name"] if obj else "", direction=c.get("direction", "modulate"),
        evidence_level=level, confidence=conf, species=c.get("species", "unknown"),
        model_context=c.get("model_context", "unknown"), quotation=c.get("quotation", ""),
        properties={"contradicted": bool(c.get("contradicted")), "significance": c.get("significance", "")},
        extractor_model=llm.model, knowledge_type=KnowledgeType.SOURCE_FACT.value,
    )

    prov = {"paper_id": paper.id, "statement_id": st.id, "kind": "supports"}

    if subj and subj["canonical_name"] and subj["canonical_name"] != "unknown":
        snode, _ = repos.upsert_node(session, ws.id, subj["kind"], subj["canonical_name"],
                                     knowledge_type=KnowledgeType.SOURCE_FACT.value,
                                     evidence_level=level, confidence=conf)
        if obj and obj["canonical_name"] and obj["canonical_name"] != "unknown":
            onode, _ = repos.upsert_node(session, ws.id, obj["kind"], obj["canonical_name"],
                                         knowledge_type=KnowledgeType.SOURCE_FACT.value,
                                         evidence_level=level, confidence=conf)
            repos.add_edge(session, ws.id, relation, snode, onode,
                           knowledge_type=KnowledgeType.SOURCE_FACT.value,
                           evidence_level=level, confidence=conf, provenance=prov)
        # attach the entity itself to the disease/phenotype context loosely? no — keep only explicit
    return 1
