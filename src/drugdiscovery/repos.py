"""Repository layer — all persistence operations, knowledge-graph upserts with provenance.

Every graph mutation accepts explicit knowledge_type / evidence_level / confidence and
attaches provenance (paper+statement). Nothing uncited becomes silent fact.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import (
    Edge, Experiment, Hypothesis, JobStatus, KnowledgeType, Node, Paper, Statement,
    TimelineEntry, Workspace,
)

log = logging.getLogger("dd.repos")


def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------
def create_workspace(session: Session, name: str, disease: str = "", description: str = "",
                     config: Optional[dict] = None) -> Workspace:
    slug = name.strip().lower().replace(" ", "-").replace("/", "-")
    base, i = slug, 1
    while session.execute(select(Workspace).where(Workspace.slug == slug)).scalar_one_or_none():
        slug = f"{base}-{i}"
        i += 1
    ws = Workspace(name=name, slug=slug, disease=disease, description=description,
                   config_json=config or {}, state_json=_init_state())
    session.add(ws)
    session.flush()
    session.add(TimelineEntry(workspace_id=ws.id, agent="director", level="info",
                              message=f"Workspace '{name}' created for disease: {disease or 'unspecified'}"))
    return ws


def _init_state() -> dict:
    return {"open_questions": [], "research_priorities": {}, "counters": {"cycles": 0}}


def get_workspace(session: Session, ws_id: str) -> Optional[Workspace]:
    return session.get(Workspace, ws_id)


def list_workspaces(session: Session) -> list[Workspace]:
    return list(session.execute(select(Workspace).order_by(Workspace.created_at.desc())).scalars())


def set_research_enabled(session: Session, ws: Workspace, enabled: bool) -> None:
    ws.research_enabled = enabled
    ws.research_state = "running" if enabled else ("paused" if ws.research_state == "running" else ws.research_state)


# ---------------------------------------------------------------------------
# Knowledge graph nodes
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def find_node(session: Session, ws_id: str, kind: str, name: str) -> Optional[Node]:
    """Match node by canonical name (case-insensitive) or synonym."""
    n = _norm(name)
    row = session.execute(select(Node).where(
        Node.workspace_id == ws_id, Node.kind == kind,
        func.lower(Node.canonical_name) == n,
    )).scalar_one_or_none()
    if row:
        return row
    for node in session.execute(select(Node).where(Node.workspace_id == ws_id, Node.kind == kind)).scalars():
        if any(_norm(s) == n for s in (node.synonyms or [])):
            return node
    return None


def upsert_node(session: Session, ws_id: str, kind: str, name: str,
                synonyms: Optional[list] = None, properties: Optional[dict] = None,
                knowledge_type: str = KnowledgeType.SOURCE_FACT.value,
                evidence_level: str = "speculative", confidence: float = 0.0,
                label: str = "", summary: str = "") -> tuple[Node, bool]:
    synonyms = list(dict.fromkeys([s for s in (synonyms or []) if s])) if synonyms else []
    node = find_node(session, ws_id, kind, name)
    if node is None:
        node = Node(workspace_id=ws_id, kind=kind, canonical_name=name.strip(), label=label or name.strip(),
                    synonyms=synonyms, properties=properties or {}, knowledge_type=knowledge_type,
                    evidence_level=evidence_level, confidence=confidence, summary=summary)
        session.add(node)
        session.flush()
        return node, True
    # merge synonyms / bump confidence to max
    for s in synonyms:
        if s not in (node.synonyms or []):
            node.synonyms = (node.synonyms or []) + [s]
    node.confidence = max(node.confidence or 0.0, confidence)
    if confidence >= (node.confidence or 0.0) - 1e-9 and confidence > 0:
        node.evidence_level = evidence_level
    if summary and not node.summary:
        node.summary = summary
    return node, False


# ---------------------------------------------------------------------------
# Edges with provenance
# ---------------------------------------------------------------------------
def add_edge(session: Session, ws_id: str, edge_type: str, source: Node, target: Node,
             knowledge_type: str = KnowledgeType.SOURCE_FACT.value,
             evidence_level: str = "speculative", confidence: float = 0.0,
             provenance: Optional[dict] = None, directional: bool = True,
             properties: Optional[dict] = None) -> Edge:
    prov = provenance or {}
    existing = session.execute(select(Edge).where(
        Edge.workspace_id == ws_id, Edge.source_id == source.id, Edge.type == edge_type,
        Edge.target_id == target.id, Edge.knowledge_type == knowledge_type,
    )).scalar_one_or_none()
    if existing:
        if prov and prov not in (existing.provenance or []):
            existing.provenance = (existing.provenance or []) + [prov]
        existing.confidence = max(existing.confidence or 0.0, confidence)
        if confidence >= (existing.confidence or 0.0) - 1e-9:
            existing.evidence_level = evidence_level
            existing.knowledge_type = knowledge_type
        return existing
    edge = Edge(workspace_id=ws_id, source_id=source.id, target_id=target.id, type=edge_type,
                directional=directional, knowledge_type=knowledge_type, evidence_level=evidence_level,
                confidence=confidence, provenance=[prov] if prov else [], properties=properties or {})
    session.add(edge)
    session.flush()
    return edge


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
def upsert_paper(session: Session, ws_id: str, rec) -> tuple[Paper, bool]:
    # dedup by content hash (robust to missing pmid/doi)
    existing = session.execute(select(Paper).where(
        Paper.workspace_id == ws_id, Paper.content_hash == rec.content_hash)).scalar_one_or_none()
    if existing is None and rec.pmid:
        existing = session.execute(select(Paper).where(Paper.workspace_id == ws_id, Paper.pmid == rec.pmid)).scalar_one_or_none()
    if existing is None and rec.doi:
        existing = session.execute(select(Paper).where(Paper.workspace_id == ws_id, Paper.doi == rec.doi)).scalar_one_or_none()
    if existing:
        if not existing.abstract and rec.abstract:
            existing.abstract = rec.abstract
            existing.content_hash = rec.content_hash
            existing.status = "fetched"
        return existing, False
    paper = Paper(workspace_id=ws_id, pmid=rec.pmid or None, doi=rec.doi or None, title=rec.title,
                  abstract=rec.abstract, authors_json=rec.authors, journal=rec.journal, year=rec.year,
                  source=rec.source, url=rec.url, content_hash=rec.content_hash)
    session.add(paper)
    session.flush()
    return paper, True


def paper_count(session: Session, ws_id: str) -> int:
    return session.execute(select(func.count()).select_from(Paper)
                           .where(Paper.workspace_id == ws_id)).scalar_one()


# ---------------------------------------------------------------------------
# Statements / claims
# ---------------------------------------------------------------------------
def add_statement(session: Session, ws_id: str, text: str, *, paper_id: Optional[str] = None,
                  claim_type: str = "mechanism", relation: str = "", subject: str = "", object: str = "",
                  direction: str = "modulate", evidence_level: str = "speculative", confidence: float = 0.0,
                  species: str = "unknown", model_context: str = "unknown", quotation: str = "",
                  properties: Optional[dict] = None, extractor_model: str = "",
                  knowledge_type: str = KnowledgeType.SOURCE_FACT.value) -> Statement:
    st = Statement(workspace_id=ws_id, source_paper_id=paper_id, text=text, claim_type=claim_type,
                   relation=relation, subject=subject, object=object, direction=direction,
                   evidence_level=evidence_level, confidence=confidence, species=species,
                   model_context=model_context, quotation=quotation, properties=properties or {},
                   extractor_model=extractor_model, knowledge_type=knowledge_type)
    session.add(st)
    session.flush()
    return st


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------
def add_hypothesis(session: Session, ws_id: str, *, title: str, body: str, mechanism_reasoning: str,
                   predicted_effect: str, supporting_evidence: Optional[list] = None,
                   counter_evidence: Optional[list] = None, uncertainty: str = "",
                   testable_predictions: Optional[list] = None, next_experiment: str = "",
                   status: str = "active", confidence: float = 0.0, target_id: str = "",
                   pathway_id: str = "", evidence_profile: Optional[dict] = None) -> Hypothesis:
    h = Hypothesis(workspace_id=ws_id, title=title, body=body, mechanism_reasoning=mechanism_reasoning,
                   predicted_effect=predicted_effect, supporting_evidence=supporting_evidence or [],
                   counter_evidence=counter_evidence or [], uncertainty=uncertainty,
                   testable_predictions=testable_predictions or [], next_experiment=next_experiment,
                   status=status, confidence=confidence, target_id=target_id or None,
                   pathway_id=pathway_id or None, evidence_profile=evidence_profile or {})
    session.add(h)
    session.flush()
    return h


# ---------------------------------------------------------------------------
# Experiments / queue
# ---------------------------------------------------------------------------
def create_experiment(session: Session, ws_id: str, kind: str, *, question: str = "", reason: str = "",
                      expected_information_gain: float = 0.0, scientific_importance: float = 0.0,
                      confidence_impact: float = 0.0, computational_cost: float = 1.0,
                      priority: float = 0.0, params: Optional[dict] = None,
                      dependencies: Optional[list] = None, worker: str = "director") -> Experiment:
    ex = Experiment(workspace_id=ws_id, kind=kind, question=question, reason=reason,
                    expected_information_gain=expected_information_gain, scientific_importance=scientific_importance,
                    confidence_impact=confidence_impact, computational_cost=computational_cost,
                    priority=priority, params_json=params or {}, dependencies=dependencies or [],
                    status=JobStatus.QUEUED.value, worker=worker)
    session.add(ex)
    session.flush()
    return ex


def next_queued(session: Session, ws_id: str) -> Optional[Experiment]:
    return session.execute(select(Experiment).where(
        Experiment.workspace_id == ws_id, Experiment.status == JobStatus.QUEUED.value,
        Experiment.dependencies == [],
    ).order_by(Experiment.priority.desc(), Experiment.created_at.asc())).scalars().first()


def mark_experiment(session: Session, ex: Experiment, status: str, result_summary: str = "",
                    result_json: Optional[dict] = None, error: str = "", worker: str = "") -> None:
    ex.status = status
    if result_summary:
        ex.result_summary = result_summary
    if result_json is not None:
        ex.result_json = result_json
    if error:
        ex.error = error
    if worker:
        ex.worker = worker
    if status in (JobStatus.RUNNING.value,):
        ex.started_at = now()
    elif status in (JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
        ex.finished_at = now()


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
def add_timeline(session: Session, ws_id: str, agent: str, level: str, message: str,
                 data: Optional[dict] = None) -> TimelineEntry:
    e = TimelineEntry(workspace_id=ws_id, agent=agent, level=level, message=message, data_json=data or {})
    session.add(e)
    session.flush()
    return e


# ---------------------------------------------------------------------------
# Graph projection for UI
# ---------------------------------------------------------------------------
def graph_projection(session: Session, ws_id: str) -> dict:
    nodes = session.execute(select(Node).where(Node.workspace_id == ws_id)).scalars().all()
    edges = session.execute(select(Edge).where(Edge.workspace_id == ws_id)).scalars().all()
    nl = [{"id": n.id, "label": n.label or n.canonical_name, "kind": n.kind,
           "evidence_level": n.evidence_level, "confidence": round(n.confidence, 3),
           "knowledge_type": n.knowledge_type} for n in nodes]
    el = [{"id": e.id, "source": e.source_id, "target": e.target_id, "type": e.type,
           "evidence_level": e.evidence_level, "confidence": round(e.confidence, 3),
           "knowledge_type": e.knowledge_type} for e in edges]
    return {"nodes": nl, "edges": el}
