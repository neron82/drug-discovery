"""Research Director — the planning brain that decides WHAT to investigate next.

Scoring: Priority = EIG * importance * confidence_impact / cost (weights configurable).
It converts knowledge-graph gaps into a deduplicated, prioritized experiment queue.
Deterministic + transparent (no hidden logic), so autonomy is auditable.
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from . import repos
from .config import settings
from .models import Experiment, Hypothesis, JobStatus, Node, Paper, Statement
from sqlalchemy import func, select

log = logging.getLogger("dd.planner")

KINDS = [
    "literature_search", "paper_extraction", "contradiction_search",
    "hypothesis_generation", "question_generation",
]


def compute_priority(weights, eig=0.0, importance=0.0, confimpact=0.0, cost=1.0):
    cost = max(0.05, float(cost))
    return round((weights.get("expected_information_gain", 1) * eig
                  * weights.get("scientific_importance", 1) * importance
                  * weights.get("confidence_impact", 1) * confimpact)
                 / cost, 4)


def _already_queued(session, ws_id, kind, params_key):
    rows = session.execute(select(Experiment).where(
        Experiment.workspace_id == ws_id, Experiment.kind == kind,
        Experiment.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
    )).scalars().all()
    for r in rows:
        if _params_key(r.params_json or {}) == params_key:
            return True
    return False


def _params_key(p: dict) -> str:
    return str(sorted((k, str(v)) for k, v in p.items()))


def paper_counts(session, ws_id):
    total = session.execute(select(Paper).where(Paper.workspace_id == ws_id)).scalars().all()
    by_status = {}
    for p in total:
        by_status[p.status] = by_status.get(p.status, 0) + 1
    return total, by_status


def ensure_pipeline(session, ws, weights=None) -> list[Experiment]:
    """Create missing, deduplicated, prioritized experiments for current KG gaps."""
    weights = weights or settings.research.default_weights
    ws_id = ws.id
    created: list[Experiment] = []
    cfg = ws.config_json or {}
    seed_queries = cfg.get("seed_queries", [])
    state = ws.state_json or {}

    papers, by_status = paper_counts(session, ws_id)
    n_papers = len(papers)
    n_extracted = by_status.get("extracted", 0)
    n_targets = session.execute(select(Node).where(
        Node.workspace_id == ws_id, Node.kind.in_(["receptor", "enzyme", "protein"]))).scalars().all()
    n_hyp = session.execute(select(func.count()).select_from(Hypothesis)
                            .where(Hypothesis.workspace_id == ws_id)).scalar_one()

    # 1) seed literature searches if the KG is empty
    if n_papers == 0 and seed_queries:
        for q in seed_queries:
            pk = _params_key({"query": q, "size": 10, "provider": settings.literature.get("default_provider", "europepmc")})
            if _already_queued(session, ws_id, "literature_search", pk):
                continue
            created.append(repos.create_experiment(
                session, ws_id, "literature_search", question=f"Seed survey: {q}",
                reason="Bootstrap knowledge graph from real literature",
                expected_information_gain=0.9, scientific_importance=0.8, confidence_impact=0.5,
                computational_cost=1.0, priority=compute_priority(weights, 0.9, 0.8, 0.5, 1.0),
                params={"query": q, "size": 10, "provider": settings.literature.get("default_provider", "europepmc")},
                worker="director"))

    # 2) extract new papers
    to_extract = [p for p in papers if p.status == "fetched"][:10]
    for p in to_extract:
        pk = _params_key({"paper_id": p.id})
        if _already_queued(session, ws_id, "paper_extraction", pk):
            continue
        created.append(repos.create_experiment(
            session, ws_id, "paper_extraction", question=f"Extract claims: {p.title[:60]}…",
            reason="Convert raw abstract into structured KG claims",
            expected_information_gain=0.7, scientific_importance=0.6, confidence_impact=0.4,
            computational_cost=1.2, priority=compute_priority(weights, 0.7, 0.6, 0.4, 1.2),
            params={"paper_id": p.id}, worker="biologist"))

    # 3) hypothesis generation once there is evidence but few/zero hypotheses
    if n_extracted > 0 and n_hyp < 2:
        pk = _params_key({"limit": 3})
        if not _already_queued(session, ws_id, "hypothesis_generation", pk):
            created.append(repos.create_experiment(
                session, ws_id, "hypothesis_generation", question="Generate evidence-grounded hypotheses",
                reason="Move from facts to testable mechanistic hypotheses",
                expected_information_gain=0.8, scientific_importance=0.9, confidence_impact=0.6,
                computational_cost=1.2, priority=compute_priority(weights, 0.8, 0.9, 0.6, 1.2),
                params={"limit": 3}, worker="director"))

    # 4) contradiction search if contradictions present or few targets explained
    n_contra = session.execute(select(Statement).where(Statement.workspace_id == ws_id)).scalars().all()
    contradicts = sum(1 for s in n_contra if s.properties.get("contradicted"))
    if contradicts > 0:
        pk = _params_key({"topic": "contradictions"})
        if not _already_queued(session, ws_id, "contradiction_search", pk):
            created.append(repos.create_experiment(
                session, ws_id, "contradiction_search",
                question="Search for contradicting/negative evidence",
                reason="Actively attempt to falsify current claims",
                expected_information_gain=0.6, scientific_importance=0.8, confidence_impact=0.7,
                computational_cost=1.0, priority=compute_priority(weights, 0.6, 0.8, 0.7, 1.0),
                params={"topic": "androgenetic alopecia negative"}, worker="skeptic"))

    # 5) open question generation
    if not state.get("open_questions"):
        pk = _params_key({})
        if not _already_queued(session, ws_id, "question_generation", pk):
            created.append(repos.create_experiment(
                session, ws_id, "question_generation", question=f"Propose next research questions for {ws.disease or ws.name}",
                reason="Define high-value unknowns that drive the next cycle",
                expected_information_gain=0.5, scientific_importance=0.7, confidence_impact=0.3,
                computational_cost=0.3, priority=compute_priority(weights, 0.5, 0.7, 0.3, 0.3),
                params={}, worker="director"))

    # 5b) chemical intelligence: measured chemical matter for evidence-backed targets
    _COMPOUND_EDGE_TYPES = ("INHIBITS", "AGONIST_OF", "ANTAGONIST_OF", "BINDS_TO", "TARGETS")
    _COMPOUND_KINDS = ("drug", "compound", "drug_class")
    chem_made = 0
    for t in n_targets:
        has_compound = any(
            e.type in _COMPOUND_EDGE_TYPES
            and (e.target.kind in _COMPOUND_KINDS or e.source.kind in _COMPOUND_KINDS)
            for e in list(t.out_edges) + list(t.in_edges)
        )
        if has_compound:
            continue
        pk = _params_key({"target_node_id": t.id})
        if _already_queued(session, ws_id, "compound_discovery", pk):
            continue
        created.append(repos.create_experiment(
            session, ws_id, "compound_discovery", question=f"Chemical matter for {t.canonical_name}",
            reason="Target lacking catalogued molecules — pull measured ChEMBL activity",
            expected_information_gain=0.8, scientific_importance=0.7, confidence_impact=0.6,
            computational_cost=0.6, priority=compute_priority(weights, 0.8, 0.7, 0.6, 0.6),
            params={"target_node_id": t.id, "max_n": 10}, worker="chemist"))
        chem_made += 1
        if chem_made >= 2:
            break

    # 6) sustain the loop: when the queue is empty, deepen coverage on ligand-less
    #    top targets and the first open question (hysteresis via state['targeted_searches']).
    queued_now = session.execute(select(func.count()).select_from(Experiment).where(
        Experiment.workspace_id == ws_id,
        Experiment.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]))).scalar_one()
    if queued_now == 0:
        targeted = state.get("targeted_searches", [])
        state = dict(state)
        changed = False
        # (a) ligand-less evidence-backed targets
        for t in n_targets:
            has_link = any(e.type in ("INHIBITS", "AGONIST_OF", "ANTAGONIST_OF", "BINDS_TO", "TARGETS")
                           and (e.target.kind in ("drug", "compound", "drug_class") or e.source.kind in ("drug", "compound", "drug_class"))
                           for e in list(t.out_edges) + list(t.in_edges))
            if not has_link:
                topic = f"{t.canonical_name} hair follicle mechanism"
                if topic in targeted:
                    continue
                created.append(repos.create_experiment(
                    session, ws_id, "literature_search",
                    question=f"Targeted expansion: {t.canonical_name}",
                    reason="Evidence-backed target lacks mapped chemical matter — search deeper",
                    expected_information_gain=0.7, scientific_importance=0.7,
                    confidence_impact=0.5, computational_cost=1.0,
                    priority=compute_priority(weights, 0.7, 0.7, 0.5, 1.0),
                    params={"query": f"{t.canonical_name} AND (hair follicle OR alopecia) AND (inhibitor OR antagonist OR ligand OR treatment)",
                            "size": 8, "provider": settings.literature.get("default_provider", "europepmc")},
                    worker="director"))
                targeted.append(topic)
                changed = True
                if len(created) >= 2:
                    break
        # (b) first open question
        if not changed:
            questions = state.get("open_questions") or []
            if questions:
                qtext = questions[0].get("q", "") if isinstance(questions[0], dict) else str(questions[0])
                topic = _topic_from_question(qtext)
                if topic and topic not in targeted:
                    created.append(repos.create_experiment(
                        session, ws_id, "literature_search",
                        question=f"Open question exploration: {qtext[:60]}",
                        reason="Pursue a defined open research question",
                        expected_information_gain=0.6, scientific_importance=0.8,
                        confidence_impact=0.3, computational_cost=1.0,
                        priority=compute_priority(weights, 0.6, 0.8, 0.3, 1.0),
                        params={"query": topic, "size": 8,
                                "provider": settings.literature.get("default_provider", "europepmc")},
                        worker="director"))
                    targeted.append(topic)
                    changed = True
        if changed:
            state["targeted_searches"] = targeted
            ws.state_json = state

    return created


def _topic_from_question(q: str) -> str:
    """Derive a searchable topic from an open question (strip the interrogative)."""
    import re
    z = re.sub(r"(^|\s)(Is|Which|What|How|Does|Do|Can|Are|Why|When)(\s|$)", " ", q)
    z = re.sub(r"\?.*$", "", z)
    z = z.replace("therapeutically targetable", "therapeutic target").strip()
    z = re.sub(r"\s+", " ", z).strip(" ,.:")
    return z[:90] if len(z) > 8 else ""


def choose_next(session, ws_id) -> Experiment | None:
    return repos.next_queued(session, ws_id)
