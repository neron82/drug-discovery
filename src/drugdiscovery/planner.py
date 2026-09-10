"""Research Director — the planning brain that decides WHAT to investigate next.

Scoring: Priority = EIG * importance * confidence_impact / cost (weights configurable).
It converts knowledge-graph gaps into a deduplicated, prioritized experiment queue.
Deterministic + transparent (no hidden logic), so autonomy is auditable.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select

from . import repos
from .config import settings
from .models import Compound, Experiment, Hypothesis, JobStatus, Node, Paper, Statement, Structure
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


def _already_queued(session, ws_id, kind, params_key, terminal=False):
    statuses = [JobStatus.QUEUED.value, JobStatus.RUNNING.value]
    if terminal:
        statuses += [JobStatus.DONE.value]
    rows = session.execute(select(Experiment).where(
        Experiment.workspace_id == ws_id, Experiment.kind == kind,
        Experiment.status.in_(statuses),
    )).scalars().all()
    for r in rows:
        if _params_key(r.params_json or {}) == params_key:
            return True
    return False


def _focus_term(cfg: dict) -> str:
    """Primary tissue/phenotype focus for literature queries (config-driven)."""
    tis = cfg.get("tissues") or []
    phen = cfg.get("phenotype") or []
    terms = (tis if len(tis) >= len(phen) else phen)
    return terms[0] if terms else "disease mechanism"

def _focus_query(cfg: dict) -> str:
    """Config-driven body-region/phenotype filter for targeted searches."""
    tis = cfg.get("tissues") or []
    phen = cfg.get("phenotype") or []
    parts = [t for t in (tis + phen) if t]
    if not parts:
        return "(e.g. tissue OR cell-type)"
    return "(" + " OR ".join(parts) + ")"

def _params_key(p: dict) -> str:
    return repos.params_key(p)


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
    safety_cap = int(settings.research.cycle_safety_cap)

    def _create(*args, **kwargs):
        """Create at most cycle_safety_cap experiments in this planning pass,
        and never re-schedule an identity that already burned its retry
        budget (bounded autonomy: failures converge, they don't loop)."""
        if safety_cap > 0 and len(created) >= safety_cap:
            return None
        kind = args[1] if len(args) > 1 else kwargs.get("kind")
        params = kwargs.get("params") or {}
        n_fail = repos.count_failures(session, ws_id, kind, _params_key(params))
        retry_limit = int(settings.research.max_retries_per_kind)
        if retry_limit > 0 and n_fail >= retry_limit:
            log.info("WS %s: %s retry limit reached (%d failures), not re-scheduling",
                     ws_id, kind, n_fail)
            return None
        ex = repos.create_experiment(session, *args, **kwargs)
        created.append(ex)
        return ex

    pending = session.execute(select(func.count()).select_from(Experiment).where(
        Experiment.workspace_id == ws_id,
        Experiment.status.in_(["queued", "running"]),
    )).scalar_one()
    max_pending = int(settings.research.max_pending_experiments)
    if max_pending > 0 and pending >= max_pending:
        log.info("WS %s: planner backpressure pending=%d limit=%d", ws_id, pending, max_pending)
        return created

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
            _create(
                ws_id, "literature_search", question=f"Seed survey: {q}",
                reason="Bootstrap knowledge graph from real literature",
                expected_information_gain=0.9, scientific_importance=0.8, confidence_impact=0.5,
                computational_cost=1.0, priority=compute_priority(weights, 0.9, 0.8, 0.5, 1.0),
                params={"query": q, "size": 10, "provider": settings.literature.get("default_provider", "europepmc")},
                worker="director")

    # 2) extract new papers
    to_extract = [p for p in papers if p.status == "fetched"][:10]
    for p in to_extract:
        pk = _params_key({"paper_id": p.id})
        if _already_queued(session, ws_id, "paper_extraction", pk, terminal=True):
            continue
        _create(
            ws_id, "paper_extraction", question=f"Extract claims: {p.title[:60]}…",
            reason="Convert raw abstract into structured KG claims",
            expected_information_gain=0.7, scientific_importance=0.6, confidence_impact=0.4,
            computational_cost=1.2, priority=compute_priority(weights, 0.7, 0.6, 0.4, 1.2),
            params={"paper_id": p.id}, worker="biologist")

    # 3) hypothesis generation once there is evidence but few/zero hypotheses
    if n_extracted > 0 and n_hyp < 2:
        pk = _params_key({"limit": 3})
        if not _already_queued(session, ws_id, "hypothesis_generation", pk):
            _create(
                ws_id, "hypothesis_generation", question="Generate evidence-grounded hypotheses",
                reason="Move from facts to testable mechanistic hypotheses",
                expected_information_gain=0.8, scientific_importance=0.9, confidence_impact=0.6,
                computational_cost=1.2, priority=compute_priority(weights, 0.8, 0.9, 0.6, 1.2),
                params={"limit": 3}, worker="director")

    # 4) contradiction search if contradictions present or few targets explained
    n_contra = session.execute(select(Statement).where(Statement.workspace_id == ws_id)).scalars().all()
    contradicts = sum(1 for s in n_contra if s.properties.get("contradicted"))
    if contradicts > 0:
        pk = _params_key({"topic": "contradictions"})
        if not _already_queued(session, ws_id, "contradiction_search", pk):
            _create(
                ws_id, "contradiction_search",
                question="Search for contradicting/negative evidence",
                reason="Actively attempt to falsify current claims",
                expected_information_gain=0.6, scientific_importance=0.8, confidence_impact=0.7,
                computational_cost=1.0, priority=compute_priority(weights, 0.6, 0.8, 0.7, 1.0),
                params={"topic": f"{ws.disease or ws.name} negative"}, worker="skeptic")

    # 5) open question generation
    if not state.get("open_questions"):
        pk = _params_key({})
        if not _already_queued(session, ws_id, "question_generation", pk):
            _create(
                ws_id, "question_generation", question=f"Propose next research questions for {ws.disease or ws.name}",
                reason="Define high-value unknowns that drive the next cycle",
                expected_information_gain=0.5, scientific_importance=0.7, confidence_impact=0.3,
                computational_cost=0.3, priority=compute_priority(weights, 0.5, 0.7, 0.3, 0.3),
                params={}, worker="director")

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
        pk = _params_key({"target_node_id": t.id, "max_n": 10})
        if _already_queued(session, ws_id, "compound_discovery", pk, terminal=True):
            continue
        _create(
            ws_id, "compound_discovery", question=f"Chemical matter for {t.canonical_name}",
            reason="Target lacking catalogued molecules — pull measured ChEMBL activity",
            expected_information_gain=0.8, scientific_importance=0.7, confidence_impact=0.6,
            computational_cost=0.6, priority=compute_priority(weights, 0.8, 0.7, 0.6, 0.6),
            params={"target_node_id": t.id, "max_n": 10}, worker="chemist")
        chem_made += 1
        if chem_made >= 2:
            break

    # 5c) structural intelligence: experimental structures for evidence-backed targets
    struct_made = 0
    for t in n_targets:
        if (t.properties or {}).get("structural", {}).get("acquired"):
            continue
        pk = _params_key({"node_id": t.id})
        if _already_queued(session, ws_id, "structure_analysis", pk, terminal=True):
            continue
        _create(
            ws_id, "structure_analysis", question=f"Structural coverage for {t.canonical_name}",
            reason="Target lacks acquired structures — resolve UniProt/PDB/AlphaFold",
            expected_information_gain=0.7, scientific_importance=0.7, confidence_impact=0.4,
            computational_cost=0.4, priority=compute_priority(weights, 0.7, 0.7, 0.4, 0.4),
            params={"node_id": t.id}, worker="structural_biologist")
        struct_made += 1
        if struct_made >= 2:
            break

    # 5d) docking: compounds vs structurally-covered targets with a crystallised
    #     ligand pocket. Scheduling at most a small batch per pipeline pass.
    dock_made = 0
    compounds = list(session.execute(select(Compound).order_by(Compound.created_at.asc()).limit(10)).scalars())
    dockable = []
    for t in n_targets:
        # target has acquired structures, ideally with a co-crystal ligand
        acq = (t.properties or {}).get("structural", {})
        if not acq.get("acquired"):
            continue
        targets_for = [e.target for e in t.out_edges if e.target.kind in ("drug", "compound")]
        targets_for += [e.source for e in t.in_edges if e.source.kind in ("drug", "compound")]
        if not targets_for:
            continue
        # pick the best (earliest) acquired structure for this target
        structure = session.execute(
            select(Structure).where(Structure.workspace_id == ws_id,
                                    Structure.protein_node_id == t.id)
            .order_by(Structure.resolution.is_(None).asc(), Structure.resolution.asc())
            .limit(1)).scalar_one_or_none()
        if structure is None:
            continue
        # valid co-crystal ligand detection (must match runner.py: a ligand
        # entry with a usable comp_id) — malformed truthy metadata like [None]
        # must not bypass the blind-box guard.
        ligands = structure.ligands or []
        comp_ids = [l.get("comp_id") for l in ligands if isinstance(l, dict)]
        comp_ids = [c for c in comp_ids if c]
        blind = not comp_ids
        # HARD BOX GUARD (planner side): a whole-protein blind box on a large
        # structure exceeds MAX_BOX_EDGE and would be refused by DockingEngine
        # (and historically exhausted memory). Skip it here (fail fast, with
        # provenance) instead of queueing jobs that all fail. Fail CLOSED on
        # errors: persist skip provenance rather than queueing an unsafe job.
        if blind:
            fp = structure.file_path
            if fp and Path(fp).exists():
                try:
                    from .sim import pocket_blind
                    from .sim.engine import MAX_BOX_EDGE, DEFAULT_BOX_PADDING
                    pk = pocket_blind(fp)
                    edge = float(pk["size"]) + DEFAULT_BOX_PADDING
                    if edge > MAX_BOX_EDGE:
                        state = dict(ws.state_json or {})
                        skipped = dict(state.get("docking_skipped") or {})
                        skipped[structure.id] = {
                            "reason": "blind box edge exceeds safety limit",
                            "edge_A": round(edge, 1),
                            "max_edge_A": MAX_BOX_EDGE,
                        }
                        state["docking_skipped"] = skipped
                        ws.state_json = state
                        log.info("Skipping oversized blind docking structure %s (edge %.1f A > %d A)",
                                 structure.source_id, edge, MAX_BOX_EDGE)
                        continue
                except Exception as e:
                    # fail closed: persist skip provenance so we do NOT queue an
                    # unsafe job we could not verify.
                    state = dict(ws.state_json or {})
                    skipped = dict(state.get("docking_skipped") or {})
                    skipped[structure.id] = {
                        "reason": f"could not compute blind box: {e}",
                        "error": str(e)[:200],
                    }
                    state["docking_skipped"] = skipped
                    ws.state_json = state
                    log.warning("Failing closed on blind box for %s: %s", structure.source_id, e)
                    continue
            else:
                # fail closed: a blind structure with no readable file cannot be
                # verified for box size; persist skip provenance and do not queue.
                state = dict(ws.state_json or {})
                skipped = dict(state.get("docking_skipped") or {})
                skipped[structure.id] = {
                    "reason": "blind structure has no readable file_path; cannot verify box size",
                }
                state["docking_skipped"] = skipped
                ws.state_json = state
                log.warning("Failing closed on blind structure %s (no readable file_path)",
                            structure.source_id)
                continue
        # a compound whose targets include this target
        for c in compounds:
            # ONE params dict feeds both the dedup guard and create_experiment —
            # a guard key computed separately can drift from stored params and
            # silently defeat dedup (see compound_discovery max_n bug, dbea0c4).
            pdict = {"structure_id": structure.id, "compound_id": c.id,
                     "compound_smiles": c.canonical_smiles}
            if _already_queued(session, ws_id, "docking", _params_key(pdict), terminal=True):
                continue
            dockable.append({"structure": structure, "compound": c,
                             "target": t, "blind": blind})
            if len(dockable) >= 4:
                break
        if len(dockable) >= 4:
            break
    for d in dockable:
        is_blind = d["blind"]
        _create(
            ws_id, "docking",
            question=f"Virtual screening: {d['compound'].canonical_name} vs {d['target'].canonical_name}",
            reason=("Target structure has NO co-crystal ligand — blind whole-protein "
                    "box docking (site-agnostic, weaker evidence)" if is_blind else
                    "Target has a crystallised ligand pocket and related chemical matter — "
                    "docking predicts plausible binding geometry"),
            expected_information_gain=0.4 if is_blind else 0.6,
            scientific_importance=0.6, confidence_impact=0.3 if is_blind else 0.4,
            computational_cost=0.9,
            priority=compute_priority(weights, 0.4 if is_blind else 0.6, 0.6,
                                      0.3 if is_blind else 0.4, 0.9),
            params={"structure_id": d["structure"].id, "compound_id": d["compound"].id,
                    "compound_smiles": d["compound"].canonical_smiles},
            worker="simulation")
        dock_made += 1

    # 5e) skeptic pass: adversarial critique of the strongest active hypotheses.
    #     Deliberately cheap (one LLM call per hypothesis) and high information
    #     gain: falsification is what makes the knowledge base trustworthy.
    crit_h = list(session.execute(
        select(Hypothesis).where(
            Hypothesis.workspace_id == ws_id,
            Hypothesis.status.in_(["active", "promising"]),
            Hypothesis.confidence > 0.1)   # fully-weakened hypotheses stay refuted, no re-critique loop
        .order_by(Hypothesis.confidence.desc()).limit(3)).scalars())
    for h in crit_h:
        last = (h.evidence_profile or {}).get("last_critique", {})
        last_at = last.get("at", "") if isinstance(last, dict) else ""
        if last_at and h.updated_at and h.updated_at.isoformat() <= last_at:
            continue  # already critiqued after last update
        if _already_queued(session, ws_id, "hypothesis_critique", {"hypothesis_id": h.id}):
            continue
        _create(
            ws_id, "hypothesis_critique",
            question=f"Falsification attempt: {h.title[:80]}",
            reason="High-confidence hypothesis has not been adversarially reviewed yet — "
                   "the skeptic searches for alternative explanations and weaknesses",
            expected_information_gain=0.8, scientific_importance=0.8, confidence_impact=0.8,
            computational_cost=0.2,
            priority=compute_priority(weights, 0.8, 0.8, 0.8, 0.2),
            params={"hypothesis_id": h.id},
            worker="skeptic")

    # 5f) molecular dynamics: expensive secondary validation, gated to the
    # top-scoring docked structures (spec §11 — never applied indiscriminately).
    from .models import SimulationRun
    md_sel = session.execute(
        select(SimulationRun.structure_id, func.min(SimulationRun.score))
        .where(SimulationRun.kind == "docking", SimulationRun.status == "done")
        .group_by(SimulationRun.structure_id)
        .having(func.min(SimulationRun.score) <= -5.0)
        .order_by(func.min(SimulationRun.score).asc()).limit(2)).all()
    md_candidates = [(sid, round(score, 2)) for sid, score in md_sel if sid]
    md_max_atoms = 20_000  # CPU safety budget for explicit-solvent MD
    for sid, best in md_candidates:
        structure = session.get(Structure, sid)
        if structure and structure.file_path:
            try:
                atom_count = sum(
                    1 for line in Path(structure.file_path).read_text(errors="ignore").splitlines()
                    if line.startswith("ATOM  ")
                )
            except OSError:
                atom_count = 0
            if atom_count > md_max_atoms:
                state = dict(ws.state_json or {})
                skipped = dict(state.get("md_skipped") or {})
                skipped[sid] = {
                    "reason": "protein atom count exceeds CPU MD budget",
                    "atom_count": atom_count,
                    "max_protein_atoms": md_max_atoms,
                }
                state["md_skipped"] = skipped
                ws.state_json = state
                log.info("Skipping MD for %s: %d atoms > %d budget", sid, atom_count, md_max_atoms)
                continue
        if _already_queued(session, ws_id, "molecular_dynamics", {"structure_id": sid}, terminal=True):
            continue
        done_md = session.execute(select(func.count()).select_from(SimulationRun).where(
            SimulationRun.structure_id == sid,
            SimulationRun.kind == "md", SimulationRun.status == "done")).scalar_one()
        if done_md:
            continue
        _create(
            ws_id, "molecular_dynamics",
            question=f"Pose persistence / pocket thermal stability: {sid[:8]}",
            reason=f"Docking scored {best} kcal/mol — validate the pose and binding-"
                   "pocket stability under explicit-solvent dynamics, and record pocket "
                   "flexibility for candidate prioritisation",
            expected_information_gain=0.55, scientific_importance=0.7,
            confidence_impact=0.5, computational_cost=0.9,
            priority=compute_priority(weights, 0.55, 0.7, 0.5, 0.9),
            params={"structure_id": sid}, worker="simulation")

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
                topic = f"{t.canonical_name} {_focus_term(cfg)}"
                if topic in targeted:
                    continue
                made = _create(
                    ws_id, "literature_search",
                    question=f"Targeted expansion: {t.canonical_name}",
                    reason="Evidence-backed target lacks mapped chemical matter — search deeper",
                    expected_information_gain=0.7, scientific_importance=0.7,
                    confidence_impact=0.5, computational_cost=1.0,
                    priority=compute_priority(weights, 0.7, 0.7, 0.5, 1.0),
                    params={"query": f"{t.canonical_name} AND {_focus_query(cfg)} AND (inhibitor OR antagonist OR ligand OR treatment)",
                            "size": 8, "provider": settings.literature.get("default_provider", "europepmc")},
                    worker="director")
                if made is None:
                    break
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
                    made = _create(
                        ws_id, "literature_search",
                        question=f"Open question exploration: {qtext[:60]}",
                        reason="Pursue a defined open research question",
                        expected_information_gain=0.6, scientific_importance=0.8,
                        confidence_impact=0.3, computational_cost=1.0,
                        priority=compute_priority(weights, 0.6, 0.8, 0.3, 1.0),
                        params={"query": topic, "size": 8,
                                "provider": settings.literature.get("default_provider", "europepmc")},
                        worker="director")
                    if made is None:
                        return created
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
