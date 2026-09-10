"""Repository layer — all persistence operations, knowledge-graph upserts with provenance.

Every graph mutation accepts explicit knowledge_type / evidence_level / confidence and
attaches provenance (paper+statement). Nothing uncited becomes silent fact.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import DateTime, String, func, select, text
from sqlalchemy.orm import Session
from sqlalchemy.sql import bindparam, column

from .models import (
    Compound, Edge, Experiment, Hypothesis, JobStatus, KnowledgeType, Node, Paper, Statement,
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


def recover_stale_experiments(session: Session, *, stale_after_minutes: int,
                               workspace_id: str | None = None) -> list[Experiment]:
    """Requeue jobs left RUNNING by a crashed or restarted worker.

    The experiment row remains the audit record: recovery metadata is appended to
    ``result_json`` and the previous worker/error are preserved there. Only jobs
    older than the lease are touched, so a legitimately long docking/MD run is
    not stolen by a second worker.
    """
    cutoff = now() - timedelta(minutes=max(1, stale_after_minutes))
    query = select(Experiment).where(
        Experiment.status == JobStatus.RUNNING.value,
        Experiment.started_at.is_not(None),
    )
    if workspace_id:
        query = query.where(Experiment.workspace_id == workspace_id)
    candidates = session.execute(query.order_by(Experiment.started_at.asc())).scalars().all()
    # SQLite returns timezone-aware DateTime columns as naive values. Compare in
    # Python after normalising them, rather than relying on dialect-specific SQL
    # string coercion (which can make a fresh lease look stale).
    stale = []
    for ex in candidates:
        started = ex.started_at
        if started is not None and started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if started is not None and started < cutoff:
            stale.append(ex)
    recovered_at = now().isoformat()
    for ex in stale:
        lifecycle = dict(ex.result_json or {})
        lifecycle["recovery"] = {
            "recovered_at": recovered_at,
            "reason": "stale running lease; worker likely crashed or was restarted",
            "previous_worker": ex.worker,
            "previous_started_at": ex.started_at.isoformat() if ex.started_at else None,
            "attempt": int(lifecycle.get("recovery", {}).get("attempt", 0)) + 1,
        }
        ex.result_json = lifecycle
        ex.status = JobStatus.QUEUED.value
        ex.started_at = None
        ex.finished_at = None
        ex.worker = ex.worker or "recovered"
        ex.error = "Requeued after stale worker lease; see result_json.recovery"
        ex.result_summary = "Requeued after stale worker lease"
    if stale:
        log.warning("Recovered %d stale running experiment(s)%s", len(stale),
                    f" in workspace {workspace_id}" if workspace_id else "")
    return stale


# ---------------------------------------------------------------------------
# Bounded autonomy: retries, cancellation, metrics, drain lease
# ---------------------------------------------------------------------------
def params_key(p: Optional[dict]) -> str:
    """Canonical, order-independent fingerprint of experiment params.

    ONE implementation feeds every dedup/retry guard and every stored
    params dict — a separately computed key can drift from stored params
    and silently defeat dedup (see dbea0c4 / 9b101e4).
    """
    return str(sorted((k, str(v)) for k, v in (p or {}).items()))


def count_failures(session: Session, ws_id: str, kind: str, pkey: str) -> int:
    """Number of FAILED attempts for one (kind, params) identity."""
    rows = session.execute(select(Experiment).where(
        Experiment.workspace_id == ws_id, Experiment.kind == kind,
        Experiment.status == JobStatus.FAILED.value,
    )).scalars().all()
    return sum(1 for r in rows if params_key(r.params_json or {}) == pkey)


def cancel_queued_experiments(session: Session, ws_id: str, reason: str) -> int:
    """Cancel all queued (not yet running) experiments for a workspace.

    Explicit stop semantics: queued work is discarded with provenance;
    history is preserved (status=cancelled, never deleted). Running work
    is left to finish — it is bounded by the stale-running lease.
    """
    rows = session.execute(select(Experiment).where(
        Experiment.workspace_id == ws_id,
        Experiment.status == JobStatus.QUEUED.value,
    )).scalars().all()
    for ex in rows:
        lifecycle = dict(ex.result_json or {})
        lifecycle["cancellation"] = {"at": now().isoformat(), "reason": reason}
        ex.result_json = lifecycle
        ex.status = JobStatus.CANCELLED.value
        ex.error = f"Cancelled: {reason}"
        ex.result_summary = f"Cancelled: {reason}"
        ex.finished_at = now()
    if rows:
        log.info("Cancelled %d queued experiment(s) in workspace %s", len(rows), ws_id)
    return len(rows)


def queue_metrics(session: Session, ws_id: str) -> dict:
    """Honest queue accounting: by status, by kind, and lifecycle totals.

    Distinguishes what was created vs completed vs failed vs cancelled vs
    skipped, attributes queue growth (newly scheduled in the last hour),
    separates NEW DISCOVERY from RETRY (maintenance re-runs of a previously
    failed identity), and reports the live drain-lease holder — so queue
    behaviour can be attributed rather than guessed at.
    """
    rows = session.execute(select(Experiment).where(
        Experiment.workspace_id == ws_id)).scalars().all()
    by_status: dict[str, int] = {}
    pending_by_kind: dict[str, int] = {}
    failed_identities = set()
    for ex in rows:
        by_status[ex.status] = by_status.get(ex.status, 0) + 1
        if ex.status == JobStatus.FAILED.value:
            failed_identities.add((ex.kind, params_key(ex.params_json or {})))
    now_dt = now()
    newly = 0
    discovery = 0
    retry = 0
    for ex in rows:
        created = ex.created_at
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created is not None and now_dt - created <= timedelta(hours=1):
            newly += 1
        if ex.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            pending_by_kind[ex.kind] = pending_by_kind.get(ex.kind, 0) + 1
            # retry = a FAILED attempt with the identical (kind, params)
            # already exists; everything else pending is new discovery.
            if (ex.kind, params_key(ex.params_json or {})) in failed_identities:
                retry += 1
            else:
                discovery += 1
    pending = by_status.get(JobStatus.QUEUED.value, 0) + by_status.get(JobStatus.RUNNING.value, 0)
    return {
        "total": len(rows),
        "pending": pending,
        "by_status": by_status,
        "pending_by_kind": pending_by_kind,
        "created": len(rows),
        "completed": by_status.get(JobStatus.DONE.value, 0),
        "failed": by_status.get(JobStatus.FAILED.value, 0),
        "cancelled": by_status.get(JobStatus.CANCELLED.value, 0),
        "skipped": by_status.get(JobStatus.SKIPPED.value, 0),
        "newly_scheduled_last_hour": newly,
        "pending_discovery": discovery,
        "pending_retry": retry,
        "lease_holder": drain_lease_holder(session, ws_id),
    }


# Atomic single-drain lease. The claim is a conditional upsert: INSERT wins
# on an empty lease; ON CONFLICT DO UPDATE wins only when the existing lease
# is expired or already held by the same holder. SQLite serialises the
# statement, so two processes racing on an empty/expired lease cannot both
# win. A dead holder cannot wedge the workspace — its lease expires and the
# next claimant takes over.
_DRAIN_CLAIM = text("""
    INSERT INTO drain_leases (workspace_id, holder, acquired_at, expires_at)
    VALUES (:ws_id, :holder, :acquired, :expires)
    ON CONFLICT (workspace_id) DO UPDATE SET
        holder = excluded.holder,
        acquired_at = excluded.acquired_at,
        expires_at = excluded.expires_at
    WHERE drain_leases.expires_at < excluded.acquired_at
       OR drain_leases.holder = excluded.holder
""").bindparams(
    bindparam("acquired", type_=DateTime(timezone=True)),
    bindparam("expires", type_=DateTime(timezone=True)),
)
_DRAIN_RELEASE = text(
    "DELETE FROM drain_leases WHERE workspace_id = :ws_id AND holder = :holder"
)
_DRAIN_SELECT = text(
    "SELECT holder, expires_at FROM drain_leases WHERE workspace_id = :ws_id"
).columns(
    column("holder", String),
    column("expires_at", DateTime(timezone=True)),
)


def acquire_drain_lease(session: Session, ws: Workspace, holder: str,
                        lease_seconds: int) -> bool:
    """Atomically claim the single-drain lease for a workspace.

    Returns True if the lease is now held by ``holder``. Callers must run
    this inside the same transaction as the cycle it guards, so a rolled-
    back cycle also rolls back its claim.
    """
    now_dt = now()
    expires_dt = now_dt + timedelta(seconds=max(30, lease_seconds))
    res = session.execute(_DRAIN_CLAIM, {
        "ws_id": ws.id, "holder": holder,
        "acquired": now_dt, "expires": expires_dt,
    })
    return res.rowcount > 0


def release_drain_lease(session: Session, ws: Workspace, holder: str) -> None:
    """Release the drain lease if (and only if) we hold it."""
    session.execute(_DRAIN_RELEASE, {"ws_id": ws.id, "holder": holder})


def _aware_utc(v) -> Optional[datetime]:
    """Normalise a possibly-naive datetime or ISO string to an aware UTC datetime.

    Defensive: typed result columns yield datetimes, but a raw/untyped read
    yields the stored ISO string — both must parse without crashing.
    """
    if v is None:
        return None
    if isinstance(v, str):
        v = datetime.fromisoformat(v)
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v


def drain_lease_holder(session: Session, ws_id: str) -> Optional[str]:
    """Current live lease holder, or None if the lease is free/expired."""
    row = session.execute(_DRAIN_SELECT, {"ws_id": ws_id}).mappings().first()
    if not row:
        return None
    expires = _aware_utc(row["expires_at"])
    if expires is None or now() >= expires:
        return None
    return row["holder"]


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


# ---------------------------------------------------------------------------
# Compound catalogue (Phase 2)
# ---------------------------------------------------------------------------
def upsert_compound(session: Session, *, name: str, canonical_smiles: str = "",
                    inchikey: str = "", synonyms: list[str] | None = None,
                    sources: dict | None = None, properties: dict | None = None,
                    research_status: str = "catalogued") -> tuple[Compound, bool]:
    """Insert or update a compound in the global catalogue.

    Identity is structure-aware: primary key of dedup = InChIKey (computed
    client-side by chem.engine). A compound is re-catalogued (matched by
    inchikey) rather than duplicated. Returns (compound, created: bool).
    """
    if inchikey:
        obj = session.execute(select(Compound).where(Compound.inchikey == inchikey)).scalar_one_or_none()
        if obj is not None:
            obj.canonical_smiles = canonical_smiles or obj.canonical_smiles
            if name and name not in (obj.canonical_name,):
                if obj.synonyms is None:
                    obj.synonyms = []
                obj.synonyms = _push_unique(obj.synonyms, name)
            obj.properties_json = {**(obj.properties_json or {}), **(properties or {})}
            obj.sources_json = {**(obj.sources_json or {}), **(sources or {})}
            return obj, False
    # new compound
    fp = properties.get("fingerprint_hex") if properties else None
    if not fp and canonical_smiles:
        from .chem import morgan_fp_hex
        fp = morgan_fp_hex(canonical_smiles)
    props = {**(properties or {}), "fingerprint_hex": fp} if fp else (properties or {})
    obj = Compound(
        canonical_name=name or canonical_smiles[:80] or "unnamed",
        synonyms=synonyms or [], canonical_smiles=canonical_smiles, inchikey=inchikey,
        formula=props.get("formula", ""), molmass=props.get("molmass"),
        logp=props.get("logp"), tpsa=props.get("tpsa"),
        hbd=props.get("hbd"), hba=props.get("hba"), rotb=props.get("rotb"),
        sources_json=sources or {}, properties_json=props, research_status=research_status,
    )
    session.add(obj)
    session.flush()
    return obj, True


def _push_unique(lst: list, x) -> list:
    return [*lst, x] if x not in lst else lst


def list_compounds(session: Session, *, with_fp: bool = False) -> list[Compound]:
    cols = [Compound.id, Compound.canonical_name, Compound.canonical_smiles,
            Compound.formula, Compound.molmass, Compound.logp, Compound.tpsa,
            Compound.hbd, Compound.hba, Compound.rotb, Compound.research_status,
            Compound.created_at, Compound.properties_json, Compound.sources_json]
    rows = session.execute(select(*cols).order_by(Compound.canonical_name)).all()
    out = []
    for r in rows:
        d = {
            "id": r[0], "name": r[1], "smiles": r[2], "formula": r[3], "molmass": r[4],
            "logp": r[5], "tpsa": r[6], "hbd": r[7], "hba": r[8], "rotb": r[9],
            "research_status": r[10], "created_at": r[11].isoformat() if r[11] else None,
            "inchikey": (r[12] or {}).get("inchikey", ""),
            "sources": r[13] or {},
        }
        if with_fp:
            d["fingerprint_hex"] = (r[12] or {}).get("fingerprint_hex")
        out.append(d)
    return out


def get_compound(session: Session, cid: str) -> dict | None:
    c = session.get(Compound, cid)
    if c is None:
        return None
    return {
        "id": c.id, "name": c.canonical_name, "synonyms": c.synonyms or [],
        "smiles": c.canonical_smiles, "inchi": c.inchi or "", "inchikey": c.inchikey,
        "formula": c.formula, "molmass": c.molmass, "logp": c.logp, "tpsa": c.tpsa,
        "hbd": c.hbd, "hba": c.hba, "rotb": c.rotb,
        "sources": c.sources_json or {}, "properties": c.properties_json or {},
        "research_status": c.research_status, "created_at": c.created_at.isoformat() if c.created_at else None,
    }


def similarity_search(session: Session, fp_hex: str, limit: int = 20) -> list[dict]:
    """Rank the catalogue by Morgan/Tanimoto similarity to fp_hex."""
    from .chem import tanimoto
    scored = []
    for row in list_compounds(session, with_fp=True):
        theirs = row.get("fingerprint_hex")
        if not theirs:
            continue
        s = tanimoto(fp_hex, theirs)
        if s > 0.0:
            scored.append((s, row))
    scored.sort(key=lambda t: -t[0])
    return [{"similarity": s, **{k: row[k] for k in ("id", "name", "smiles", "formula", "molmass", "logp", "tpsa", "inchikey")}}
            for s, row in scored[:limit]]


def compound_node_edges(session: Session, ws_id: str, cid: str) -> dict:
    """Return the KG node for a compound (kind=compound) plus its edges in a workspace."""
    node = None
    for n in session.execute(select(Node).where(Node.workspace_id == ws_id, Node.kind == "compound")).scalars().all():
        if (n.properties or {}).get("compound_id") == cid:
            node = n
            break
    if node is None:
        return {"node": None, "edges": []}
    edges = session.execute(select(Edge).where(
        Edge.workspace_id == ws_id,
        (Edge.source_id == node.id) | (Edge.target_id == node.id),
    )).scalars().all()
    return {"node": node, "edges": edges}
