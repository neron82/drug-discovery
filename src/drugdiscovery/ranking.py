"""Multi-factor candidate ranking (Phase 5, Spec section 31).

A candidate = (compound, target) pair. Instead of one opaque magic score we
assemble an explicit dimension vector, each component traceable to real data.
Every component is 0..1 and shown separately; the composite is a weighted mean
whose weights the caller may override. Nothing here claims biological activity -
it is a triage signal deciding what to simulate and scrutinise next.

Compound rows are global; KG wiring lives on Node(kind=compound) whose
properties carry {"compound_id": ...}. Edges link compound nodes to targets.
"""
from __future__ import annotations

from sqlalchemy import select

from .models import Compound, Node, SimulationRun, Structure

DEFAULT_WEIGHTS = {
    "mechanism_relevance": 0.25,
    "target_confidence": 0.20,
    "binding_evidence": 0.25,
    "structural_coverage": 0.10,
    "drug_likeness": 0.10,
    "chemical_provenance": 0.05,
    "uncertainty": 0.05,
}

TARGET_KINDS = ("receptor", "enzyme", "protein", "gene")


def score_to_binding(vina) -> float:
    """Vina score -> 0..1 (linear ramp: -3 -> 0.0, -12 -> 1.0)."""
    if vina is None:
        return 0.0
    return max(0.0, min(1.0, (-float(vina) - 3.0) / 9.0))


def _drug_likeness(c: Compound) -> float:
    props = c.properties_json or {}
    mw = props.get("molmass") or c.molmass
    logp = props.get("logp") or c.logp
    hbd = props.get("hbd", c.hbd)
    hba = props.get("hba", c.hba)
    pts = 0.0
    if mw is not None and 200 <= float(mw) <= 500:
        pts += 0.35
    if logp is not None and -1 <= float(logp) <= 5:
        pts += 0.35
    if hbd is not None and int(hbd) <= 5:
        pts += 0.15
    if hba is not None and int(hba) <= 10:
        pts += 0.15
    return pts


def rank_candidates(session, ws_id: str, weights: dict | None = None) -> list[dict]:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    total_w = sum(w.values()) or 1.0

    compounds = list(session.execute(select(Compound)).scalars())
    # KG compound nodes wired to this catalogue row
    c_nodes = list(session.execute(
        select(Node).where(Node.workspace_id == ws_id,
                           Node.kind.in_(("compound", "drug")))
    ).scalars())
    node_by_cid = {(n.properties or {}).get("compound_id"): n for n in c_nodes}
    # name-based fallback: seed compounds may lack explicit compound_id wiring
    node_by_name = {n.canonical_name.lower(): n for n in c_nodes}

    runs = list(session.execute(select(SimulationRun)).scalars())
    dock_by_cid: dict[str, list[float]] = {}
    for r in runs:
        if r.score is not None:
            dock_by_cid.setdefault(r.compound_id, []).append(float(r.score))

    out = []
    for c in compounds:
        node = node_by_cid.get(c.id) or node_by_name.get(c.canonical_name.lower())

        # mechanism relevance: strongest outgoing edge to a target node
        mech, target = 0.0, None
        if node is not None:
            for e in node.out_edges:
                if e.target.kind in TARGET_KINDS:
                    wt = float(e.confidence if e.confidence is not None else 0.5)
                    if wt > mech:
                        mech, target = wt, e.target

        t_conf = float(target.confidence or 0.5) if target else 0.0

        scores = dock_by_cid.get(c.id, [])
        binding = score_to_binding(min(scores)) if scores else 0.0

        struct_cov = 0.0
        if target is not None:
            st = session.execute(
                select(Structure).where(
                    Structure.workspace_id == ws_id,
                    Structure.protein_node_id == target.id)
                .order_by(Structure.resolution.is_(None).asc(),
                          Structure.resolution.asc()).limit(1)
            ).scalar_one_or_none()
            if st is not None:
                struct_cov = 0.5
                if st.ligands:
                    struct_cov = 1.0

        likeness = _drug_likeness(c)
        prov = 1.0 if any(s in ("pubchem", "chembl", "drugbank", "bindingdb")
                          for s in (c.sources_json or {}).keys()) else 0.3

        covered = [mech > 0, bool(scores), struct_cov > 0, likeness > 0.5]
        uncertainty = 1.0 - (sum(covered) / len(covered))

        dims = {
            "mechanism_relevance": round(mech, 3),
            "target_confidence": round(t_conf, 3),
            "binding_evidence": round(binding, 3),
            "structural_coverage": round(struct_cov, 3),
            "drug_likeness": round(likeness, 3),
            "chemical_provenance": round(prov, 3),
            "uncertainty": round(uncertainty, 3),
        }
        composite = sum(dims[k] * w[k] for k in dims) / total_w
        out.append({
            "compound_id": c.id,
            "compound_name": c.canonical_name,
            "target": target.canonical_name if target else None,
            "target_id": target.id if target else None,
            "best_docking": min(scores) if scores else None,
            "dimensions": dims,
            "composite": round(composite, 3),
            "weights_used": w,
        })
    out.sort(key=lambda r: r["composite"], reverse=True)
    return out
