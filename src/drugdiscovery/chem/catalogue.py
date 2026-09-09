"""Compound catalogue service.

Bridges the global compound repository (chem/providers + repos.upsert_compound)
and the per-workspace knowledge graph. A compound is identity-deduplicated by
InChIKey across the whole platform; wiring into a workspace adds a compound KG
node plus typed, provenance-carrying edges to targets.
"""
from __future__ import annotations

import logging

from .. import repos
from . import engine
from .providers import ChEMBLProvider, PubChemProvider, CompoundRecord

log = logging.getLogger("dd.catalogue")

# ChEMBL standard_type -> our edge type
_EDGE_BY_TYPE = {
    "IC50": "INHIBITS",
    "Ki": "INHIBITS",
    "Kd": "BINDS_TO",
    "EC50": "AGONIST_OF",
    "Potency": "REGULATES",
}


def catalogue_compound(session, ws, *, name: str, smiles: str = "", inchikey: str = "",
                       full: dict | None = None, activity: dict | None = None,
                       source: str = "catalogue", source_id: str = "",
                       source_url: str = "", worker: str = "chemist") -> dict:
    """Compute identity, persist to global catalogue, wire into the KG. Returns report dict."""
    # 1) structure-aware identity
    cs = engine.canonical_smiles(smiles) if smiles else (full or {}).get("canonical_smiles", "")
    if not cs and smiles:
        cs = engine.canonical_smiles(smiles)
    props = engine.descriptors(cs) if cs else {}
    inchi, key = engine.inchi_and_key(cs) if cs else ("", inchikey)
    fp = engine.morgan_fp_hex(cs) if cs else None
    props = {**props, "fingerprint_hex": fp} if fp else props
    if key:
        props.setdefault("inchikey", key)

    # 2) global catalogue
    compound, created = repos.upsert_compound(
        session, name=name, canonical_smiles=cs, inchikey=key,
        synonyms=[name],
        sources={source: {"id": source_id, "url": source_url, "activity": activity or {}}},
        properties=props,
    )

    # 3) KG wiring (per-workspace)
    node = repos.find_node(session, ws.id, "compound", name)
    if node is None:
        node, _ = repos.upsert_node(
            session, ws.id, "compound", name, label=name,
            evidence_level="computational",
            confidence=round(max(0.0, min(1.0, float(activity.get("pchembl", 0)) / 10.0 if activity else 0.1)), 3),
            knowledge_type="source_fact",
            properties={"compound_id": compound.id, "inchikey": key},
        )
    # 4) activity -> evidence-grounded edge to the resolved target (if any)
    edge_added = False
    if activity and activity.get("target_node_id"):
        etype = _EDGE_BY_TYPE.get(activity.get("standard_type", ""), "TARGETS")
        confidence = max(0.1, min(1.0, (activity.get("pchembl") or 4.0) / 9.0))
        from ..models import Node  # noqa: PLC0415  (the only Node model lives at drugdiscovery.models, NOT chem.models)
        from sqlalchemy import select as _select
        tn = session.execute(_select(Node).where(
            Node.id == activity["target_node_id"], Node.workspace_id == ws.id)).scalar_one_or_none()
        if tn is not None and node is not None:
            repos.add_edge(
                session, ws.id, etype, node, tn,
                evidence_level="established" if activity.get("pchembl", 0) >= 6.5 else "probable",
                confidence=confidence, knowledge_type="source_fact",
                provenance={"source": source, "source_id": source_id, "url": source_url,
                            "standard_type": activity.get("standard_type"),
                            "pchembl": activity.get("pchembl")},
            )
            edge_added = True
            repos.add_timeline(session, ws.id, worker, "success",
                               f"Catalogued {name} [{key[:10]}…] → {tn.label or tn.canonical_name} "
                               f"({etype}, pChEMBL {activity.get('pchembl')})", {"type": etype})
    return {"id": compound.id, "created": created, "smiles": cs, "inchikey": key,
            "name": compound.canonical_name, "node_id": node.id, "edge_added": edge_added}


# ---------------------------------------------------------------------------
# High-level ingestion helpers used by the research executor
# ---------------------------------------------------------------------------
def fetch_from_pubchem(name: str) -> CompoundRecord | None:
    return PubChemProvider().fetch_by_name(name)


def compounds_for_target_chembl(target_node, session, ws, max_n: int = 12) -> dict:
    """Autonomous compound discovery: ChEMBL activities measured against a target."""
    prov = ChEMBLProvider()
    target_name = target_node.canonical_name or target_node.label
    # ChEMBL free-text search tokenises Greek letters away: "5α-reductase" matches
    # bacterial/HMG-CoA reductases, "5-alpha-reductase" matches human SRD5A. Normalize.
    _GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "μ": "mu", "ω": "omega"}
    for g, a in _GREEK.items():
        target_name = target_name.replace(g, a)
    hits = prov.target_search(target_name)
    if not hits:
        # fall back: search by gene/protein alias
        hits = prov.target_search((target_node.properties or {}).get("gene", target_name))
    if not hits:
        return {"ok": False, "reason": "no ChEMBL target match", "ingested": 0}

    # prefer a Homo sapiens target, else best score
    hits.sort(key=lambda t: 0 if t.get("organism") == "Homo sapiens" else 1)
    best = hits[0]
    records = prov.compounds_for_target(best["id"], min_pchembl=4.0, limit=max_n)
    source_mechanism = False
    if not records:
        # fallback: known mechanism-of-action drugs when measured activity is thin
        mechanisms = prov.mechanisms_for_target(best["id"], limit=max_n)
        if mechanisms:
            source_mechanism = True
            records = [{
                "molecule_chembl_id": r["molecule_chembl_id"],
                "name": r["name"],
                "smiles": r["smiles"],
                "inchikey": r.get("inchikey", ""),
                "standard_type": r.get("action_type") or "MECHANISM",
                "standard_value": None,
                "standard_units": None,
                "relation": None,
                "pchembl": None,
                "assay_type": "B",
                "mechanism": r.get("mechanism"),
                "action_type": r.get("action_type"),
            } for r in mechanisms]
    ingested = 0
    for rec in records:
        act = {
            **rec,
            "target_node_id": target_node.id,
            "target_name": best.get("name"),
            "target_chembl_id": best["id"],
            "source": "chembl_mechanism" if source_mechanism else "chembl",
        }
        try:
            r = catalogue_compound(
                session, ws, name=rec.get("name") or rec["molecule_chembl_id"],
                smiles=rec["smiles"], inchikey=rec.get("inchikey", ""), activity=act,
                source=act["source"], source_id=rec["molecule_chembl_id"],
                source_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{rec['molecule_chembl_id']}/",
            )
            ingested += 1
        except Exception:  # noqa: BLE001
            log.warning("catalogue fail for %s: %s", rec.get("name"), exc_info=True)
    return {"ok": True, "target": best.get("name"), "chembl_id": best["id"],
            "ingested": ingested, "n_records": len(records),
            "source": "chembl_mechanism" if source_mechanism else "chembl"}