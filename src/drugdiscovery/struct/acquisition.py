"""Structure acquisition service (Phase 3).

For each protein/receptor/enzyme node in a workspace: resolve gene symbol ->
UniProt accession -> best PDB entries (resolution-ranked, ligand-bearing
preferred) -> download cif into artifact store (sha256, dedup by hash) ->
persist Structure rows wired to the protein node. Targets without experimental
structures fall back to AlphaFold, stored as COMPUTATIONAL PREDICTION.
"""
from __future__ import annotations

import logging
import os
import re

from sqlalchemy import select

from .. import repos
from ..config import settings
from ..models import Node, Structure
from . import providers

log = logging.getLogger("dd.acquire")

_GENE_RE = re.compile(r"\(([A-Z][A-Z0-9-]{1,9})\)$")
_KINDS = ("protein", "receptor", "enzyme", "gene")


def gene_symbol(node: Node) -> str:
    m = _GENE_RE.search(node.canonical_name or "")
    if m:
        return m.group(1)
    for s in (node.synonyms or []):
        m = _GENE_RE.search(s)
        if m:
            return m.group(1)
    return (node.canonical_name or "").split()[0].upper()


def _artifact_path(ws_id: str, source: str, source_id: str, data: bytes) -> tuple[str, str]:
    h = providers.sha256(data)
    d = os.path.join(settings.storage.artifact_dir, "structures", source)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{source_id}_{h[:12]}.cif")
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(data)
    return p, h


def _store(session, ws_id, node, *, source, source_id, uniprot, method, resolution,
           organism, ligands, data, limitations, confidence=None):
    p, h = _artifact_path(ws_id, source, source_id, data)
    st = session.execute(select(Structure).where(
        Structure.source == source, Structure.source_id == source_id)).scalar_one_or_none()
    if st is None:
        st = Structure(source=source, source_id=source_id, workspace_id=ws_id,
                       protein_node_id=node.id, uniprot_accession=uniprot, method=method,
                       resolution=resolution, confidence=confidence, organism=organism,
                       ligands=ligands, file_path=p, file_hash=h, limitations=limitations)
        session.add(st)
        session.flush()
        return st, True
    st.protein_node_id = st.protein_node_id or node.id
    st.file_path = st.file_path or p
    st.file_hash = st.file_hash or h
    return st, False


def acquire_for_target(session, ws, node, *, max_pdb: int = 3) -> dict:
    """Resolve one protein node to structures. Returns report dict."""
    gene = gene_symbol(node)
    up = providers.uniprot_for_gene(gene)
    if not up or not up["accession"]:
        return {"gene": gene, "status": "no_uniprot"}
    acc = up["accession"]
    xrefs = providers.pdb_cross_refs(acc)
    got_new, stored = 0, []
    if xrefs:
        xrefs.sort(key=lambda x: (x["resolution"] is None, x["resolution"] or 99))
        for x in xrefs[:max_pdb]:
            meta = providers.pdb_entry_meta(x["pdb_id"])
            data = providers.fetch_cif(x["pdb_id"])
            if not data:
                continue
            lim = "" if x["method"] == "x-ray" else f"{x['method']} structure; interpret pose carefully"
            st, new = _store(session, ws.id, node, source="pdb", source_id=x["pdb_id"],
                             uniprot=acc, method=x["method"] or meta["method"],
                             resolution=x["resolution"] or meta["resolution"],
                             organism=meta["organism"], ligands=meta["ligands"],
                             data=data, limitations=(lim or meta["title"])[:160])
            stored.append({"source": "pdb", "id": x["pdb_id"], "new": new, "ligands": len(meta["ligands"])})
            got_new += int(new)
    if not stored:
        af = providers.alphafold(acc)
        if af and af["cif_url"]:
            data = providers._get_bytes(af["cif_url"])
            if data:
                st, new = _store(session, ws.id, node, source="alphafold",
                                 source_id=af["source_id"] or f"AF-{acc}", uniprot=acc,
                                 method="predicted", resolution=None, confidence=af.get("confidence"),
                                 organism=af.get("organism") or "Homo sapiens",
                                 ligands=[], data=data,
                                 limitations="AlphaFold predicted model: no experimental coordinates; side-chain placement uncertain in flexible regions")
                stored.append({"source": "alphafold", "id": af["source_id"], "new": new, "ligands": 0})
                got_new += int(new)
    props = dict(node.properties or {})
    props["structural"] = {"uniprot": acc, "pdb_count": len(xrefs), "acquired": [s["id"] for s in stored]}
    node.properties = props
    return {"gene": gene, "accession": acc, "pdb_candidates": len(xrefs),
            "stored": stored, "new": got_new, "status": "ok" if stored else "no_structures"}


def acquire_structures(session, ws, *, limit: int = 8) -> dict:
    """Sweep protein/receptor/enzyme/gene nodes lacking acquired structures."""
    nodes = session.execute(select(Node).where(
        Node.workspace_id == ws.id, Node.kind.in_(_KINDS)
    ).order_by(Node.confidence.desc())).scalars().all()
    todo = [n for n in nodes if not ((n.properties or {}).get("structural", {}).get("acquired"))]
    reports, done = [], 0
    for n in todo[:limit]:
        rep = acquire_for_target(session, ws, n)
        reports.append({"node": n.canonical_name, **rep})
        done += rep.get("new", 0)
    return {"targets_scanned": len(todo), "acquired_new": done, "reports": reports}
