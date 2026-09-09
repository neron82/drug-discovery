"""Seed real chemical identities for the molecular names already present in the
MPB knowledge graph (from literature extraction). Structures come from PubChem
(real data); the compound KG node is linked to the catalogue by compound_id so
editorial, literature-derived edges stay untouched. Never fabricates structures.

Usage: uv run python scripts/seed_chem.py
"""
import sys, time
from sqlalchemy import select

sys.path.insert(0, "src")
from drugdiscovery.db import get_session
from drugdiscovery.models import Node
from drugdiscovery.chem import engine, catalogue
from drugdiscovery.chem.providers import PubChemProvider

WS = "f8f894ac-aac9-478c-8467-0ac2ea3ebfc8"

# molecule names that already exist as KG nodes via literature extraction
NAMES = [
    "Finasteride", "Dutasteride", "Minoxidil", "Tretinoin", "Latanoprost",
    "Latanoprost acid", "Spironolactone", "Bicalutamide",
]

prov = PubChemProvider()
s = get_session()
stats = {"created": 0, "merged": 0, "failed": 0}
for name in NAMES:
    try:
        rec = prov.fetch_by_name(name)
    except Exception as e:
        print(f"[{name}] lookup error: {e}")
        stats["failed"] += 1
        continue
    if rec is None or not rec.smiles:
        print(f"[{name}] no structure")
        stats["failed"] += 1
        continue
    cs = engine.canonical_smiles(rec.smiles)
    if not cs:
        print(f"[{name}] invalid smiles")
        stats["failed"] += 1
        continue
    props = engine.descriptors(cs)
    key = engine.inchi_and_key(cs)[1]
    from drugdiscovery import repos as R
    compound, created = R.upsert_compound(
        s, name=name, canonical_smiles=cs, inchikey=key, synonyms=rec.synonyms,
        sources={"pubchem": {"cid": rec.source_id, "url": rec.source_url}},
        properties=props,
    )
    # wire structure identity into the existing literature node (drug/compound) if present,
    # so editorial target edges are preserved; otherwise create a compound node.
    existing = None
    for k2 in ("drug", "compound"):
        existing = R.find_node(s, WS, k2, name)
        if existing:
            break
    if existing is not None:
        props = dict(existing.properties or {})
        props["compound_id"] = compound.id
        props["inchikey"] = key
        existing.properties = props
        node = existing
        created_node = False
    else:
        nnode, created_node = R.upsert_node(
            s, WS, "compound", name, label=name,
            evidence_level="established", confidence=0.8,
            knowledge_type="source_fact",
            properties={"compound_id": compound.id, "inchikey": key})
        node = nnode
    stats["merged" if not created else "created"] += 1
    print(f"[{name}] {'created' if created else 'merged'} compound={compound.id[:8]} "
          f"node={node.id[:8]}(exist={existing is not None}) key={key[:14]} formula={props.get('formula')} MW={props.get('molmass')}")
    time.sleep(0.4)  # be gentle to PubChem

s.commit()
# cleanup: drop duplicate compound-kind nodes that carry a compound_id but no edges
# (leftovers from the earlier seed that created fresh nodes instead of merging)
from drugdiscovery.models import Edge
WO = select(Node).where(Node.workspace_id == WS, Node.kind == "compound")
removed = 0
for n in s.execute(WO).scalars().all():
    if not (n.properties or {}).get("compound_id"):
        continue
    ne = s.execute(select(Edge).where(Edge.workspace_id == WS,
                                      (Edge.source_id == n.id) | (Edge.target_id == n.id))).scalars().all()
    if ne:
        continue
    s.delete(n)
    removed += 1
s.commit()
print("STATS:", stats)
print("orphan compound nodes removed:", removed)
print("catalogue rows now:", len(R.list_compounds(s)))