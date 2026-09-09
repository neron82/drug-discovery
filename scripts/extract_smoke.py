"""Smoke test for the extraction pipeline on real EuropePMC data (background-run safe)."""
import time, sys
from drugdiscovery.lit.providers import get_provider
from drugdiscovery.llm import LLMProvider
from drugdiscovery.db import get_session, init_db
from drugdiscovery import repos
from drugdiscovery.models import Workspace
from sqlalchemy import select

init_db()
with get_session() as s:
    for w in s.execute(select(Workspace).where(Workspace.slug == "xtest")).scalars():
        s.delete(w)

p = get_provider("europepmc")
recs = p.search("androgenetic alopecia dermal papilla androgen receptor", 4)
print("records:", len(recs), flush=True)
with get_session() as s:
    ws = repos.create_workspace(s, "XTEST", "scratch")
    s.commit(); wid = ws.id
    for r in recs[:2]:
        t0 = time.time()
        paper, _ = repos.upsert_paper(s, wid, r); s.commit()
        from drugdiscovery.extract import process_paper
        process_paper(s, ws, paper, LLMProvider()); s.commit()
        print(f"[{time.time()-t0:.0f}s] PAPER {paper.status} claims={len(paper.statements)} :: {paper.title[:50]}", flush=True)
        for st in paper.statements[:8]:
            print("   >", st.subject, "|", st.relation, "|", st.object, "|", st.evidence_level, st.species, st.model_context, flush=True)
    g = repos.graph_projection(s, wid)
    print("nodes:", len(g["nodes"]), "edges:", len(g["edges"]), flush=True)
print("SMOKE_OK", flush=True)
