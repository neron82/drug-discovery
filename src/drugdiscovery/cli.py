"""CLI: init-db, seed (Phase-1 MPB bootstrap), research cycles, daemon, serve."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dd.cli")

from .db import init_db, get_session  # noqa: E402
from . import repos  # noqa: E402


MPB_CONFIG = {
    "disease": "androgenetic_alopecia",
    "tissues": ["scalp", "hair_follicle"],
    "relevant_cell_types": ["dermal_papilla_cell", "keratinocyte"],
    "phenotype": ["follicular_miniaturization"],
    "seed_queries": [
        "androgenetic alopecia dermal papilla androgen receptor",
        "5-alpha-reductase finasteride androgenetic alopecia",
        "hair follicle Wnt beta-catenin signaling",
        "hair follicle TGF-beta catagen apoptosis",
        "prostaglandin D2 receptor hair loss",
        "androgenetic alopecia oxidative stress inflammation",
        "minoxidil mechanism hair growth",
        "dutasteride dual 5-alpha-reductase hair",
    ],
}


def _bootstrap(ws_id: str, extract_n: int):
    """Deterministic Phase-1 bootstrap: ingest seed queries, extract, hypothesize, questions."""
    from .llm import LLMProvider
    from .lit.providers import get_provider
    from . import hypoth
    from .research import derive_open_questions
    from .models import Paper, Hypothesis
    from sqlalchemy import select, func

    llm = LLMProvider()
    provider = get_provider("europepmc")

    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        cfg = ws.config_json or {}
        for q in cfg.get("seed_queries", MPB_CONFIG["seed_queries"]):
            print(f"[seed] search: {q}")
            try:
                recs = provider.search(q, 10)
            except Exception as e:
                print(f"  search error: {e}")
                continue
            new = 0
            for r in recs:
                _, created = repos.upsert_paper(s, ws_id, r)
                new += int(created)
            repos.add_timeline(s, ws_id, "researcher", "info", f"Seed search '{q}': {new} new papers", {})
            s.commit()
            print(f"  -> {new} new")
        s.commit()

    # extraction
    extracted = 0
    with get_session() as s:
        papers = s.execute(select(Paper).where(Paper.workspace_id == ws_id, Paper.status == "fetched")
                           .order_by(Paper.year.desc())).scalars().all()
        print(f"[seed] extracting up to {extract_n} papers (LLM: {llm.model})")
        for p in papers[:extract_n]:
            if not p.abstract:
                p.status = "skipped"
                continue
            from .extract import process_paper
            process_paper(s, ws, p, llm)
            s.commit()
            extracted += 1
            print(f"  {extracted}/{min(extract_n, len(papers))} {p.title[:50]} -> {p.status} ({len(p.statements)} claims)")
    print(f"[seed] extracted from {extracted} papers")

    # hypotheses + questions
    with get_session() as s:
        ws = repos.get_workspace(s, ws_id)
        n_hyp = s.execute(select(func.count()).select_from(Hypothesis).where(Hypothesis.workspace_id == ws_id)).scalar_one()
        if n_hyp == 0:
            made = hypoth.generate_hypotheses(s, ws, llm, limit=3)
            s.commit()
            print(f"[seed] generated {len(made)} hypotheses")
        questions = derive_open_questions(s, ws)
        st = dict(ws.state_json or {})
        st["open_questions"] = [{"q": q} for q in questions[:8]]
        ws.state_json = st
        s.commit()
        print("[seed] open questions:")
        for q in questions[:8]:
            print(f"   ? {q}")


def cmd_init(_):
    init_db()
    print("db initialised at", __import__("drugdiscovery.config", fromlist=["settings"]).settings.storage.db_path)


def cmd_seed(args):
    init_db()
    with get_session() as s:
        existing = repos.list_workspaces(s)
        ws = None
        for w in existing:
            if w.slug == "androgenetic-alopecia" or w.disease == "androgenetic_alopecia":
                ws = w
                break
        if ws is None:
            ws = repos.create_workspace(s, "Androgenetic Alopecia", disease="androgenetic_alopecia",
                                        description="Autonomous in-silico MPB drug-discovery workspace",
                                        config=dict(MPB_CONFIG))
            s.commit()
            print(f"created workspace {ws.slug} id={ws.id}")
        else:
            print(f"reusing workspace {ws.slug} id={ws.id}")
        wid = ws.id
    _bootstrap(wid, extract_n=args.extract)
    print(f"\nDONE. workspace id={wid}\n"
          f"  UI:  http://127.0.0.1:{ __import__('drugdiscovery.config', fromlist=['settings']).settings.api.port }/"
          f"  research control: POST /api/workspaces/{wid}/research `{{'action':'start'}}`")


def cmd_research(args):
    from .research import run_cycle
    from . import planner
    init_db()
    n = args.cycles or 1
    print(f"running {n} research cycle(s) on workspace {args.ws}")
    for i in range(1, n + 1):
        with get_session() as s:
            ws = repos.get_workspace(s, args.ws)
            if ws:
                ws.research_enabled = True
        r = run_cycle(args.ws)
        print(f"cycle {i}: {r}")
        if r.get("noop"):
            break


def cmd_daemon(args):
    from .research import research_loop
    init_db()
    with get_session() as s:
        ws = repos.get_workspace(s, args.ws)
        if ws:
            ws.research_enabled = True
    ev = asyncio.Event()
    asyncio.run(research_loop(args.ws, ev))


def cmd_serve(args):
    import uvicorn
    from .config import settings
    uvicorn.run("drugdiscovery.api:app", host=args.host, port=args.port, reload=args.reload)


def main():
    p = argparse.ArgumentParser(prog="drug-discovery")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init)

    s_seed = sub.add_parser("seed", help="bootstrap the MPB workspace (real PubMed data)")
    s_seed.add_argument("--extract", type=int, default=12, help="max papers to LLM-extract")
    s_seed.set_defaults(func=cmd_seed)

    s_res = sub.add_parser("research", help="run N research cycles synchronously")
    s_res.add_argument("--ws", required=True)
    s_res.add_argument("--cycles", type=int, default=1)
    s_res.set_defaults(func=cmd_research)

    s_da = sub.add_parser("daemon", help="run the autonomous research loop continuously (ctrl-c to stop)")
    s_da.add_argument("--ws", required=True)
    s_da.set_defaults(func=cmd_daemon)

    s_sv = sub.add_parser("serve", help="run the API + UI server")
    s_sv.add_argument("--host", default=None)
    s_sv.add_argument("--port", type=int, default=None)
    s_sv.add_argument("--reload", action="store_true")
    s_sv.set_defaults(func=cmd_serve)

    args = p.parse_args()
    # map omitted ports to settings
    if args.cmd == "serve":
        from .config import settings
        if args.host is None:
            args.host = settings.api.host
        if args.port is None:
            args.port = settings.api.port
    args.func(args)


if __name__ == "__main__":
    main()
