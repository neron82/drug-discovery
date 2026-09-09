# Drug Discovery Platform — in-silico computational research lab

An **autonomous, disease-agnostic in-silico drug-discovery laboratory** that runs a
persistent research loop (OBSERVE → SEARCH → EXTRACT → MODEL → HYPOTHESIZE → SCORE →
CRITIQUE → PRIORITIZE → STORE → PLAN NEXT → REPEAT). The initial demonstration workspace
is **androgenetic alopecia (MPB)**; the architecture generalises to any disease via a
per-workspace config + knowledge graph.

> **Safety boundary.** This is a computational research & hypothesis-generation system.
> It analyses published experiments and predicts interactions; it does **not** generate
> synthesis protocols, manufacturing recipes, or dosing/self-experimentation instructions.
> Predictions are always labelled as predictions — never as validated science or medicine.

## What it does today (Phase 1 — Scientific Research Core)

- **Workspaces** — disease-agnostic project containers (disease, tissues, cell types, phenotypes, seed queries).
- **Literature engine** — adapter-based search (EuropePMC default, PubMed fallback) with real abstracts.
- **LLM claim extraction** — grounded structured claims from real abstracts (no fabricated science),
  driven by the **local multimodal model** `:8080` (Qwen3.8-Flash-Next-exl3, 262k ctx, vision).
- **Knowledge graph** — typed, directional nodes/edges (Node/Edge/Statement) where **every edge
  carries provenance** (paper + statement) and a `knowledge_type`
  (`source_fact | model_inference | computational_prediction | hypothesis`).
- **Evidence engine** — transparent multi-dimensional scoring (sources, replication, human/animal/
  in-vitro/clinical mass, contradiction, recency, model relevance) — never one opaque number.
- **Hypothesis engine** — draft hypotheses **grounded only in collected evidence**, with mechanism,
  predicted effect, supporting/counter evidence, uncertainty, predictions, next experiment.
- **Research Director** — deterministic planner that converts KG gaps into a *prioritized* queue
  (`Priority = EIG × importance × confidence-impact ÷ cost`) and picks the next experiment.
- **Skeptic** — searches for negative/contradictory evidence.
- **Autonomous loop** — `START / PAUSE / STOP` + compute budgets; runs continuously.
- **UI** — polished dark-mode dashboard (Overview, Mechanism Map, Targets, Literature, Hypotheses,
  Research Queue, Timeline, System + live Agent-Activity panel).

Phases 2–7 (compounds/RDKit, structural biology, docking, MD, generalization) are scaffolded with
schemas/adapters ready.

## Stack

FastAPI (API-first) · SQLAlchemy 2 (SQLite today — repository layer is dialect-agnostic for Postgres/pgvector swap)
· local OpenAI-compatible LLM + bge-m3 embeddings · vanilla-JS dark UI + vis-network graph.

## Quickstart

```bash
cd ~/projects/drug-discovery
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'

# 1) initialise DB, create the MPB workspace and bootstrap it from REAL PubMed/EuropePMC data
uv run python -m drugdiscovery.cli seed --extract 12

# 2) start the API + dashboard
uv run python -m drugdiscovery.cli serve --host 0.0.0.0 --port 8020
# open http://127.0.0.1:8020

# 3) autonomous research: run N cycles now (or press ▶ Start in the UI)
uv run python -m drugdiscovery.cli research --ws <WORKSPACE_ID> --cycles 5
# continuous daemon:
uv run python -m drugdiscovery.cli daemon --ws <WORKSPACE_ID>
```

## Configuration

Platform config: `config.yaml` (LLM endpoint/model, embeddings, literature provider, research
budgets/weights). Runtime overrides: `.env` (copy from `.env.example`). No cloud keys required —
everything runs against the local multimodal LLM and bge-m3. Swap the LLM by changing
`LLM_BASE_URL`/`LLM_MODEL_ID` (any OpenAI-compatible endpoint).

## Test

```bash
uv run python -m pytest -q
```

## Project layout (highlights)

```
src/drugdiscovery/
  config.py        config load (yaml + env)
  db.py            engine/session (SQLite, WAL)
  models.py        ORM: Workspace, Node, Edge, Statement, Paper, Hypothesis, Experiment(Job), TimelineEntry, Compound
  repos.py         repository layer + provenance-aware KG upserts
  entity.py        entity normalization / alias map (MPB seed; generic fallback)
  evidence.py      transparent multi-dimensional evidence scoring
  llm/             OpenAI-compatible adapter (local multimodal)
  lit/providers.py literature adapters (EuropePMC, PubMed)
  extract/         grounded LLM claim extraction
  hypoth.py        evidence-grounded hypothesis engine
  planner.py       Research Director (scoring queue)
  research.py      experiment executor + autonomous loop
  api.py           FastAPI (API-first + static UI + research-loop manager)
  static/          dark-mode dashboard (index.html/style.css/app.js + vis-network)
  cli.py           init-db / seed / research / daemon / serve
```
