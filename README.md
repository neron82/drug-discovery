# Drug Discovery Platform — an in-silico research lab that critiques itself

An autonomous, disease-agnostic in-silico drug-discovery laboratory. It runs a persistent research
loop — **OBSERVE → SEARCH → EXTRACT → MODEL → HYPOTHESIZE → SCORE → CRITIQUE → PRIORITIZE → STORE →
PLAN NEXT** — against real literature, a provenance-carrying knowledge graph, and a typed
experiment queue. The first workspace is androgenetic alopecia (MPB); another disease is a new
workspace config, not a fork.

> **Safety boundary.** This is a computational research and hypothesis-generation system. It
> analyses published experiments and predicts interactions; it does **not** generate synthesis
> protocols, manufacturing recipes, or dosing/self-experimentation instructions. Every prediction is
> labelled as a prediction — never as validated science or medicine.

## Status

Phases 1–6 are implemented and covered by the test suite:

| Phase | What is in the repository |
|---|---|
| 1 · Research core | Workspaces, EuropePMC/PubMed adapters, grounded LLM claim extraction, typed knowledge graph with per-edge provenance, evidence scoring, evidence-grounded hypotheses, deterministic research director, autonomous loop, dashboard |
| 2 · Chemistry | RDKit engine: canonical SMILES, descriptors, fingerprints, similarity, Murcko scaffolds, 2D SVG; PubChem/ChEMBL providers; PAINS/reactive-group safety screens |
| 3 · Structural biology | Gene → UniProt → resolution-ranked PDB entries (ligand-bearing preferred) → CIF artifacts with SHA-256; AlphaFold fallback labelled as prediction |
| 4 · Docking | AutoDock Vina + meeko preparation behind a `DockingEngine` seam, with full reproducibility records (versions, params, seed, hashes, pose artifact, runtime) |
| 5 · Ranking and critique | Multi-factor candidate ranking with every dimension shown separately; adversarial Skeptic that re-gathers evidence, asks the local LLM to falsify, and moves confidence conservatively |
| 6 · Molecular dynamics | OpenMM TIP3P explicit-solvent minimization + short NPT production, backbone RMSD, per-residue RMSF, ligand-contact persistence; persisted `SimulationRun` records |

Phase 7 (generalisation beyond the first disease) is scaffolded through the workspace model and
provider seams.

## Stack

FastAPI (API-first) · SQLAlchemy 2 over SQLite (repository layer kept dialect-agnostic for a
PostgreSQL/pgvector swap) · local OpenAI-compatible multimodal LLM plus bge-m3 embeddings · RDKit ·
AutoDock Vina/meeko · OpenMM · vanilla-JS dark dashboard with vis-network and 3Dmol. No cloud keys
are required.

## Quickstart

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'

# 1) initialise the database, create the MPB workspace, bootstrap from real PubMed/EuropePMC data
uv run python -m drugdiscovery.cli seed --extract 12

# 2) start the API and dashboard
./scripts/dd-service.sh start
./scripts/dd-service.sh status
# open http://127.0.0.1:8020

# 3) enable autonomous research explicitly (separate from the API service)
./scripts/dd-service.sh research-start
./scripts/dd-service.sh research-stop

# one forced manual cycle, without enabling continuous research
./scripts/dd-service.sh cycle <WORKSPACE_ID>
```

`start` starts only the API/UI. Autonomous research is deliberately a separate switch, so a service
restart does not silently resume scientific work.

## Configuration

Platform configuration lives in `config.yaml` (LLM endpoint/model, embeddings, literature provider,
research weights and budgets). Runtime overrides come from `.env`, copied from `.env.example`.

- `LLM_BASE_URL` / `LLM_MODEL_ID` — any OpenAI-compatible endpoint; the default profile is a local
  Qwen3.8-Flash-Next EXL3 server with vision and a 262k context.
- `EMB_BASE_URL` / `EMB_MODEL_ID` — bge-m3 embeddings.
- `NCBI_API_KEY` / `NCBI_MAIL` — optional; raises E-utilities rate limits and sets etiquette.
- `DATA_DIR`, `SQLITE_PATH`, `LOG_DIR`, `ARTIFACT_DIR` — storage locations.

Research scheduling is bounded in `config.yaml`: per-cycle LLM calls, literature requests per hour,
pending-experiment backpressure, a per-cycle cap on new experiments, and stale-job recovery.

## API and CLI

The API is the single source of truth for the UI and the loop. Main groups:

- `/api/status`, `/api/workspaces` and workspace `overview`, `graph`, `papers`, `statements`,
  `hypotheses`, `targets`, `queue`, `timeline`, `activity`, `structures`, `simulations`, `candidates`,
  `critiques`
- `/api/papers/{id}` with per-source full text, `/api/nodes/{id}`
- `/api/compounds` with SVG, similarity and workspace views
- `POST /api/workspaces/{id}/search`, `POST /api/workspaces/{id}/research` (`start` / `pause` / `stop`)
- `/api/simulations/{id}/reproduce` and `/pose`

CLI: `init-db`, `seed --extract N`, `research --cycles N`, `daemon`, `serve`.

## Layout

```
src/drugdiscovery/
  config.py        YAML + environment configuration
  db.py            engine/session (SQLite, WAL)
  models.py        ORM: Workspace, Node, Edge, Statement, Paper, Hypothesis, Experiment, TimelineEntry, Compound, Structure, SimulationRun
  repos.py         repository layer with provenance-aware graph upserts
  entity.py        entity normalization and alias map
  evidence.py      transparent multi-dimensional evidence scoring
  llm/             OpenAI-compatible adapter for the local multimodal model
  lit/             EuropePMC/PubMed providers and sources
  extract/         grounded claim extraction from real abstracts
  hypoth.py        evidence-grounded hypothesis engine
  planner.py       research director (expected information gain × importance × confidence impact ÷ cost)
  research.py      experiment executor and autonomous loop
  critique.py      adversarial Skeptic
  ranking.py       multi-factor candidate ranking
  chem/            RDKit engine, PubChem/ChEMBL providers, safety screens
  struct/          UniProt/PDB/AlphaFold structure acquisition
  sim/             docking (Vina) and MD (OpenMM) engines and runner
  api.py           FastAPI app, static UI and research-loop manager
  static/          dashboard (index.html, style.css, app.js, vendored vis-network and 3Dmol)
  cli.py           init-db / seed / research / daemon / serve
scripts/           operator control and smoke/seed helpers
tests/             isolated test suite
```

## Tests

```bash
uv run python -m pytest -q
```

Current state: **123 passed, 3 skipped**. The suite covers the core loop and bounded autonomy,
literature ingestion and parsing, extraction sources, chemistry and safety screens, structures,
docking, simulation records, ranking and critique.

## Limits

- No authentication or multi-user isolation. The API binds to the host in `config.yaml`; keep it on
  a trusted network.
- SQLite is the current store; the repository layer is the intended migration seam, and migrations
  are not yet managed by Alembic.
- Docking scores, MD results and AlphaFold structures are **computational predictions**. MD is a
  vertical slice (standard force field, short NPT production, whole-complex inputs); large complexes
  may be refused rather than silently approximated.
- The knowledge graph records what was extracted from real abstracts; it is not a curated database,
  and missing evidence stays missing.
