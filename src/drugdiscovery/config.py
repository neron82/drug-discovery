"""Configuration loading: .env (env) + config.yaml (platform) overlay."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen3.8-Flash-Next-exl3"
    api_key: str = ""
    temperature: float = 0.2
    json_mode: bool = True
    enable_thinking: bool = False  # reasoning models: keep off for fast, JSON-reliable extraction


class EmbedConfig(BaseModel):
    base_url: str = "http://127.0.0.1:7777/v1"
    model: str = "bge-m3"
    enabled: bool = True
    dimension: int = 1024


class Budgets(BaseModel):
    max_lit_requests_per_hour: int = 90
    max_simulations_per_hour: int = 0
    max_llm_calls_per_cycle: int = 15


class ResearchConfig(BaseModel):
    default_weights: dict = {
        "expected_information_gain": 1.0,
        "scientific_importance": 1.0,
        "confidence_impact": 1.0,
        "computational_cost": 0.5,
    }
    loop_sleep_seconds: int = 20
    cycle_safety_cap: int = 4
    max_pending_experiments: int = 64
    stale_running_after_minutes: int = 180
    # Bounded autonomy: per-(kind,params) retry ceiling. After this many FAILED
    # attempts the planner stops re-scheduling the identical job (convergence).
    max_retries_per_kind: int = 2
    # Stagnation stop: pause the workspace after this many consecutive cycles
    # that produced no new evidence/compounds/structures/statements/runs.
    stagnation_limit: int = 12
    # Drain lease: a workspace may be drained by only ONE holder at a time
    # (API loop or manual script). Lease expires if the holder dies.
    drain_lease_seconds: int = 300
    budgets: Budgets = Budgets()


class StorageConfig(BaseModel):
    data_dir: str = "./data"
    db_path: str = "./data/research.db"
    artifact_dir: str = "./data/artifacts"
    log_dir: str = "./logs"
    venv_bin: str = "./.venv/bin"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8020


class Settings(BaseModel):
    api: APIConfig = APIConfig()
    storage: StorageConfig = StorageConfig()
    llm: LLMConfig = LLMConfig()
    embeddings: EmbedConfig = EmbedConfig()
    literature: dict = {"default_provider": "europepmc", "max_requests_per_hour": 90, "max_papers_per_search": 12}
    research: ResearchConfig = ResearchConfig()


def _resolve_rel(path: str, base: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (base / p))


def load_settings() -> Settings:
    """Load config.yaml, then overlay env (.env / process env)."""
    base = _PROJECT_ROOT
    cfg_path = base / "config.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

    # .env overlay
    if (base / ".env").exists():
        for line in (base / ".env").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    # translate env keys into structs
    def e(key, default):
        return os.environ.get(key, default)

    llm = cfg.get("llm", {})
    llm_c = LLMConfig(
        base_url=e("LLM_BASE_URL", llm.get("base_url", "http://127.0.0.1:8080/v1")),
        model=e("LLM_MODEL_ID", llm.get("model", "Qwen3.8-Flash-Next-exl3")),
        api_key=e("LLM_API_KEY", llm.get("api_key", "")),
        temperature=float(e("LLM_TEMPERATURE", llm.get("temperature", 0.2))),
        json_mode=bool(llm.get("json_mode", True)),
        enable_thinking=bool(llm.get("enable_thinking", False)),
    )
    emb = cfg.get("embeddings", {})
    emb_c = EmbedConfig(
        base_url=e("EMB_BASE_URL", emb.get("base_url", "http://127.0.0.1:7777/v1")),
        model=e("EMB_MODEL_ID", emb.get("model", "bge-m3")),
        enabled=(e("EMB_ENABLED", "true").lower() == "true"),
        dimension=int(emb.get("dimension", 1024)),
    )
    st = cfg.get("storage", {})
    st_c = StorageConfig(
        data_dir=_resolve_rel(st.get("data_dir", "./data"), base),
        db_path=_resolve_rel(e("SQLITE_PATH", st.get("db_path", "./data/research.db")), base),
        artifact_dir=_resolve_rel(st.get("artifact_dir", "./data/artifacts"), base),
        log_dir=_resolve_rel(st.get("log_dir", "./logs"), base),
        venv_bin=_resolve_rel(
            e("VENV_BIN", str(Path(base) / ".venv" / "bin")), Path(base),
        ),
    )
    ap = cfg.get("api", {})
    api_c = APIConfig(host=e("API_HOST", ap.get("host", "0.0.0.0")), port=int(e("API_PORT", ap.get("port", "8020"))))

    lit = cfg.get("literature", {}) or {}
    lit_c = {
        "default_provider": e("LIT_PROVIDER", lit.get("default_provider", "europepmc")),
        "max_requests_per_hour": int(lit.get("max_requests_per_hour", 90)),
        "max_papers_per_search": int(lit.get("max_papers_per_search", 12)),
    }
    res = cfg.get("research", {}) or {}
    res_c = ResearchConfig(
        default_weights=res.get("default_weights", ResearchConfig().default_weights),
        loop_sleep_seconds=int(res.get("loop_sleep_seconds", 20)),
        cycle_safety_cap=int(res.get("cycle_safety_cap", 4)),
        max_pending_experiments=int(res.get("max_pending_experiments", 64)),
        stale_running_after_minutes=int(res.get("stale_running_after_minutes", 180)),
        max_retries_per_kind=int(res.get("max_retries_per_kind", 2)),
        stagnation_limit=int(res.get("stagnation_limit", 12)),
        drain_lease_seconds=int(res.get("drain_lease_seconds", 300)),
    )
    return Settings(api=api_c, storage=st_c, llm=llm_c, embeddings=emb_c, literature=lit_c, research=res_c)


settings = load_settings()
