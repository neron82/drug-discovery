"""LLM adapter layer — provider-agnostic OpenAI-compatible client (local multimodal)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

from ..config import settings

log = logging.getLogger("dd.llm")


class LLMProvider:
    """Thin OpenAI-compatible chat client with robust JSON extraction."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str = "",
                 temperature: float | None = None):
        self.base_url = (base_url or settings.llm.base_url).rstrip("/")
        self.model = model or settings.llm.model
        self.api_key = api_key if api_key else settings.llm.api_key
        self.temperature = settings.llm.temperature if temperature is None else temperature
        self._headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # -- introspection ----------------------------------------------------
    def list_models(self) -> list[dict]:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers, timeout=10)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:  # pragma: no cover
            log.warning("list_models failed: %s", e)
            return []

    def capabilities(self) -> dict:
        models = self.list_models()
        info: dict = {"base_url": self.base_url, "model": self.model, "support_vision": False, "context_window": None}
        for m in models:
            if m.get("id") == self.model or not models:
                info["support_vision"] = bool(m.get("supports_vision")) or "vision" in str(m.get("architecture", "")).lower()
                info["context_window"] = m.get("configured_context_window") or m.get("context_window") or m.get("max_model_len")
                info["architecture"] = m.get("architecture")
        return info

    # -- chat -------------------------------------------------------------
    def chat(self, system: str, user: str, json_schema: Optional[dict] = None,
             temperature: float | None = None, timeout: float = 180) -> str:
        """Return raw assistant text; if json_schema given, hints structured JSON output."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": 4000,
            "stream": False,
        }
        if json_schema and settings.llm.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload["max_tokens"] = max(payload["max_tokens"], 8000)
        if not settings.llm.enable_thinking:
            payload["enable_thinking"] = False  # suppress model chain-of-thought for speed + reliable JSON
        last_err = None
        for attempt in range(2):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers,
                               json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                last_err = e
                log.warning("chat attempt %d failed: %s", attempt + 1, e)
        raise RuntimeError(f"LLM chat failed: {last_err}")

    def chat_json(self, system: str, user: str, temperature: float | None = None) -> dict:
        """Chat with JSON-mode, robustly parse the returned object (with one strict retry)."""
        try:
            text = self.chat(system, user, json_schema={}, temperature=temperature)
            return _parse_json(text)
        except ValueError:
            # some reasoning models ramble; retry with a blunt final instruction
            strict = (user + "\n\nREMINDER: Reply now with ONLY one JSON object. "
                      "No reasoning, no markdown, no code fences, no commentary. "
                      'Begin directly with "{" and end with "}".')
            text2 = self.chat(system, strict, json_schema={}, temperature=0.0)
            return _parse_json(text2)

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: list[str], model: str | None = None) -> Optional[list[list[float]]]:
        """Embed via configured embeddings endpoint (bge-m3 :7777). Returns None if disabled."""
        if not settings.embeddings.enabled:
            return None
        base = settings.embeddings.base_url.rstrip("/")
        mdl = model or settings.embeddings.model
        try:
            r = httpx.post(f"{base}/embeddings",
                           json={"model": mdl, "input": texts}, timeout=120)
            r.raise_for_status()
            data = r.json().get("data", [])
            out = [None] * len(texts)
            for item in data:
                idx = item.get("index")
                if idx is not None and idx < len(texts):
                    out[idx] = item["embedding"]
            return [v for v in out if v is not None] or None
        except Exception as e:
            log.warning("embed failed: %s", e)
            return None


def _parse_json(text: str) -> dict:
    """Return the LAST valid JSON object embedded in arbitrary LLM text.

    Robust against chain-of-thought: reasoning preamble, code fences and example
    objects are ignored; we scan every '{' and keep the final object that parses.
    """
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    hits = []
    i = 0
    n = len(t)
    while i < n:
        if t[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            c = t[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            hits.append(json.loads(t[i:j + 1]))
                        except Exception:
                            pass
                        break
            j += 1
        i += 1
        if hits:
            # if we found one starting here, jump past it
            if hits:
                i = j + 1
    if hits:
        return hits[-1]
    raise ValueError(f"Could not parse JSON from LLM output: {text[:300]}")

