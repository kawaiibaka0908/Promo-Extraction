"""Model-agnostic LLM client.

Every provider is reached through the OpenAI-compatible chat API, so swapping
Gemini -> Groq -> any Kuku-provided model is a config change only.
Features: JSON-only responses validated by pydantic, retries, disk cache,
simple rate limiting, and raw prompt/response logging for the "raw outputs" folder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, TypeAdapter, ValidationError

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> Any:
    """Parse JSON even if the model wrapped it in <thought> tags, ```json fences, or added chatter."""
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the outermost {...} or [...]
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Could not parse JSON from model output: {text[:200]}")


class LLMClient:
    def __init__(self, cfg: dict, raw_log_dir: Optional[Path] = None, offline: bool = False):
        self.cfg = cfg
        prov_name = cfg.get("provider", "gemini")
        prov = cfg["providers"][prov_name]
        self.provider = prov_name
        self.api_key = os.environ.get(prov["api_key_env"], "")
        self.base_url = prov["base_url"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_retries = cfg.get("max_retries", 3)
        self.min_gap = cfg.get("min_seconds_between_calls", 0.0)
        self.cache_dir = Path(cfg.get("cache_dir", ".cache/llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_log_dir = raw_log_dir
        if raw_log_dir:
            raw_log_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline or not self.api_key
        self._last_call = 0.0
        self._client = None
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @property
    def available(self) -> bool:
        return not self.offline

    def model_for(self, role: str) -> str:
        return self.cfg["models"][role]

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45, max_retries=0)
        return self._client

    def _cache_key(self, model: str, system: str, prompt: str) -> str:
        # Cache by role-independent prompt content so fallback models reuse cache seamlessly
        h = hashlib.sha256(f"{self.provider}|{self.temperature}|{system}|{prompt}".encode()).hexdigest()
        return h[:24]

    def _log_raw(self, tag: str, model: str, system: str, prompt: str, response: str, cached: bool):
        if not self.raw_log_dir:
            return
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
        with open(self.raw_log_dir / f"{safe}.json", "w", encoding="utf-8") as f:
            json.dump({"tag": tag, "provider": self.provider, "model": model, "cached": cached,
                       "system": system, "prompt": prompt, "response": response}, f, ensure_ascii=False, indent=2)

    def complete_text(self, prompt: str, *, role: str, system: str = "", tag: str = "call") -> str:
        model = self.model_for(role)
        key = self._cache_key(model, system, prompt)
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            self.stats["cache_hits"] += 1
            resp = json.loads(cpath.read_text(encoding="utf-8"))["response"]
            self._log_raw(tag, model, system, prompt, resp, cached=True)
            return resp
        if self.offline:
            raise LLMError("LLM offline (no API key) and no cached response")

        last_err: Optional[Exception] = None
        fallbacks = [m for m in self.cfg.get("fallbacks", []) if m != model]
        model_chain = [model] + fallbacks
        for attempt in range(1, self.max_retries + 1):
            cur_model = model_chain[min(attempt - 1, len(model_chain) - 1)]
            wait = self.min_gap - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.time()
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
                use_json_mode = ("json" in (system + prompt).lower()) and ("gemma" not in cur_model.lower())
                r = self._get_client().chat.completions.create(
                    model=cur_model, messages=msgs, temperature=self.temperature,
                    response_format={"type": "json_object"} if use_json_mode else None,
                )
                self.stats["calls"] += 1
                if r.usage:
                    self.stats["prompt_tokens"] += r.usage.prompt_tokens or 0
                    self.stats["completion_tokens"] += r.usage.completion_tokens or 0
                resp = r.choices[0].message.content or ""
                cpath.write_text(json.dumps({"model": cur_model, "response": resp}, ensure_ascii=False), encoding="utf-8")
                self._log_raw(tag, cur_model, system, prompt, resp, cached=False)
                return resp
            except Exception as e:  # network, 429, 5xx...
                last_err = e
                backoff = min(15, 2 * attempt)
                if "429" in str(e) or "rate" in str(e).lower():
                    backoff = max(backoff, 10)
                log.warning("LLM call failed on %s (%s/%s): %s; retrying in %ss", cur_model, attempt, self.max_retries, str(e)[:160], backoff)
                time.sleep(backoff)
        self.stats["failures"] += 1
        raise LLMError(f"LLM call failed after retries: {last_err}")

    def complete_json(self, prompt: str, *, role: str, schema: Any, system: str = "", tag: str = "call") -> Any:
        """Call the model and validate against a pydantic model / type. One repair retry on bad output."""
        adapter = TypeAdapter(schema)
        text = self.complete_text(prompt, role=role, system=system, tag=tag)
        try:
            return adapter.validate_python(_extract_json(text))
        except (LLMError, ValidationError) as e:
            log.warning("Invalid JSON from %s (%s); asking for a repair", tag, str(e)[:200])
            repair = (prompt + "\n\nYour previous answer was invalid: " + str(e)[:500] +
                      "\nReturn ONLY valid JSON matching the requested format.")
            text = self.complete_text(repair, role=role, system=system, tag=tag + "_repair")
            return adapter.validate_python(_extract_json(text))
