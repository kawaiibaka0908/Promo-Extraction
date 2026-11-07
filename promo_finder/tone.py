"""Step 2B - tone-word intensity via sentence embeddings (no hand-written word lists).

Each tone word from the dialogue tags is embedded and projected onto an
intensity axis = mean(high anchors) - mean(low anchors). Unseen words are
handled automatically because similar meanings get similar vectors.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def score_tones(words: list[str], cfg: dict, cache_dir: Path = Path(".cache")) -> dict[str, float]:
    if not cfg.get("enabled", True) or not words:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / "tone_scores.json"
    cache: dict[str, float] = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    anchors_key = "|".join(cfg["high_anchors"] + ["/"] + cfg["low_anchors"] + [cfg["model"]])
    if cache.get("__anchors__") != anchors_key:
        cache = {"__anchors__": anchors_key}
    todo = sorted({w for w in words if w not in cache})
    if todo:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # library missing -> degrade gracefully
            log.warning("sentence-transformers unavailable (%s); tone signal disabled", e)
            return {}
        try:
            model = SentenceTransformer(cfg["model"])
        except Exception as e:  # e.g. no network to download the model
            log.warning("Could not load embedding model (%s); tone signal disabled", e)
            return {}
        hi = model.encode(cfg["high_anchors"], normalize_embeddings=True).mean(0)
        lo = model.encode(cfg["low_anchors"], normalize_embeddings=True).mean(0)
        axis = hi - lo
        axis /= np.linalg.norm(axis)
        # also project the anchors to define the 0..1 scale
        ref = model.encode(cfg["high_anchors"] + cfg["low_anchors"], normalize_embeddings=True) @ axis
        lo_ref, hi_ref = float(ref[len(cfg["high_anchors"]):].mean()), float(ref[: len(cfg["high_anchors"])].mean())
        vecs = model.encode(todo, normalize_embeddings=True, batch_size=128)
        proj = vecs @ axis
        for w, p in zip(todo, proj):
            cache[w] = float(np.clip((p - lo_ref) / (hi_ref - lo_ref + 1e-9), 0, 1))
        cpath.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return {w: cache[w] for w in words if w in cache}
