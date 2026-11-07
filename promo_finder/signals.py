"""Step 2 - per-beat signals and the per-episode heat curve.

A. Structural features (code, reads the screenplay *format*, no word lists)
B. Tone intensity (embeddings, see tone.py)
C. LLM beat tagging (meaning; returns beat_ids only, validated)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from .llm import LLMClient, LLMError
from .parse import Beat, Episode

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


class BeatTag(BaseModel):
    beat_id: str
    intensity: float = Field(ge=0, le=10)
    event: str = "other"
    hook_worthy: bool = False
    unresolved: bool = False
    needs_context: Literal["low", "medium", "high"] = "medium"
    why: str = ""


class TagResponse(BaseModel):
    tags: list[BeatTag]


# --------------------------------------------------------------------------- A. structure

def structural_scores(ep: Episode) -> dict[str, float]:
    """0-1 score per beat from format cues: SFX, cut-to-black, fast cutting, punctuation,
    short punchy lines, rapid speaker switching, end-of-episode position."""
    beats = ep.beats
    dur = max(ep.duration, 1.0)
    cam_starts = np.array([b.start for b in beats if b.type == "camera"])
    dlg = [b for b in beats if b.type == "dialogue"]
    sfx = [b for b in beats if b.type == "sfx"]
    out: dict[str, float] = {}
    for b in beats:
        mid = (b.start + b.end) / 2
        # fast cutting: shots starting within +-2.5 s (5+ shots in 5 s = max)
        shot_rate = min(1.0, float(((cam_starts >= mid - 2.5) & (cam_starts <= mid + 2.5)).sum()) / 5.0)
        # speaker switches within +-5 s
        near = [d.speaker for d in dlg if abs((d.start + d.end) / 2 - mid) <= 5]
        switches = sum(1 for a, c in zip(near, near[1:]) if a != c)
        switch_rate = min(1.0, switches / 4.0)
        has_sfx = 1.0 if any(s.start <= b.end and s.end >= b.start for s in sfx) else 0.0
        punct = 0.0
        short = 0.0
        if b.type == "dialogue":
            punct = 1.0 if "!" in b.text else (0.6 if "?" in b.text else 0.0)
            short = 1.0 if len(b.text.split()) <= 7 else 0.0
        cut_black = 1.0 if (b.type == "camera" and re.search(r"cut to black|fade to black", b.text, re.I)) else 0.0
        end_bonus = 1.0 if b.start >= dur - 15 else 0.0
        out[b.beat_id] = float(np.clip(
            0.25 * shot_rate + 0.15 * switch_rate + 0.2 * has_sfx + 0.15 * punct + 0.1 * short
            + 0.1 * end_bonus + 0.05 * cut_black, 0, 1))
    return out


# --------------------------------------------------------------------------- C. LLM tagging

def beat_line(b: Beat) -> str:
    who = ""
    if b.type == "dialogue":
        who = f"{b.speaker} ({', '.join(b.tones)})" if b.tones else (b.speaker or "")
    return f"{b.beat_id} | {b.start:.1f}-{b.end:.1f} | {b.type} | {who} | {b.text}"


def llm_tags(ep: Episode, llm: LLMClient) -> tuple[dict[str, BeatTag], dict]:
    """One call per episode. Unknown beat_ids are dropped and counted (hallucination guard)."""
    info = {"episode": ep.number, "status": "skipped", "returned": 0, "invalid_ids": []}
    if not llm.available and not llm.cache_dir.exists():
        return {}, info
    template = (PROMPTS / "tag_beats.txt").read_text(encoding="utf-8")
    prompt = template.format(logline=ep.logline or "(none)", beats="\n".join(beat_line(b) for b in ep.beats))
    try:
        resp: TagResponse = llm.complete_json(prompt, role="tagger", schema=TagResponse, tag=f"tag_E{ep.number:02d}")
    except (LLMError, Exception) as e:  # never kill the run
        info["status"] = f"failed: {str(e)[:120]}"
        log.warning("Tagging failed for episode %s: %s", ep.number, e)
        return {}, info
    valid = {b.beat_id for b in ep.beats}
    tags = {}
    for t in resp.tags:
        if t.beat_id in valid:
            tags[t.beat_id] = t
        else:
            info["invalid_ids"].append(t.beat_id)
    info.update(status="ok", returned=len(resp.tags))
    return tags, info


# --------------------------------------------------------------------------- heat

class EpisodeSignals(BaseModel):
    episode: int
    struct: dict[str, float]
    tone: dict[str, float]
    tags: dict[str, BeatTag]
    heat: dict[str, float]          # per beat
    grid_t: list[float]             # heat curve timeline (s)
    grid_heat: list[float]
    weights_used: dict[str, float]


def compute_signals(ep: Episode, llm: Optional[LLMClient], tone_scores: dict[str, float], cfg: dict) -> tuple[EpisodeSignals, dict]:
    struct = structural_scores(ep)
    tags, info = llm_tags(ep, llm) if llm is not None else ({}, {"episode": ep.number, "status": "no llm"})

    # tone per beat: max over the beat's tone words (dialogue only); other beats inherit nearby dialogue tone
    tone: dict[str, float] = {}
    for b in ep.beats:
        if b.type == "dialogue" and b.tones:
            vals = [tone_scores[t] for t in b.tones if t in tone_scores]
            tone[b.beat_id] = max(vals) if vals else 0.5
    dl = [(b.start, b.end, tone[b.beat_id]) for b in ep.beats if b.beat_id in tone]
    for b in ep.beats:
        if b.beat_id not in tone:
            ov = [v for s, e, v in dl if s <= b.end + 1 and e >= b.start - 1]
            tone[b.beat_id] = max(ov) if ov else 0.3

    w_llm, w_tone, w_struct = cfg["w_llm"], cfg["w_tone"], cfg["w_struct"]
    if not tags:  # degraded mode: renormalise without the LLM signal
        w_llm = 0.0
    if not tone_scores:
        w_tone = 0.0
    tot = (w_llm + w_tone + w_struct) or 1.0
    w = {"llm": w_llm / tot, "tone": w_tone / tot, "struct": w_struct / tot}

    heat = {}
    for b in ep.beats:
        llm_v = tags[b.beat_id].intensity / 10 if b.beat_id in tags else 0.1
        heat[b.beat_id] = w["llm"] * llm_v + w["tone"] * tone[b.beat_id] + w["struct"] * struct[b.beat_id]

    # curve on a 0.25 s grid: max heat of beats covering t, then rolling max + light smoothing
    dt = 0.25
    grid = np.arange(0, ep.duration + dt, dt)
    h = np.zeros_like(grid)
    for b in ep.beats:
        i0, i1 = int(b.start / dt), int(np.ceil(b.end / dt)) + 1
        h[i0:i1] = np.maximum(h[i0:i1], heat[b.beat_id])
    win = max(1, int(cfg.get("smooth_window_s", 2.0) / dt))
    rolled = np.array([h[max(0, i - win // 2): i + win // 2 + 1].max() for i in range(len(h))])
    kernel = np.ones(3) / 3
    smooth = np.convolve(rolled, kernel, mode="same")
    sig = EpisodeSignals(episode=ep.number, struct=struct, tone=tone, tags=tags, heat=heat,
                         grid_t=[round(float(x), 2) for x in grid], grid_heat=[round(float(x), 4) for x in smooth],
                         weights_used=w)
    return sig, info
