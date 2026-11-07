"""Step 5b - Cold-viewer LLM judge.

Sees ONLY the clip text (never the episode logline or surrounding context),
retells what happened in 1-2 sentences, and scores hook, cliffhanger,
standalone clarity, and spoiler risk from 1 to 5.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from .candidates import Candidate
from .llm import LLMClient, LLMError
from .parse import Episode

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


class JudgeResponse(BaseModel):
    retell: str
    want_to_know_next: str
    hook: int = Field(ge=1, le=5)
    cliffhanger: int = Field(ge=1, le=5)
    standalone: int = Field(ge=1, le=5)
    spoiler_risk: int = Field(ge=1, le=5)


def format_clip_for_cold_viewer(c: Candidate, episodes: dict[int, Episode]) -> str:
    """Render only the beats inside the promo segments as a bare script excerpt."""
    lines: list[str] = []
    t_offset = 0.0
    for seg in c.segments:
        ep = episodes[seg.episode]
        if len(c.segments) > 1:
            lines.append(f"--- [Cut to segment, +{t_offset:.1f}s] ---")
        for b in ep.beats:
            if b.start >= seg.start - 0.05 and b.end <= seg.end + 0.05:
                rel_s = max(0.0, b.start - seg.start) + t_offset
                rel_e = max(0.0, b.end - seg.start) + t_offset
                if b.type == "dialogue":
                    tone = f" ({', '.join(b.tones)})" if b.tones else ""
                    lines.append(f"[{rel_s:04.1f}-{rel_e:04.1f}s] {b.speaker}{tone}: \"{b.text}\"")
                elif b.type == "action":
                    lines.append(f"[{rel_s:04.1f}-{rel_e:04.1f}s] ACTION: {b.text}")
                elif b.type == "sfx":
                    lines.append(f"[{rel_s:04.1f}-{rel_e:04.1f}s] SFX: {b.text}")
                else:
                    lines.append(f"[{rel_s:04.1f}-{rel_e:04.1f}s] SHOT: {b.text}")
        t_offset += seg.duration
    return "\n".join(lines)


def judge_candidate(c: Candidate, episodes: dict[int, Episode], llm: LLMClient) -> Candidate:
    if not llm.available and not llm.cache_dir.exists():
        return c
    clip_text = format_clip_for_cold_viewer(c, episodes)
    template = (PROMPTS / "judge.txt").read_text(encoding="utf-8")
    prompt = template.format(duration_s=c.duration, clip_text=clip_text)
    try:
        resp: JudgeResponse = llm.complete_json(
            prompt, role="judge", schema=JudgeResponse, tag=f"judge_{c.promo_id}"
        )
    except (LLMError, Exception) as e:
        log.warning("Judge failed for %s: %s", c.promo_id, e)
        return c

    updated = c.model_copy(deep=True)
    updated.scores["judge_hook"] = float(resp.hook)
    updated.scores["judge_cliff"] = float(resp.cliffhanger)
    updated.scores["judge_standalone"] = float(resp.standalone)
    updated.scores["judge_spoiler"] = float(resp.spoiler_risk)
    updated.scores["judge_retell"] = resp.retell  # type: ignore[assignment]
    updated.scores["judge_question"] = resp.want_to_know_next  # type: ignore[assignment]
    return updated
