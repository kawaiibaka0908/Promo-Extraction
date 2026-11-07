"""Step 4 - LLM boundary refinement and copy writing (hook, cliffhanger, why)."""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from .candidates import Candidate, Segment, clean_boundaries
from .llm import LLMClient, LLMError
from .parse import Episode
from .signals import beat_line
from .validate import validate

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


class RefineResponse(BaseModel):
    start_beat: str
    end_beat: str
    hook_beat: str
    hook: str
    cliffhanger: str
    why: str


def _snap_to_clean(t: float, bounds: list[float], mode: str = "nearest") -> float:
    if not bounds:
        return round(t, 2)
    if mode == "floor":
        le = [b for b in bounds if b <= t + 0.05]
        return le[-1] if le else bounds[0]
    if mode == "ceil":
        ge = [b for b in bounds if b >= t - 0.05]
        return ge[0] if ge else bounds[-1]
    return min(bounds, key=lambda b: abs(b - t))


def refine_candidate(c: Candidate, episodes: dict[int, Episode], llm: LLMClient, cfg: dict) -> Candidate:
    """Ask the LLM to tighten start/end beats and write punchy hook/cliffhanger/why.
    If the refined boundaries violate hard rules, keep the original valid boundaries
    and only adopt the LLM's text descriptions."""
    if not llm.available and not llm.cache_dir.exists():
        return c
    if c.kind != "single" or not c.segments:
        return c

    seg = c.segments[0]
    ep = episodes[seg.episode]
    ctx_beats = [b for b in ep.beats if b.end >= seg.start - 10.0 and b.start <= seg.end + 10.0]
    in_beats = [b for b in ep.beats if b.start >= seg.start - 0.05 and b.end <= seg.end + 0.05]
    if not in_beats:
        return c

    template = (PROMPTS / "refine.txt").read_text(encoding="utf-8")
    prompt = template.format(
        episode=ep.number,
        logline=ep.logline or "(none)",
        beats="\n".join(beat_line(b) for b in ctx_beats),
        start_beat=in_beats[0].beat_id,
        start_s=seg.start,
        end_beat=in_beats[-1].beat_id,
        end_s=seg.end,
        duration_s=c.duration,
    )
    try:
        resp: RefineResponse = llm.complete_json(
            prompt, role="refiner", schema=RefineResponse, tag=f"refine_{c.promo_id}"
        )
    except (LLMError, Exception) as e:
        log.warning("Refine failed for %s: %s", c.promo_id, e)
        return c

    # always update text copy if non-empty
    updated = c.model_copy(deep=True)
    if resp.hook.strip():
        updated.hook = resp.hook.strip()
    if resp.cliffhanger.strip():
        updated.cliffhanger = resp.cliffhanger.strip()
    if resp.why.strip():
        updated.why = resp.why.strip()

    # attempt boundary nudge via beat_ids
    bmap = {b.beat_id: b for b in ep.beats}
    if resp.start_beat in bmap and resp.end_beat in bmap:
        bounds = clean_boundaries(ep)
        new_start = _snap_to_clean(bmap[resp.start_beat].start, bounds, mode="floor")
        new_end = _snap_to_clean(bmap[resp.end_beat].end, bounds, mode="ceil")
        new_hook = resp.hook_beat if resp.hook_beat in bmap else updated.hook_beat
        # if hook beat isn't within 3s of new_start, pick the first beat inside the new window
        if new_hook not in bmap or not (new_start - 0.01 <= bmap[new_hook].start <= new_start + cfg["hook_window_s"] + 0.01):
            first_in = [b for b in ep.beats if new_start - 0.01 <= b.start <= new_start + cfg["hook_window_s"]]
            if first_in:
                new_hook = first_in[0].beat_id

        trial = updated.model_copy(deep=True)
        trial.segments = [Segment(episode=ep.number, start=new_start, end=new_end)]
        trial.hook_beat = new_hook
        trial.cliff_beat = resp.end_beat
        trial.beat_ids = [b.beat_id for b in ep.beats if b.start >= new_start - 0.01 and b.end <= new_end + 0.01]
        if not validate(trial, episodes, cfg):
            trial.promo_id = f"E{ep.number:02d}_{new_start:06.2f}_{new_end:06.2f}"
            if "refined_bounds" not in trial.flags:
                trial.flags.append("refined_bounds")
            return trial
        else:
            if "refine_bounds_rejected" not in updated.flags:
                updated.flags.append("refine_bounds_rejected")
    return updated
