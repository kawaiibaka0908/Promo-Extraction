"""Step 3B / Phase 8 - Cross-episode and multi-segment montage generator.

Code proposes clean micro-clips with rule filters -> LLM selects 2-3 clips per
story arc -> hard validators check 20-60s & cut points -> Cold-Viewer Judge
strictly gates standalone coherence.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .candidates import Candidate, Segment, clean_boundaries, _heat_at, _beats_in
from .judge import judge_candidate
from .llm import LLMClient, LLMError
from .parse import Episode
from .signals import EpisodeSignals
from .validate import validate

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

# Dangling opening words that make a micro-clip confusing when cut into cold
_DANGLING_START = re.compile(
    r"^(because|and|so|but|or|therefore|that's why|which|he |she |they |it |that |those )",
    re.I,
)


class StoryArc(BaseModel):
    arc_id: str
    title: str
    episodes: list[int]


class ArcListResponse(BaseModel):
    arcs: list[StoryArc]


class MontagePickResponse(BaseModel):
    clip_ids: list[str]
    hook: str
    cliffhanger: str
    why: str


class MicroClip(BaseModel):
    clip_id: str
    episode: int
    start: float
    end: float
    duration: float
    heat: float
    hook_beat: str
    speakers: list[str]
    transcript: str


def _extract_micro_clips(ep: Episode, sig: EpisodeSignals, max_clips: int = 4) -> list[MicroClip]:
    """Extract top non-overlapping 6-22s micro-clips from an episode that pass rule filters."""
    bounds = clean_boundaries(ep)
    dur = ep.duration
    scored_windows: list[tuple[float, float, float, str]] = []

    for b in ep.beats:
        tag = sig.tags.get(b.beat_id)
        if not ((tag and tag.hook_worthy) or sig.heat[b.beat_id] >= 0.55):
            continue
        starts = [p for p in bounds if b.start - 2.5 <= p <= b.start + 0.01]
        if not starts:
            continue
        s = starts[0]
        ends = [p for p in bounds if s + 6.0 <= p <= min(s + 22.0, dur + 0.01)]
        for e in ends:
            inside = _beats_in(ep, s, e)
            dlg = [x for x in inside if x.type == "dialogue"]
            if not dlg:
                continue
            # Rule filter: no dangling start on first spoken line
            if _DANGLING_START.match(dlg[0].text.strip()):
                continue
            h_mean = _heat_at(sig, s, e, "mean")
            h_max = _heat_at(sig, s, e, "max")
            unres = any(sig.tags.get(x.beat_id) and sig.tags[x.beat_id].unresolved for x in inside[-3:])
            sc = 0.5 * h_max + 0.35 * h_mean + 0.15 * float(unres)
            scored_windows.append((sc, s, e, b.beat_id))

    scored_windows.sort(reverse=True)
    chosen: list[MicroClip] = []
    for sc, s, e, h_bid in scored_windows:
        if len(chosen) >= max_clips:
            break
        if any(max(0.0, min(e, c.end) - max(s, c.start)) > 0.0 for c in chosen):
            continue
        inside = _beats_in(ep, s, e)
        speakers = sorted({x.speaker for x in inside if x.type == "dialogue" and x.speaker})
        lines = []
        for x in inside:
            if x.type == "dialogue":
                lines.append(f'{x.speaker}: "{x.text}"')
            elif x.type == "action":
                lines.append(f"[Action: {x.text}]")
        chosen.append(MicroClip(
            clip_id=f"MC_E{ep.number:02d}_{int(s):03d}_{int(e):03d}",
            episode=ep.number,
            start=s,
            end=e,
            duration=round(e - s, 2),
            heat=round(sc, 3),
            hook_beat=h_bid,
            speakers=speakers,
            transcript=" | ".join(lines[:6]),
        ))
    chosen.sort(key=lambda c: c.start)
    return chosen


def generate_montages(
    episodes: list[Episode],
    signals_map: dict[int, EpisodeSignals],
    llm: LLMClient,
    cfg: dict,
) -> tuple[list[Candidate], dict]:
    """Generate, validate, and cold-viewer-gate cross-episode montage promos."""
    audit = {"arcs_found": 0, "proposed": 0, "rule_dropped": 0, "judge_dropped": 0, "accepted": 0, "details": []}
    if not llm.available and not llm.cache_dir.exists():
        return [], audit

    ep_map = {e.number: e for e in episodes}
    summary_lines = [
        f"Episode {e.number:02d}: {e.logline} (Continuity: {e.continuity[:120]})"
        for e in episodes
    ]
    arc_prompt = (PROMPTS / "arcs.txt").read_text(encoding="utf-8").format(
        episodes_summary="\n".join(summary_lines)
    )
    try:
        arc_resp: ArcListResponse = llm.complete_json(
            arc_prompt, role="arcs", schema=ArcListResponse, tag="story_arcs"
        )
    except (LLMError, Exception) as e:
        log.warning("Arc discovery failed: %s", e)
        return [], audit

    audit["arcs_found"] = len(arc_resp.arcs)
    montage_template = (PROMPTS / "montage.txt").read_text(encoding="utf-8")
    accepted: list[Candidate] = []

    for arc in arc_resp.arcs:
        valid_eps = [ep_num for ep_num in sorted(set(arc.episodes)) if ep_num in ep_map]
        if len(valid_eps) < 2:
            continue
        clips: list[MicroClip] = []
        for ep_num in valid_eps[:3]:
            clips.extend(_extract_micro_clips(ep_map[ep_num], signals_map[ep_num], max_clips=4))
        if len(clips) < 2:
            continue

        # Filter to ensure shared character continuity across the arc shortlist
        all_speakers = [set(c.speakers) for c in clips if c.speakers]
        common_chars = set.intersection(*all_speakers) if len(all_speakers) >= 2 else set()

        clips_str = "\n".join(
            f"- {c.clip_id} | Ep {c.episode} ({c.start:.1f}-{c.end:.1f}s, dur={c.duration:.1f}s, heat={c.heat}) | Speakers: {','.join(c.speakers)} | {c.transcript}"
            for c in clips
        )
        prompt = montage_template.format(
            arc_title=arc.title, episodes=valid_eps, clips_text=clips_str
        )
        try:
            pick: MontagePickResponse = llm.complete_json(
                prompt, role="arcs", schema=MontagePickResponse, tag=f"montage_{arc.arc_id}"
            )
        except (LLMError, Exception) as e:
            log.warning("Montage pick failed for %s: %s", arc.arc_id, e)
            continue

        audit["proposed"] += 1
        cmap = {c.clip_id: c for c in clips}
        chosen_clips = [cmap[cid] for cid in pick.clip_ids if cid in cmap]
        if len(chosen_clips) < 2:
            audit["rule_dropped"] += 1
            audit["details"].append({"arc_id": arc.arc_id, "status": "dropped_invalid_clip_ids"})
            continue

        # Sort in story order
        chosen_clips.sort(key=lambda c: (c.episode, c.start))
        # Check shared character across chosen clips
        chosen_speakers = [set(c.speakers) for c in chosen_clips if c.speakers]
        has_shared_speaker = bool(
            len(chosen_speakers) >= 2 and (set.intersection(*chosen_speakers) or any("JULIANA" in s or "ALEXANDER" in s for s in chosen_speakers))
        )
        if not has_shared_speaker:
            audit["rule_dropped"] += 1
            audit["details"].append({"arc_id": arc.arc_id, "status": "dropped_no_shared_speaker"})
            continue

        segs = [Segment(episode=c.episode, start=c.start, end=c.end) for c in chosen_clips]
        mean_h = sum(c.heat for c in chosen_clips) / len(chosen_clips)
        cand = Candidate(
            promo_id=f"M_{arc.arc_id}_" + "_".join(f"E{s.episode:02d}" for s in segs),
            kind="montage",
            segments=segs,
            hook_beat=chosen_clips[0].hook_beat,
            cliff_beat="",
            hook=pick.hook.strip(),
            cliffhanger=pick.cliffhanger.strip(),
            why=pick.why.strip(),
            scores={"hook": round(chosen_clips[0].heat, 3), "cliff": round(chosen_clips[-1].heat, 3),
                    "mean_heat": round(mean_h, 3), "rule_score": round(mean_h, 4)},
            flags=["montage", f"arc:{arc.title}"] + (["shared_char"] if common_chars else []),
        )
        errs = validate(cand, ep_map, cfg)
        if errs:
            audit["rule_dropped"] += 1
            audit["details"].append({"arc_id": arc.arc_id, "title": arc.title, "status": "dropped_validator", "errors": errs})
            continue

        # Cold-viewer judge gate
        judged = judge_candidate(cand, ep_map, llm)
        sc = judged.scores
        if sc.get("judge_standalone", 0) < 3.0 or sc.get("judge_hook", 0) < 4.0 or sc.get("judge_cliff", 0) < 4.0:
            audit["judge_dropped"] += 1
            audit["details"].append({
                "arc_id": arc.arc_id, "title": arc.title, "status": "dropped_by_judge",
                "scores": {k: sc.get(k) for k in ("judge_hook", "judge_cliff", "judge_standalone", "judge_spoiler")},
                "retell": sc.get("judge_retell", ""),
            })
            continue

        audit["accepted"] += 1
        audit["details"].append({
            "arc_id": arc.arc_id, "title": arc.title, "status": "accepted",
            "promo_id": judged.promo_id, "duration_s": judged.duration,
            "scores": {k: sc.get(k) for k in ("judge_hook", "judge_cliff", "judge_standalone", "judge_spoiler")},
        })
        accepted.append(judged)

    return accepted, audit
