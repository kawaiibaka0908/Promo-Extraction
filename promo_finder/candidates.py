"""Step 3A - single-stretch promo candidates (pure code).

hook anchors -> start (hook inside first 3 s, on a clean boundary)
-> try every clean end 20-60 s later -> cliffhanger score -> window score
-> overlap removal (temporal IoU).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
from pydantic import BaseModel

from .parse import Beat, Episode, fmt_ts
from .signals import EpisodeSignals


class Segment(BaseModel):
    episode: int
    start: float
    end: float
    start_ts: str = ""
    end_ts: str = ""

    def model_post_init(self, _):
        self.start_ts, self.end_ts = fmt_ts(self.start), fmt_ts(self.end)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Candidate(BaseModel):
    promo_id: str
    kind: str = "single"                      # single | montage
    segments: list[Segment]
    hook_beat: str = ""
    cliff_beat: str = ""
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    scores: dict[str, float] = {}
    beat_ids: list[str] = []
    flags: list[str] = []

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)


# --------------------------------------------------------------------------- helpers

def clean_boundaries(ep: Episode) -> list[float]:
    """Cut points that do not fall inside any spoken line (never cut mid-sentence)."""
    dlg = [(b.start, b.end) for b in ep.beats if b.type == "dialogue"]
    pts = set(ep.cut_points()) | {round(b.start, 2) for b in ep.beats if b.type in ("dialogue", "action")}
    ok = [p for p in pts if not any(s + 0.05 < p < e - 0.05 for s, e in dlg)]
    return sorted(ok)


def _heat_at(sig: EpisodeSignals, t0: float, t1: float, how: str = "max") -> float:
    t = np.asarray(sig.grid_t)
    h = np.asarray(sig.grid_heat)
    m = (t >= t0) & (t <= t1)
    if not m.any():
        return 0.0
    return float(h[m].max() if how == "max" else h[m].mean())


def _longest_low_run(sig: EpisodeSignals, t0: float, t1: float, thr: float) -> float:
    t = np.asarray(sig.grid_t)
    h = np.asarray(sig.grid_heat)
    m = (t >= t0) & (t <= t1)
    run = best = 0.0
    dt = (t[1] - t[0]) if len(t) > 1 else 0.25
    for v in h[m]:
        run = run + dt if v < thr else 0.0
        best = max(best, run)
    return best


def _beats_in(ep: Episode, t0: float, t1: float) -> list[Beat]:
    return [b for b in ep.beats if b.start >= t0 - 0.01 and b.end <= t1 + 0.01]


def _best_text(beats: list[Beat], sig: EpisodeSignals, prefer_last: bool = False) -> Optional[Beat]:
    if not beats:
        return None
    ranked = sorted(beats, key=lambda b: (sig.heat[b.beat_id] + (0.05 if b.type == "dialogue" else 0)
                                          + (0.001 * b.start if prefer_last else -0.001 * b.start)), reverse=True)
    return ranked[0]


def _describe(b: Optional[Beat]) -> str:
    if b is None:
        return ""
    if b.type == "dialogue":
        return f'{b.speaker.title() if b.speaker else "?"}: "{b.text}"'
    return b.text


def iou(a: Candidate, b: Candidate) -> float:
    """Temporal IoU over (episode, time) segments."""
    inter = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode:
                inter += max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
    union = a.duration + b.duration - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- main

def episode_candidates(ep: Episode, sig: EpisodeSignals, cfg: dict) -> list[Candidate]:
    min_len, max_len, hook_w = cfg["min_len_s"], cfg["max_len_s"], cfg["hook_window_s"]
    bounds = clean_boundaries(ep)
    dur = ep.duration
    content = [b for b in ep.beats if b.type != "camera" or sig.heat[b.beat_id] > 0]
    heats = np.array([sig.heat[b.beat_id] for b in content])
    thr = float(np.quantile(heats, 1 - cfg["anchor_top_pct"])) if len(heats) else 1.0

    anchors: list[Beat] = []
    for b in ep.beats:
        tag = sig.tags.get(b.beat_id)
        if (tag and tag.hook_worthy) or sig.heat[b.beat_id] >= thr:
            anchors.append(b)
    if ep.beats:
        anchors.append(ep.beats[0])  # cold open
    low_thr = float(np.quantile(sig.grid_heat, 0.3)) if sig.grid_heat else 0.2

    cands: list[Candidate] = []
    seen = set()
    for a in anchors:
        # start: latest clean boundary <= anchor.start such that the hook lands inside the first `hook_w` s
        starts = [p for p in bounds if a.start - hook_w + 0.5 <= p <= a.start + 0.01]
        start = starts[0] if starts else None   # earliest allowed = a little lead-in
        if start is None:
            continue
        if start in seen:
            continue
        seen.add(start)
        hook_tag = sig.tags.get(a.beat_id)
        hook_score = 0.6 * _heat_at(sig, start, start + hook_w) + 0.4 * (1.0 if hook_tag and hook_tag.hook_worthy else 0.0)

        ends = [p for p in bounds if start + min_len <= p <= min(start + max_len, dur + 0.01)]
        scored = []
        for e in ends:
            tail = _beats_in(ep, e - 4, e)
            unresolved = any(sig.tags.get(b.beat_id) and sig.tags[b.beat_id].unresolved for b in tail)
            cut_black = any(b.type == "camera" and re.search(r"cut to black|fade to black", b.text, re.I) for b in tail)
            last_dlg = [b for b in tail if b.type == "dialogue"]
            open_line = bool(last_dlg and re.search(r"[?!]|\.\.\.$", last_dlg[-1].text))
            end_heat = _heat_at(sig, e - 4, e)
            # tension continuing right after the cut = we cut *before* the resolution
            after = _heat_at(sig, e, e + 4) if e < dur - 0.5 else 0.0
            cliff = (0.45 * end_heat + 0.25 * unresolved + 0.15 * cut_black + 0.1 * open_line + 0.05 * min(1.0, after / max(end_heat, 1e-6)))
            scored.append((cliff, e, unresolved, cut_black))
        scored.sort(reverse=True)
        for cliff, e, unresolved, cut_black in scored[: cfg["ends_per_anchor"]]:
            mean_heat = _heat_at(sig, start, e, "mean")
            dead = _longest_low_run(sig, start, e, low_thr)
            dead_pen = 0.15 if dead > cfg["dead_air_s"] else 0.0
            inside = _beats_in(ep, start, e)
            ctx = [sig.tags[b.beat_id].needs_context for b in inside if b.beat_id in sig.tags]
            ctx_pen = 0.1 * (sum(c == "high" for c in ctx) / max(len(ctx), 1))
            score = 0.4 * hook_score + 0.4 * cliff + 0.2 * mean_heat - dead_pen - ctx_pen
            cliff_beat = _best_text(_beats_in(ep, e - 6, e), sig, prefer_last=True)
            flags = (["dead_air"] if dead_pen else []) + (["cut_to_black"] if cut_black else []) + (["unresolved_tag"] if unresolved else [])
            cands.append(Candidate(
                promo_id=f"E{ep.number:02d}_{start:06.2f}_{e:06.2f}",
                segments=[Segment(episode=ep.number, start=start, end=e)],
                hook_beat=a.beat_id, cliff_beat=cliff_beat.beat_id if cliff_beat else "",
                hook=_describe(a), cliffhanger=_describe(cliff_beat),
                why=(hook_tag.why if hook_tag and hook_tag.why else "High-intensity opening moment") ,
                scores={"hook": round(hook_score, 3), "cliff": round(cliff, 3), "mean_heat": round(mean_heat, 3),
                        "dead_air_s": round(dead, 1), "rule_score": round(score, 4)},
                beat_ids=[b.beat_id for b in inside], flags=flags,
            ))
    return nms(cands, cfg["overlap_iou"])


def nms(cands: list[Candidate], thr: float) -> list[Candidate]:
    keep: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.scores.get("rule_score", 0), reverse=True):
        if all(iou(c, k) <= thr for k in keep):
            keep.append(c)
    return keep
