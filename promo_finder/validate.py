"""Step 5a - hard validators. Code, not AI: every promo must obey these rules."""
from __future__ import annotations

from .candidates import Candidate, clean_boundaries
from .parse import Episode


def validate(c: Candidate, episodes: dict[int, Episode], cfg: dict) -> list[str]:
    """Return a list of rule violations (empty list = valid)."""
    errs: list[str] = []
    lo, hi, hook_w = cfg["min_len_s"], cfg["max_len_s"], cfg["hook_window_s"]
    if not (lo - 0.01 <= c.duration <= hi + 0.01):
        errs.append(f"duration {c.duration:.1f}s outside {lo}-{hi}s")
    prev = None
    for s in c.segments:
        ep = episodes.get(s.episode)
        if ep is None:
            errs.append(f"episode {s.episode} does not exist")
            continue
        if s.end <= s.start:
            errs.append(f"segment E{s.episode} {s.start}-{s.end} has non-positive length")
        if s.start < -0.01 or s.end > ep.duration + 0.01:
            errs.append(f"segment E{s.episode} {s.start}-{s.end} outside episode (0-{ep.duration})")
        bounds = set(round(x, 2) for x in clean_boundaries(ep))
        if round(s.start, 2) not in bounds:
            errs.append(f"start {s.start} in E{s.episode} is not a clean cut point")
        if round(s.end, 2) not in bounds:
            errs.append(f"end {s.end} in E{s.episode} is not a clean cut point")
        if prev is not None:
            if (s.episode, s.start) < (prev.episode, prev.start) and "hook_first" not in c.flags:
                errs.append("segments out of story order")
            if s.episode == prev.episode and s.start < prev.end - 0.01:
                errs.append(f"overlapping segments in E{s.episode}: {prev.start}-{prev.end} and {s.start}-{s.end}")
        prev = s
    # hook must land within the first `hook_w` seconds of the promo
    first = c.segments[0] if c.segments else None
    if first and c.hook_beat and first.episode in episodes:
        hb = next((b for b in episodes[first.episode].beats if b.beat_id == c.hook_beat), None)
        if hb is None:
            errs.append(f"hook beat {c.hook_beat} not found in first segment's episode")
        elif not (first.start - 0.01 <= hb.start <= first.start + hook_w + 0.01):
            errs.append(f"hook beat starts at {hb.start}, not within {hook_w}s of promo start {first.start}")
    return errs
