"""Step 6 - ranking with a simple diversity rule."""
from __future__ import annotations

from .candidates import Candidate


def final_score(c: Candidate, cfg: dict) -> float:
    s = c.scores
    min_ep = min((seg.episode for seg in c.segments), default=50)
    early_bonus = cfg.get("early_ep_bonus", 0.05) if min_ep <= 5 else 0.0
    if "judge_hook" in s:  # judged by the cold-viewer LLM (scores are 1-5)
        n = lambda k: (float(s.get(k, 3)) - 1.0) / 4.0
        return (cfg["w_judge_hook"] * n("judge_hook") + cfg["w_judge_cliff"] * n("judge_cliff")
                + cfg["w_standalone"] * n("judge_standalone") + cfg["w_heat"] * float(s.get("rule_score", 0))
                - cfg["w_spoiler"] * n("judge_spoiler") + early_bonus)
    return float(s.get("rule_score", 0.0)) + early_bonus * 0.5


def rank(cands: list[Candidate], cfg: dict, top_k: int = 10) -> list[Candidate]:
    """Sort by final score; within the top_k allow at most N promos per episode, rest follow."""
    for c in cands:
        c.scores["final"] = round(final_score(c, cfg), 4)
    ordered = sorted(cands, key=lambda c: c.scores["final"], reverse=True)
    top, rest, per_ep = [], [], {}
    cap = cfg["max_per_episode_top10"]
    for c in ordered:
        eps = {s.episode for s in c.segments}
        if len(top) < top_k and all(per_ep.get(e, 0) < cap for e in eps):
            top.append(c)
            for e in eps:
                per_ep[e] = per_ep.get(e, 0) + 1
        else:
            rest.append(c)
    return top + rest
