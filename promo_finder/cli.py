"""End-to-end Promo Finder pipeline and CLI."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import yaml

from .candidates import Candidate, episode_candidates, nms
from .judge import judge_candidate
from .llm import LLMClient
from .montages import generate_montages
from .parse import Episode, load_show
from .plots import plot_episode_heat
from .rank import rank
from .refine import refine_candidate
from .signals import EpisodeSignals, compute_signals
from .tone import score_tones
from .validate import validate

log = logging.getLogger(__name__)


def _post_first_rationale(rank_idx: int, c: Candidate) -> str:
    ep_nums = sorted({s.episode for s in c.segments})
    ep_str = f"Ep {ep_nums[0]}" if len(ep_nums) == 1 else f"Eps {','.join(map(str, ep_nums))}"
    hook_s = c.scores.get("judge_hook", c.scores.get("hook", 0))
    cliff_s = c.scores.get("judge_cliff", c.scores.get("cliff", 0))
    stand_s = c.scores.get("judge_standalone", 5)
    spoil_s = c.scores.get("judge_spoiler", 1)
    if rank_idx <= 3:
        return (f"Post wave 1 ({ep_str}): top standalone clarity ({stand_s}/5), "
                f"hook ({hook_s}/5) & cliffhanger ({cliff_s}/5) with low spoiler risk ({spoil_s}/5).")
    if ep_nums[0] <= 5:
        return f"Early-episode hook ({ep_str}): establishes premise with zero backstory needed."
    return f"High-intensity escalation ({ep_str}, final={c.scores.get('final', 0):.3f}): strong mid-series conversion test."


def run_pipeline(
    input_path: str | Path,
    cfg_path: str | Path = "config.yaml",
    out_dir: Optional[str | Path] = None,
    offline: bool = False,
    skip_refine: bool = False,
    skip_judge: bool = False,
    skip_tone: bool = False,
    skip_montages: bool = False,
    max_episodes: Optional[int] = None,
) -> Path:
    t0 = time.time()
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if skip_tone:
        cfg["tone"]["enabled"] = False

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else Path("outputs") / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, out / "config_used.yaml")

    # Step 1: Parse
    episodes, parse_warnings = load_show(input_path)
    if max_episodes:
        episodes = episodes[:max_episodes]
    ep_map: dict[int, Episode] = {e.number: e for e in episodes}

    # Step 2B: Tone embeddings across all unique tone words
    all_tones = sorted({t for e in episodes for b in e.beats for t in b.tones})
    tone_scores = score_tones(all_tones, cfg["tone"], cache_dir=Path(".cache"))
    if tone_scores:
        with open(out / "tone_scores.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tone_word", "intensity_score"])
            for k, v in sorted(tone_scores.items(), key=lambda x: x[1], reverse=True):
                w.writerow([k, round(v, 4)])

    # Step 2A + 2C: Signals & Heat Curve + Step 3A: Candidates
    llm = LLMClient(cfg["llm"], raw_log_dir=out / "llm_raw", offline=offline)
    signals_map: dict[int, EpisodeSignals] = {}
    tag_audit: list[dict] = []
    raw_cands: list[Candidate] = []

    for ep in episodes:
        sig, info = compute_signals(ep, llm, tone_scores, cfg["heat"])
        signals_map[ep.number] = sig
        tag_audit.append(info)
        raw_cands.extend(episode_candidates(ep, sig, cfg["candidates"]))

    all_cands = nms(raw_cands, cfg["candidates"]["overlap_iou"])

    # Validate initial candidates
    valid_cands: list[Candidate] = []
    validation_drops: list[dict] = []
    for c in all_cands:
        errs = validate(c, ep_map, cfg["candidates"])
        if errs:
            validation_drops.append({"promo_id": c.promo_id, "stage": "pre_refine", "errors": errs})
        else:
            valid_cands.append(c)

    # Pre-rank by rule_score; guarantee best candidate from each of Episodes 1..early_n + top by rule_score
    valid_cands.sort(key=lambda c: c.scores.get("rule_score", 0), reverse=True)
    top_n = cfg["rank"]["top_n_refine"]
    early_max = cfg["rank"].get("early_episodes_guarantee", 5)
    shortlist: list[Candidate] = []
    seen_ids: set[str] = set()
    for ep_i in range(1, early_max + 1):
        ep_best = [c for c in valid_cands if any(s.episode == ep_i for s in c.segments)]
        if ep_best:
            shortlist.append(ep_best[0])
            seen_ids.add(ep_best[0].promo_id)
    for c in valid_cands:
        if len(shortlist) >= top_n:
            break
        if c.promo_id not in seen_ids:
            shortlist.append(c)
            seen_ids.add(c.promo_id)
    remainder = [c for c in valid_cands if c.promo_id not in seen_ids]

    refined_list: list[Candidate] = []
    for c in shortlist:
        r = refine_candidate(c, ep_map, llm, cfg["candidates"]) if not skip_refine else c
        errs = validate(r, ep_map, cfg["candidates"])
        if errs:
            validation_drops.append({"promo_id": r.promo_id, "stage": "post_refine", "errors": errs})
            r = c  # fall back to pre-refined valid candidate
        j = judge_candidate(r, ep_map, llm) if not skip_judge else r
        refined_list.append(j)

    # Step 3B / Phase 8: Cross-episode montages (code shortlist -> LLM pick -> validator -> judge gate)
    montage_cands: list[Candidate] = []
    montage_audit: dict = {}
    if not skip_montages:
        montage_cands, montage_audit = generate_montages(episodes, signals_map, llm, cfg["candidates"])

    # Deduplicate again in case refinement shifted windows onto each other
    merged = nms(refined_list + montage_cands + remainder, cfg["candidates"]["overlap_iou"])
    ranked = rank(merged, cfg["rank"], top_k=10)

    # Save heat curve plots for top 10 episodes + Episodes 1-5
    top10 = ranked[:10]
    plot_eps = {s.episode for c in top10 for s in c.segments} | {e.number for e in episodes[:5]}
    for ep_num in sorted(plot_eps):
        ep = ep_map[ep_num]
        ep_promos = [c for c in top10 if any(s.episode == ep_num for s in c.segments)]
        plot_episode_heat(ep, signals_map[ep_num], ep_promos, out / "heat_curves" / f"E{ep_num:02d}.png")

    # Export promos.json and promos.csv
    records = []
    for i, c in enumerate(ranked, 1):
        rationale = _post_first_rationale(i, c)
        rec = {
            "rank": i,
            "promo_id": c.promo_id,
            "kind": c.kind,
            "duration_s": c.duration,
            "segments": [s.model_dump() for s in c.segments],
            "hook": c.hook,
            "cliffhanger": c.cliffhanger,
            "why": c.why,
            "post_order_rationale": rationale,
            "scores": c.scores,
            "flags": c.flags,
        }
        records.append(rec)

    summary = {
        "input": str(input_path),
        "episodes_parsed": len(episodes),
        "parse_warnings": parse_warnings,
        "unique_tone_words": len(all_tones),
        "tone_words_scored": len(tone_scores),
        "total_candidates": len(all_cands),
        "valid_candidates": len(valid_cands),
        "shortlist_refined_judged": len(refined_list),
        "montage_audit": montage_audit,
        "validation_drops": validation_drops,
        "tag_invalid_ids_total": sum(len(x.get("invalid_ids", [])) for x in tag_audit),
        "llm_stats": llm.stats,
        "runtime_s": round(time.time() - t0, 2),
        "promos": records,
    }
    (out / "promos.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "promo_id", "episodes", "segments", "duration_s",
            "final_score", "rule_score", "judge_hook", "judge_cliff", "judge_standalone", "judge_spoiler",
            "hook", "cliffhanger", "why", "post_order_rationale", "judge_retell", "judge_question",
        ])
        for r in records:
            segs_str = " + ".join(f"E{s['episode']:02d} {s['start_ts']}-{s['end_ts']}" for s in r["segments"])
            eps_str = ",".join(str(s["episode"]) for s in r["segments"])
            sc = r["scores"]
            w.writerow([
                r["rank"], r["promo_id"], eps_str, segs_str, r["duration_s"],
                sc.get("final", ""), sc.get("rule_score", ""),
                sc.get("judge_hook", ""), sc.get("judge_cliff", ""),
                sc.get("judge_standalone", ""), sc.get("judge_spoiler", ""),
                r["hook"], r["cliffhanger"], r["why"], r["post_order_rationale"],
                sc.get("judge_retell", ""), sc.get("judge_question", ""),
            ])

    return out


def main():
    p = argparse.ArgumentParser(description="Find and rank microdrama promos from screenplays.")
    p.add_argument("input", help="Path to show .zip or folder of episode .txt files")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--out", default=None, help="Output directory (default: outputs/run_<timestamp>)")
    p.add_argument("--offline", action="store_true", help="Run without live LLM calls (cache or rule-only)")
    p.add_argument("--skip-refine", action="store_true", help="Skip Step 4 LLM boundary refinement")
    p.add_argument("--skip-judge", action="store_true", help="Skip Step 5 cold-viewer LLM judge")
    p.add_argument("--skip-tone", action="store_true", help="Skip Step 2B tone embeddings")
    p.add_argument("--skip-montages", action="store_true", help="Skip Step 3B cross-episode montages")
    p.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes (for quick tests)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = run_pipeline(
        args.input, cfg_path=args.config, out_dir=args.out, offline=args.offline,
        skip_refine=args.skip_refine, skip_judge=args.skip_judge, skip_tone=args.skip_tone,
        skip_montages=args.skip_montages, max_episodes=args.max_episodes,
    )
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
