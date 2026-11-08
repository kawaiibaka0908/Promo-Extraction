"""Step 8 / Phase 7 - Evaluation & Ablation against the human-labeled gold set.

Strictly post-hoc: reads finished pipeline outputs (`promos.json`) and evaluates
them against `eval/promo_clips_selection.csv` without ever influencing pipeline
generation or ranking.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .parse import to_seconds


def parse_gold_csv(csv_path: str | Path) -> tuple[list[dict], list[dict]]:
    """Parse `eval/promo_clips_selection.csv` into good (G*) and bad (B*) entries."""
    good, bad = [], []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ep_raw = row["episode"].strip()
            default_eps = [int(x) for x in re.findall(r"\d+", ep_raw)]
            default_ep = default_eps[0] if default_eps else 1

            starts_raw = [x.strip() for x in row["start"].split(";") if x.strip()]
            ends_raw = [x.strip() for x in row["end"].split(";") if x.strip()]
            segs = []
            for s_tok, e_tok in zip(starts_raw, ends_raw):
                m_s = re.match(r"^[Ee](\d+)\s+(.*)$", s_tok)
                m_e = re.match(r"^[Ee](\d+)\s+(.*)$", e_tok)
                ep_num = int(m_s.group(1)) if m_s else default_ep
                s_val = to_seconds(m_s.group(2) if m_s else s_tok)
                e_val = to_seconds(m_e.group(2) if m_e else e_tok)
                if e_val > s_val:
                    segs.append({"episode": ep_num, "start": s_val, "end": e_val})

            by_ep: dict[int, list[dict]] = {}
            for s in segs:
                by_ep.setdefault(s["episode"], []).append(s)
            spans = [
                {"episode": ep_n, "start": min(x["start"] for x in s_list), "end": max(x["end"] for x in s_list)}
                for ep_n, s_list in sorted(by_ep.items())
            ]

            item = {
                "id": row["id"].strip(),
                "label": row["label"].strip().lower(),
                "episodes": sorted(by_ep.keys()),
                "is_cross_ep": len(by_ep) > 1,
                "is_multi_cut": len(segs) > 1,
                "segments": segs,
                "ep_spans": spans,
                "hook": row.get("hook", "").strip(),
                "cliffhanger": row.get("cliffhanger", "").strip(),
                "notes": row.get("notes", "").strip(),
            }
            if item["label"] == "good":
                good.append(item)
            else:
                bad.append(item)
    return good, bad


def _overlap_metrics(pred_segs: list[dict], gold_item: dict) -> dict[str, Any]:
    """Compute both:
    - `subclip_hit`: best coverage of any single gold highlight sub-clip (since 18/20 gold
      entries are multi-cut edits spanning up to 158s, where a continuous 20-60s promo can
      physically cover 1-2 cuts)
    - `beat_coverage`: total fraction of all gold sub-clip seconds covered
    - `best_ep_iou`: best temporal IoU against a per-episode gold span
    - `boundary_err_s`: minimum start/end boundary distance to the matched gold cut/span
    """
    inter_segs = 0.0
    gold_seg_dur = sum(g["end"] - g["start"] for g in gold_item["segments"])
    best_subclip_cov = 0.0
    best_bound_err = 999.0

    for p in pred_segs:
        for g in gold_item["segments"]:
            if p["episode"] == g["episode"]:
                ix = max(0.0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
                inter_segs += ix
                g_len = g["end"] - g["start"]
                if ix > 0 and g_len > 0:
                    best_subclip_cov = max(best_subclip_cov, ix / g_len)
                    err = min(abs(p["start"] - g["start"]), abs(p["end"] - g["end"]))
                    best_bound_err = min(best_bound_err, err)

    beat_coverage = inter_segs / gold_seg_dur if gold_seg_dur > 0 else 0.0

    best_ep_iou = 0.0
    for p in pred_segs:
        for gs in gold_item["ep_spans"]:
            if p["episode"] == gs["episode"]:
                ix = max(0.0, min(p["end"], gs["end"]) - max(p["start"], gs["start"]))
                if ix > 0:
                    u = (p["end"] - p["start"]) + (gs["end"] - gs["start"]) - ix
                    best_ep_iou = max(best_ep_iou, ix / u if u > 0 else 0.0)
                    span_err = (abs(p["start"] - gs["start"]) + abs(p["end"] - gs["end"])) / 2.0
                    best_bound_err = min(best_bound_err, span_err)

    # Strict match: covers >= 50% of all gold highlight seconds or per-episode span IoU >= 0.30
    strict_hit = (best_ep_iou >= 0.30) or (beat_coverage >= 0.50)
    # Highlight-cut match: captures >= 40% of at least one gold highlight cut (or >= 4s overlap on the same dramatic beat)
    highlight_hit = strict_hit or (best_subclip_cov >= 0.40) or (inter_segs >= 4.0)

    return {
        "strict_hit": strict_hit,
        "highlight_hit": highlight_hit,
        "best_subclip_cov": round(best_subclip_cov, 3),
        "beat_coverage": round(beat_coverage, 3),
        "best_ep_iou": round(best_ep_iou, 3),
        "overlap_s": round(inter_segs, 2),
        "boundary_err_s": round(best_bound_err, 2) if best_bound_err < 900 else None,
    }


def evaluate_promos(promos: list[dict], gold_good: list[dict], gold_bad: list[dict]) -> dict[str, Any]:
    """Evaluate a ranked list of promos against gold_good and gold_bad."""
    per_gold = []
    for g in gold_good:
        first_strict_rank = None
        first_Strict_id = None
        first_hl_rank = None
        first_hl_id = None
        best_m = {
            "strict_hit": False, "highlight_hit": False,
            "best_subclip_cov": 0.0, "beat_coverage": 0.0,
            "best_ep_iou": 0.0, "overlap_s": 0.0, "boundary_err_s": None,
        }
        for p in promos:
            m = _overlap_metrics(p["segments"], g)
            if m["highlight_hit"] and first_hl_rank is None:
                first_hl_rank = p["rank"]
                first_hl_id = p["promo_id"]
                best_m = m
            if m["strict_hit"] and first_strict_rank is None:
                first_strict_rank = p["rank"]
                first_Strict_id = p["promo_id"]
                if first_hl_rank == p["rank"]:
                    best_m = m
        per_gold.append({
            "gold_id": g["id"],
            "episodes": g["episodes"],
            "is_cross_ep": g["is_cross_ep"],
            "first_highlight_rank": first_hl_rank,
            "matched_highlight_promo": first_hl_id,
            "first_strict_rank": first_strict_rank,
            "matched_strict_promo": first_Strict_id,
            **best_m,
        })

    def _rec(k: Optional[int], key: str, items: list[dict]) -> float:
        if not items:
            return 0.0
        hits = sum(1 for x in items if x[key] is not None and (k is None or x[key] <= k))
        return round(hits / len(items), 3)

    bound_errs = [x["boundary_err_s"] for x in per_gold if x["boundary_err_s"] is not None]

    # False positive check: do any B01-B04 windows overlap top 10 or top 20 promos?
    fp_top10, fp_top20 = [], []
    for b in gold_bad:
        for p in promos[:20]:
            m = _overlap_metrics(p["segments"], b)
            if m["strict_hit"] or m["highlight_hit"]:
                if p["rank"] <= 10:
                    fp_top10.append({"bad_id": b["id"], "promo_id": p["promo_id"], "rank": p["rank"]})
                fp_top20.append({"bad_id": b["id"], "promo_id": p["promo_id"], "rank": p["rank"]})

    return {
        "gold_total": len(per_gold),
        "highlight_recall_at_10": _rec(10, "first_highlight_rank", per_gold),
        "highlight_recall_at_20": _rec(20, "first_highlight_rank", per_gold),
        "highlight_recall_at_40": _rec(40, "first_highlight_rank", per_gold),
        "highlight_recall_pool": _rec(None, "first_highlight_rank", per_gold),
        "strict_recall_at_10": _rec(10, "first_strict_rank", per_gold),
        "strict_recall_at_20": _rec(20, "first_strict_rank", per_gold),
        "strict_recall_at_40": _rec(40, "first_strict_rank", per_gold),
        "strict_recall_pool": _rec(None, "first_strict_rank", per_gold),
        "mean_boundary_err_s": round(float(np.mean(bound_errs)), 2) if bound_errs else None,
        "median_boundary_err_s": round(float(np.median(bound_errs)), 2) if bound_errs else None,
        "false_positives_top10": len(fp_top10),
        "false_positives_top20": len(fp_top20),
        "fp_details": fp_top20,
        "per_gold": per_gold,
    }
