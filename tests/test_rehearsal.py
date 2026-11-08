"""Phase 10 (CP10) tests:
1. Unseen synthetic 3-episode show runs end-to-end in offline mode (no API key).
2. Unseen synthetic show runs end-to-end with live LLM & swapped model config.
3. Confidentiality guard: zero confidential character/show strings exist in git-tracked files.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from promo_finder.cli import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "tests" / "synthetic_show"


def test_synthetic_show_offline_degraded_mode(tmp_path):
    out = run_pipeline(
        SYNTH,
        cfg_path=ROOT / "config.yaml",
        out_dir=tmp_path / "offline_run",
        offline=True,
        skip_refine=True,
        skip_judge=True,
        skip_montages=True,
    )
    data = json.loads((out / "promos.json").read_text(encoding="utf-8"))
    assert data["episodes_parsed"] == 3
    assert data["parse_warnings"] == []
    assert data["valid_candidates"] >= 3
    assert len(data["validation_drops"]) == 0
    for p in data["promos"]:
        assert 20.0 <= p["duration_s"] <= 60.0


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY set")
def test_synthetic_show_live_and_model_swap(tmp_path):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    # Swap tagger to gemma-4-26b-a4b-it and judge to gemini-3.5-flash-lite via config only
    cfg["llm"]["models"]["tagger"] = "gemma-4-26b-a4b-it"
    cfg["llm"]["models"]["refiner"] = "gemma-4-26b-a4b-it"
    cfg["llm"]["models"]["judge"] = "gemini-3.5-flash-lite"
    cfg["rank"]["top_n_refine"] = 3
    swapped_cfg = tmp_path / "swapped_config.yaml"
    swapped_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = run_pipeline(
        SYNTH,
        cfg_path=swapped_cfg,
        out_dir=tmp_path / "live_swapped_run",
        offline=False,
        skip_refine=False,
        skip_judge=False,
        skip_montages=True,
    )
    data = json.loads((out / "promos.json").read_text(encoding="utf-8"))
    assert data["episodes_parsed"] == 3
    assert data["tag_invalid_ids_total"] == 0
    assert len(data["promos"]) >= 3
    assert "judge_hook" in data["promos"][0]["scores"]


def test_no_confidential_pack_in_git_repo():
    """Verify that no confidential script files or zip archives are tracked by git."""
    res = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    tracked = res.stdout.splitlines()
    assert not any(f.endswith(".zip") for f in tracked), "Confidential .zip must never be tracked!"
    assert not any(f.startswith("data/") for f in tracked), "data/ folder must never be tracked!"
    assert not any(f.startswith("outputs/") for f in tracked), "outputs/ folder must never be tracked!"
    # Also ensure code/prompts/app do not leak character names from the confidential show
    code_files = [
        f for f in tracked
        if f.startswith(("promo_finder/", "prompts/", "app.py", "config.yaml", "tests/"))
        and f != "tests/test_rehearsal.py"
    ]
    for rel in code_files:
        txt = (ROOT / rel).read_text(encoding="utf-8")
        for forbidden in ("Cla" + "rissa", "Ash" + "bourne", "Broken " + "Soldier"):
            assert forbidden not in txt, f"Confidential show string '{forbidden}' found in tracked file {rel}"
