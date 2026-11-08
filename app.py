"""Streamlit app for Promo Finder."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from promo_finder.cli import run_pipeline
from promo_finder.parse import load_show

st.set_page_config(
    page_title="Promo Finder",
    page_icon="🎬",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config.yaml"
SYNTH_DIR = ROOT / "tests" / "synthetic_show"
LOCAL_FULL_RUN = ROOT / "outputs" / "cp8_full" / "promos.json"


def _load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


st.title("Promo Finder")
st.caption("Find and rank standalone 20–60s promo clips from episode scripts.")

# --------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Settings")
    cfg = _load_cfg()

    provider = st.selectbox(
        "Model Provider",
        options=["gemini", "groq", "openrouter", "openai", "offline (no API key)"],
        index=0,
    )
    is_offline = provider.startswith("offline")

    default_models = {
        "gemini": ("gemini-3.5-flash-lite", "gemma-4-26b-a4b-it"),
        "groq": ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        "openrouter": ("google/gemini-2.5-flash", "meta-llama/llama-3.1-8b-instruct"),
        "openai": ("gpt-4o-mini", "gpt-4o-mini"),
        "offline (no API key)": ("none", "none"),
    }
    def_tag, def_judge = default_models.get(provider, ("gemini-3.5-flash-lite", "gemma-4-26b-a4b-it"))

    tag_model = st.text_input("Tagger / Refiner Model", value=def_tag, disabled=is_offline)
    judge_model = st.text_input("Judge Model", value=def_judge, disabled=is_offline)

    api_key_input = st.text_input(
        "API Key (optional if already set)",
        type="password",
        disabled=is_offline,
    )

    st.divider()
    st.subheader("Options")
    enable_tone = st.checkbox("Tone Embeddings", value=True)
    enable_refine = st.checkbox("Refine Copy & Timestamps", value=not is_offline, disabled=is_offline)
    enable_judge = st.checkbox("Cold-Viewer Judge", value=not is_offline, disabled=is_offline)
    enable_montages = st.checkbox("Multi-Episode Montages", value=not is_offline, disabled=is_offline)
    max_eps = st.number_input("Episode Limit (0 = all)", min_value=0, max_value=500, value=0, step=5)


def _render_results(data: dict, csv_text: str, episodes_dump: dict | None = None, key_prefix: str = "run"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes Processed", data["episodes_parsed"])
    c2.metric("Promos Found", data["valid_candidates"])
    c3.metric("Time Taken", f"{data.get('runtime_s', 0)}s")

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download JSON",
        data=json.dumps(data, indent=2, ensure_ascii=False),
        file_name="promos.json",
        mime="application/json",
        use_container_width=True,
        key=f"{key_prefix}_dl_json",
    )
    dl2.download_button(
        "Download CSV",
        data=csv_text,
        file_name="promos.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_dl_csv",
    )

    promos = data["promos"]
    rows = []
    for p in promos:
        segs = " + ".join(f"E{s['episode']:02d} {s['start_ts']}–{s['end_ts']}" for s in p["segments"])
        sc = p["scores"]
        rows.append({
            "Rank": p["rank"],
            "Type": p["kind"],
            "Timestamps": segs,
            "Duration (s)": p["duration_s"],
            "Score": sc.get("final", sc.get("rule_score", 0)),
            "Hook (First 3s)": p["hook"],
            "Cliffhanger": p["cliffhanger"],
            "Why It Works": p["why"],
        })
    df = pd.DataFrame(rows)
    st.subheader("Top Promo Clips")
    st.markdown(df.head(25).to_html(index=False), unsafe_allow_html=True)

    st.subheader("Clip Details")
    for p in promos[:15]:
        segs = " + ".join(f"Ep {s['episode']:02d} ({s['start_ts']}–{s['end_ts']})" for s in p["segments"])
        sc = p["scores"]
        label = "Montage" if p["kind"] == "montage" else "Single Clip"
        with st.expander(f"#{p['rank']} | {segs} ({p['duration_s']}s) — {label} | Score: {sc.get('final', 0):.3f}"):
            st.markdown(f"- **Hook:** {p['hook']}")
            st.markdown(f"- **Cliffhanger:** {p['cliffhanger']}")
            st.markdown(f"- **Why It Works:** {p['why']}")
            if "judge_retell" in sc:
                st.info(
                    f"**Viewer Summary:** {sc['judge_retell']}\n\n"
                    f"**Hook Question:** *{sc.get('judge_question', '')}*  |  "
                    f"**Ratings:** Hook `{sc.get('judge_hook')}/5` · Cliffhanger `{sc.get('judge_cliff')}/5` · "
                    f"Clarity `{sc.get('judge_standalone')}/5` · Spoiler `{sc.get('judge_spoiler')}/5`"
                )
            ep_first = p["segments"][0]["episode"]
            heat_png = ROOT / "outputs" / "cp8_full" / "heat_curves" / f"E{ep_first:02d}.png"
            if heat_png.exists() and episodes_dump is None:
                st.image(str(heat_png), caption=f"Episode {ep_first:02d} Heat Curve")


# --------------------------------------------------------------------------- Main UI
if LOCAL_FULL_RUN.exists():
    tab_upload, tab_local = st.tabs(["Find Promos", "Saved Results"])
else:
    tab_upload = st.container()
    tab_local = None

with tab_upload:
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded = st.file_uploader(
            "Upload script `.zip` or `.txt` files",
            type=["zip", "txt"],
            accept_multiple_files=True,
        )
    with col_u2:
        st.write("")
        st.write("")
        run_btn = st.button("Find Promos", type="primary", use_container_width=True, disabled=not uploaded)
        use_synth = st.button("Try Sample Demo", use_container_width=True)

    if use_synth or run_btn:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            if use_synth:
                in_path = SYNTH_DIR
            else:
                in_dir = tmp / "input"
                in_dir.mkdir(parents=True, exist_ok=True)
                if len(uploaded) == 1 and uploaded[0].name.lower().endswith(".zip"):
                    in_path = tmp / uploaded[0].name
                    in_path.write_bytes(uploaded[0].getvalue())
                else:
                    for uf in uploaded:
                        (in_dir / uf.name).write_bytes(uf.getvalue())
                    in_path = in_dir

            run_cfg = _load_cfg()
            if not is_offline:
                run_cfg["llm"]["provider"] = provider
                run_cfg["llm"]["models"]["tagger"] = tag_model
                run_cfg["llm"]["models"]["refiner"] = tag_model
                run_cfg["llm"]["models"]["arcs"] = tag_model
                run_cfg["llm"]["models"]["judge"] = judge_model
                env_name = run_cfg["llm"]["providers"][provider]["api_key_env"]
                if api_key_input.strip():
                    os.environ[env_name] = api_key_input.strip()
                elif not os.environ.get(env_name):
                    try:
                        if env_name in st.secrets:
                            os.environ[env_name] = st.secrets[env_name]
                    except Exception:
                        pass

            tmp_cfg = tmp / "config.yaml"
            tmp_cfg.write_text(yaml.safe_dump(run_cfg), encoding="utf-8")
            out_dir = tmp / "out"

            with st.spinner("Processing scripts..."):
                res_dir = run_pipeline(
                    in_path,
                    cfg_path=tmp_cfg,
                    out_dir=out_dir,
                    offline=is_offline,
                    skip_refine=not enable_refine,
                    skip_judge=not enable_judge,
                    skip_tone=not enable_tone,
                    skip_montages=not enable_montages,
                    max_episodes=max_eps if max_eps > 0 else None,
                )
                st.session_state["last_run_json"] = (res_dir / "promos.json").read_text(encoding="utf-8")
                st.session_state["last_run_csv"] = (res_dir / "promos.csv").read_text(encoding="utf-8")
                eps_loaded, _ = load_show(in_path)
                if max_eps > 0:
                    eps_loaded = eps_loaded[:max_eps]
                st.session_state["last_run_episodes"] = {e.number: e.model_dump() for e in eps_loaded}

    if "last_run_json" in st.session_state:
        _render_results(
            json.loads(st.session_state["last_run_json"]),
            st.session_state["last_run_csv"],
            st.session_state.get("last_run_episodes"),
            key_prefix="live",
        )

if tab_local is not None:
    with tab_local:
        d_local = json.loads(LOCAL_FULL_RUN.read_text(encoding="utf-8"))
        csv_local = (LOCAL_FULL_RUN.parent / "promos.csv").read_text(encoding="utf-8")
        _render_results(d_local, csv_local, key_prefix="saved")
