"""Heat-curve plot per episode with hook anchors and chosen promo windows highlighted."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .candidates import Candidate
from .parse import Episode
from .signals import EpisodeSignals


def plot_episode_heat(
    ep: Episode,
    sig: EpisodeSignals,
    promos: list[Candidate],
    out_path: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=130)
    ax.plot(sig.grid_t, sig.grid_heat, color="#d9480f", lw=2.0, label="Combined Heat (LLM + Tone + Structure)")
    ax.fill_between(sig.grid_t, 0, sig.grid_heat, color="#fd7e14", alpha=0.18)

    # Mark hook-worthy beats
    bmap = {b.beat_id: b for b in ep.beats}
    hook_times = [bmap[bid].start for bid, t in sig.tags.items() if t.hook_worthy and bid in bmap]
    hook_heats = [sig.heat[bid] for bid, t in sig.tags.items() if t.hook_worthy and bid in bmap]
    if hook_times:
        ax.scatter(hook_times, hook_heats, color="#1864ab", zorder=5, s=28, label="LLM Hook-Worthy Beats")

    # Shade selected promo windows in this episode
    colors = ["#2b8a3e", "#5f3dc4", "#0b7285"]
    idx = 0
    for p in promos:
        for s in p.segments:
            if s.episode == ep.number:
                c = colors[idx % len(colors)]
                ax.axvspan(s.start, s.end, color=c, alpha=0.22, label=f"Promo {p.promo_id} ({s.duration:.0f}s)")
                idx += 1

    ax.set_xlim(0, max(ep.duration, 1.0))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Episode Time (seconds)")
    ax.set_ylabel("Dramatic Intensity (0-1)")
    ax.set_title(f"Episode {ep.number:02d} Heat Curve & Promo Windows", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
