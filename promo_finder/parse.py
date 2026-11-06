"""Step 1 - Parse microdrama screenplays into typed, timestamped beats.

Pure Python, no AI. The output of this module is the single source of truth
for every timestamp the pipeline emits.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

log = logging.getLogger(__name__)

BeatType = Literal["camera", "action", "dialogue", "sfx"]


class Beat(BaseModel):
    beat_id: str
    episode: int
    scene: int
    type: BeatType
    start: float
    end: float
    speaker: Optional[str] = None
    tones: list[str] = []
    text: str

    @model_validator(mode="after")
    def _check_times(self):
        if self.end < self.start:
            raise ValueError(f"{self.beat_id}: end {self.end} < start {self.start}")
        return self


class Scene(BaseModel):
    number: int
    heading: str = ""
    start: float = 0.0
    end: float = 0.0


class Episode(BaseModel):
    number: int
    logline: str = ""
    continuity: str = ""
    source: str = ""
    scenes: list[Scene] = []
    beats: list[Beat] = []

    @property
    def duration(self) -> float:
        ends = [b.end for b in self.beats] + [s.end for s in self.scenes]
        return max(ends) if ends else 0.0

    def cut_points(self) -> list[float]:
        """Times where an editor could cleanly cut: shot boundaries and dialogue-line ends."""
        pts = {0.0, round(self.duration, 2)}
        for b in self.beats:
            if b.type == "camera":
                pts.add(round(b.start, 2))
                pts.add(round(b.end, 2))
            elif b.type == "dialogue":
                pts.add(round(b.end, 2))
        return sorted(pts)


# --------------------------------------------------------------------------- text helpers

_REPLACE = {
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00a0": " ",
}


def decode_bytes(raw: bytes) -> str:
    """UTF-8 first (strip BOM), fall back to cp1252, never crash."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFKC", text)
    for k, v in _REPLACE.items():
        text = text.replace(k, v)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def to_seconds(ts: str) -> float:
    """'01:23.50' -> 83.5 ; also accepts 'h:mm:ss.ff' and plain seconds."""
    ts = ts.strip().strip("()")
    parts = ts.split(":")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return round(secs, 3)


def fmt_ts(secs: float) -> str:
    m, s = divmod(max(secs, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


# --------------------------------------------------------------------------- regexes

RE_EP_NUM = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.*)", re.I)
RE_CONT = re.compile(r"^CONTINUITY NOTES:\s*(.*)", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)", re.I)
RE_SCENE_HEAD = re.compile(r"^Scene heading:\s*(.*)", re.I)
RE_SCENE_T = re.compile(r"^Scene (start|end):\s*([\d:.]+)", re.I)
RE_LABEL = re.compile(r"^\*\*(.+?):?\*\*:?\s*$")
RE_TIME = re.compile(r"Start:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*End:\s*\(?\s*([\d:.]+)\s*\)?", re.I)
RE_EP_FROM_NAME = re.compile(r"episode-(\d+)", re.I)

_TYPE_MAP = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx", "SOUND": "sfx"}


def _classify_label(label: str) -> tuple[str, Optional[str], list[str]]:
    label = label.strip().rstrip(":").strip()
    if label.upper() in _TYPE_MAP:
        return _TYPE_MAP[label.upper()], None, []
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
    if m:
        speaker, tone_str = m.group(1).strip(), m.group(2)
        tones = [t.strip().lower() for t in tone_str.split(",") if t.strip()]
    else:
        speaker, tones = label, []
    return "dialogue", speaker.upper(), tones


# --------------------------------------------------------------------------- parser

def parse_episode(text: str, source: str = "", fallback_number: int = 0) -> tuple[Episode, list[str]]:
    """Parse one episode. Returns (episode, warnings). Malformed blocks are skipped, not fatal."""
    warnings: list[str] = []
    ep = Episode(number=fallback_number, source=source)
    scene = Scene(number=0)
    block: Optional[dict] = None
    beat_idx = 0
    in_script = False

    def flush():
        nonlocal block, beat_idx
        if block is None:
            return
        if block.get("start") is None:
            warnings.append(f"{source}: block '{block['label']}' has no timestamps - skipped")
        else:
            txt = " ".join(block["lines"]).strip()
            if block["type"] == "dialogue":
                txt = txt.strip('"').strip()
            beat_idx += 1
            try:
                ep.beats.append(Beat(
                    beat_id=f"E{ep.number:02d}_B{beat_idx:03d}", episode=ep.number, scene=scene.number,
                    type=block["type"], start=block["start"], end=block["end"],
                    speaker=block["speaker"], tones=block["tones"], text=txt,
                ))
            except ValueError as e:
                warnings.append(f"{source}: {e} - skipped")
        block = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not in_script:
            if m := RE_EP_NUM.match(line):
                ep.number = int(m.group(1))
            elif m := RE_LOGLINE.match(line):
                ep.logline = m.group(1).strip()
            elif m := RE_CONT.match(line):
                ep.continuity = m.group(1).strip()
            elif "SCRIPT START" in line.upper():
                in_script = True
                continue
            elif RE_SCENE.match(line):  # tolerate a missing SCRIPT START marker
                in_script = True
            else:
                continue
            if not in_script:
                continue
        if "SCRIPT END" in line.upper():
            flush()
            break
        if not line:
            # blank line ends a block only if it already has text
            if block is not None and block["lines"]:
                flush()
            continue
        if m := RE_SCENE.match(line):
            flush()
            scene = Scene(number=int(m.group(1)))
            ep.scenes.append(scene)
            continue
        if m := RE_SCENE_HEAD.match(line):
            scene.heading = m.group(1).strip()
            continue
        if m := RE_SCENE_T.match(line):
            setattr(scene, m.group(1).lower(), to_seconds(m.group(2)))
            continue
        if line.lower().startswith("actions and dialogue"):
            continue
        if m := RE_LABEL.match(line):
            flush()
            btype, speaker, tones = _classify_label(m.group(1))
            block = {"label": m.group(1), "type": btype, "speaker": speaker, "tones": tones,
                     "start": None, "end": None, "lines": []}
            continue
        if block is not None and block["start"] is None and (m := RE_TIME.search(line)):
            block["start"], block["end"] = to_seconds(m.group(1)), to_seconds(m.group(2))
            continue
        if block is not None:
            block["lines"].append(line)
        else:
            warnings.append(f"{source}: stray line ignored: {line[:60]}")
    flush()

    # re-stamp ids if the episode number was only found late / from filename
    for i, b in enumerate(ep.beats, 1):
        b.episode = ep.number
        b.beat_id = f"E{ep.number:02d}_B{i:03d}"
    ep.beats.sort(key=lambda b: (b.start, b.end))
    return ep, warnings


def load_show(path: str | Path) -> tuple[list[Episode], list[str]]:
    """Load a show from a .zip or a folder of episode .txt files."""
    path = Path(path)
    items: list[tuple[str, bytes]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith(".txt") and not name.endswith("/"):
                    items.append((name, z.read(name)))
    elif path.is_dir():
        for p in sorted(path.rglob("*.txt")):
            items.append((p.name, p.read_bytes()))
    elif path.is_file():
        items.append((path.name, path.read_bytes()))
    else:
        raise FileNotFoundError(path)

    episodes, warnings = [], []
    for i, (name, raw) in enumerate(sorted(items), 1):
        m = RE_EP_FROM_NAME.search(name)
        ep, w = parse_episode(decode_bytes(raw), source=Path(name).name, fallback_number=int(m.group(1)) if m else i)
        if not ep.beats:
            warnings.append(f"{name}: no beats parsed - skipped")
            continue
        episodes.append(ep)
        warnings.extend(w)
    episodes.sort(key=lambda e: e.number)
    return episodes, warnings
