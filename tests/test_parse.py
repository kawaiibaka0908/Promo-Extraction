"""CP2 tests for the parser. Synthetic snippets only - no confidential script text."""
from pathlib import Path

import pytest

from promo_finder.parse import decode_bytes, load_show, parse_episode, to_seconds

SAMPLE = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Test notes.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. HALL \u2013 NIGHT
Scene start: 00:00.00
Scene end: 00:10.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of a hall.

**ANNA (Furious, Shouting):**
Start: (00:01.50) | End: (00:03.00)
\u201cGet out!\u201d

**ACTION:**
Start: (00:04.00) | End: (00:06.00)
She slaps him.

**BROKEN BLOCK:**
no timestamps here

**SOUND FX:**
Start: (00:08.00) | End: (00:10.00)
A sting swells.

**CAMERA:**
Start: (00:09.00) | End: (00:10.00)
Cut to black.

--- SCRIPT END ---
"""


def test_to_seconds():
    assert to_seconds("01:23.50") == 83.5
    assert to_seconds("(00:04.80)") == 4.8
    assert to_seconds("1:00:00.0") == 3600.0


def test_decode_normalises_dash_and_quotes():
    t = decode_bytes("INT. HALL \u2013 DAY \u201cHi\u201d".encode("utf-8"))
    assert t == 'INT. HALL - DAY "Hi"'
    assert "caf" in decode_bytes("caf\xe9".encode("cp1252"))  # cp1252 fallback, no crash


def test_parse_sample():
    ep, warnings = parse_episode(decode_bytes(SAMPLE.encode()), source="t.txt")
    assert ep.number == 7 and ep.logline == "A test logline."
    assert len(ep.scenes) == 1 and ep.scenes[0].heading == "INT. HALL - NIGHT"
    types = [b.type for b in ep.beats]
    assert types == ["camera", "dialogue", "action", "sfx", "camera"]
    d = ep.beats[1]
    assert d.speaker == "ANNA" and d.tones == ["furious", "shouting"] and d.text == "Get out!"
    assert ep.beats[-1].text == "Cut to black." and ep.duration == 10.0
    assert any("no timestamps" in w for w in warnings)  # malformed block skipped, not fatal
    assert ep.beats[0].beat_id == "E07_B001"
    cps = ep.cut_points()
    assert 0.0 in cps and 3.0 in cps and 10.0 in cps


DATA = Path(__file__).resolve().parents[1] / "data" / "show.zip"


@pytest.mark.skipif(not DATA.exists(), reason="confidential input pack not present")
def test_full_pack_counts():
    eps, warnings = load_show(DATA)
    assert len(eps) == 50
    counts = {}
    for e in eps:
        for b in e.beats:
            counts[b.type] = counts.get(b.type, 0) + 1
    assert counts == {"camera": 1754, "dialogue": 1487, "action": 372, "sfx": 114}
    assert warnings == []
    e1 = eps[0]
    assert e1.number == 1 and e1.duration == 138.0
    assert e1.beats[0].start == 0.0 and e1.beats[-1].text.lower().startswith("cut to black")
    for e in eps:
        for b in e.beats:
            sc = next(s for s in e.scenes if s.number == b.scene)
            assert sc.start - 0.5 <= b.start <= b.end <= sc.end + 0.5
