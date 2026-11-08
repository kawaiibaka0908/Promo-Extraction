# Promo Finder

This tool reads timestamped episode scripts (`.zip` or `.txt` files) and extracts standalone 20 to 60 second video promo clips with a strong hook in the first 3 seconds and a clean cliffhanger cut at the end.

## How It Works

Instead of feeding 50 episodes directly into an LLM (which makes up timestamps and cuts mid-sentence), the pipeline splits the work between deterministic Python code and AI:

1. **Script Parser (`promo_finder/parse.py`):** Reads `CAMERA`, `ACTION`, `DIALOGUE (Tones)`, and `SOUND FX` blocks into structured `Beat` objects and extracts valid cut points (camera shot boundaries and dialogue line endings) so clips never cut mid-sentence.
2. **Dramatic Heat Signals (`promo_finder/signals.py`, `promo_finder/tone.py`):**
   - **Pacing (Code):** Measures shot-cutting speed, speaker switches, exclamation/question density, short punchy lines, sound effects, and `Cut to black` markers.
   - **Tone Intensity (Embeddings):** Embeds dialogue tone words using `all-MiniLM-L6-v2` and scores them from `0.0` (calm/gentle) to `1.0` (intense/enraged) so new tone words work without a hardcoded keyword list.
   - **Beat Tagging (LLM):** Makes one call per episode to tag dramatic beats (`slap`, `betrayal`, `reveal`, `humiliation`, `hook_worthy`, `unresolved`) using only `beat_id`s.
3. **Candidate Clips & Montages (`promo_finder/candidates.py`, `promo_finder/montages.py`):** Finds every valid 20 to 60 second clip where a hook lands in the first 3 seconds, removes overlapping windows, and builds multi-episode montages across shared story arcs.
4. **Refinement, Validation & Cold-Viewer Judge (`promo_finder/refine.py`, `promo_finder/validate.py`, `promo_finder/judge.py`):** Tightens clip boundaries, re-checks hard duration and 3s-hook rules in Python, and uses a separate judge model (`gemma-4-26b-a4b-it`) that reads only the bare clip text to score standalone clarity, hook strength, cliffhanger, and spoiler risk.

## Data Privacy & How Confidential Scripts Stay Safe

To make sure confidential show scripts are never leaked or exposed:
1. **Strict `.gitignore` Rules:** All script inputs (`data/`, `*.zip`), generated outputs (`outputs/`), local caches (`.cache/`), manual evaluation files (`eval/promo_clips_selection.csv`), and `.env` files are excluded from Git tracking.
2. **Automated Git Leak Test (`tests/test_rehearsal.py`):** Every `pytest` run executes `test_no_confidential_pack_in_git_repo`, which inspects `git ls-files` and scans tracked source files to verify that zero confidential `.zip` archives, output folders, or show-specific character names exist in the repository.
3. **In-Memory Temp Processing in Streamlit (`app.py`):** When you upload a `.zip` or `.txt` file in the web app, it is processed inside an isolated `tempfile.TemporaryDirectory()` that is automatically deleted as soon as the run finishes. API keys entered in the sidebar are held only in session memory for that run and are never logged or saved to disk.
4. **Synthetic Demo Included:** For public testing without confidential files, the repo includes a 3-episode synthetic sample show under `tests/synthetic_show/`.

## Setup & Running Locally

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Set Your API Key (Optional)
```powershell
# PowerShell
$env:GEMINI_API_KEY = "your_api_key_here"
```
```bash
# Linux / macOS
export GEMINI_API_KEY="your_api_key_here"
```
You can also switch providers (`gemini`, `groq`, `openrouter`, `openai`) in `config.yaml` or directly in the Streamlit sidebar, or run in offline mode without an API key.

### 3. Run the Web App
```bash
streamlit run app.py
```

### 4. Run from the Command Line
```bash
# Run on a script .zip file or folder
python -m promo_finder data/show.zip --out outputs/my_run

# Run on the built-in 3-episode sample show
python -m promo_finder tests/synthetic_show --out outputs/sample_run

# Run in offline mode (no API key required)
python -m promo_finder tests/synthetic_show --offline --skip-refine --skip-judge --skip-montages
```

### 5. Run Tests
```bash
pytest -v tests
```

## Project Layout

```text
promo_finder/
  parse.py        # Script parser and clean cut-point extractor
  signals.py      # Pacing features, LLM beat tagger, and heat curve
  tone.py         # MiniLM embedding scorer for dialogue tone words
  candidates.py   # 20-60s candidate window generator and overlap filter
  montages.py     # Multi-episode story-arc montage builder
  refine.py       # LLM copy writer and boundary refiner
  validate.py     # Hard rule validators (duration, 3s hook, cut points)
  judge.py        # Cold-viewer judge model (blind summary + 1-5 ratings)
  rank.py         # Final scoring and episode diversity cap
  plots.py        # Episode heat curve PNG plotter
  evaluate.py     # Evaluation and ablation metrics script
  llm.py          # OpenAI-compatible client with caching and model fallback
  cli.py          # CLI entry point
prompts/          # Prompt templates for tagging, refining, judging, and arcs
tests/            # Unit tests, git leak check, and 3-episode sample show
app.py            # Streamlit web interface
config.yaml       # Model, provider, and scoring settings
```
