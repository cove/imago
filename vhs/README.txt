VHS Digital Archive Project
===========================

This repo contains a VHS/U-matic ingest and processing pipeline.
The primary interface is `vhs.py` (single CLI).

Quick Start
-----------

1. Create/update the project environment:
   - `uv sync`
2. Bootstrap bundled runtime binaries used by VHS workflows:
   - `uv run python scripts/bootstrap_runtime.py`
   - Windows venv path: `.venv\\Scripts\\python.exe`
3. Run the unified CLI help:
   - `uv run python vhs.py -h`
4. Run tests:
   - `uv run pytest vhs/test -q`
   - `.\scripts\test.ps1 -q`

Main Commands
-------------

Convert source media into archive MKV:
- `uv run python vhs.py convert avi <file1.avi> <file2.avi> ...`
- `uv run python vhs.py convert umatic <file1.mov> <file2.mov> ...`

Build metadata + archive checksums:
- `uv run python vhs.py metadata build`

Generate `chapters.tsv` masters from existing `chapters.ffmetadata` files:
- `uv run python vhs.py metadata migrate-chapters [--overwrite]`
- `uv run python scripts/migrate_chapters_tsv.py [--overwrite]`

Lint metadata and simulate TIMEBASE conversion safety:
- `uv run python scripts/lint_metadata.py --glob "metadata/bennett*_archive" --simulate-timebase-conversion`

Embed ffmetadata into existing archive MKVs (no re-encode):
- `uv run python vhs.py metadata embed <archive1.mkv> <archive2.mkv> ...`

Verify manifests:
- `uv run python vhs.py verify archive [--sha3|--blake3] [manifest_path]`
- `uv run python vhs.py verify drive [--sha3|--blake3] [manifest_path]`

Generate proxies:
- `uv run python vhs.py proxy`

Launch the plain HTML tuner wizard (bad frames + gamma + people subtitles):
- `uv run python vhs.py tuner`

Prefill chapter people subtitle ranges from cast face matches:
- `uv run python vhs.py people prefill --archive <archive> --chapter "<chapter title>" --apply`

Render delivery clips/videos (forwards args to render pipeline):
- `uv run python vhs.py render [render args]`

Build original-vs-processed chapter comparison videos:
- `uv run python vhs.py compare [--archive ...] [--title ...] [--height ...]`

Generate drive-level checksum manifest:
- `uv run python vhs.py checksum drive`

Interactive Tools
-----------------

Tuner wizard (plain HTML web UI):
- `uv run python vhs.py tuner`
- optional: `uv run python vhs.py tuner --host 127.0.0.1 --port 8092`
- usage guide: `apps/plain_html_wizard/README.md`

Tracking-loss classifier utility:
- `uv run python tracking_loss.py -h`

Directory Notes
---------------

- `metadata/` contains per-archive metadata (`chapters.tsv` master, generated `chapters.ffmetadata`, `render_settings.json`, `people.tsv`, markers, etc.).
- `vhs_pipeline/` contains command implementations used by `vhs.py`.
- `legacy_steps/` contains legacy `step_*` entrypoints and historical step docs.
- `../models/`, `software/`, `manuals/`, `screenshots/` contain model/data/tool references.

Platform Notes
--------------

- Windows supports full AviSynth/QTGMC paths in rendering.
- Linux/macOS use FFmpeg fallback deinterlacing where AviSynth is unavailable.
- Linux FFmpeg archives in `bin/` stay compressed for Git compatibility;
  `scripts/bootstrap_runtime.py` extracts runtime binaries into `bin/`.

Render Settings
---------------

Each archive now uses `metadata/<archive>/render_settings.json` for render controls.

- `archive_settings`: archive-wide defaults (for example `transcript`).
- `chapter_settings`: optional per-chapter overrides.
- `bad_frames_by_chapter`: global BAD frame IDs keyed by exact chapter title.

`render_settings.json` includes a `_comments` object describing each setting category.

People Subtitle TSV
-------------------

`metadata/<archive>/people.tsv` now uses archive-global time ranges:

- Header: `start<TAB>end<TAB>people`
- Times are `HH:MM:SS.mmm` and `end` is exclusive.
- During `uv run python vhs.py render`, ranges are clipped to each chapter span.
- People labels are merged into chapter subtitle sidecars:
  - `.srt`: appended below dialogue in brackets, e.g. `[Jim | Linda]`
  - `.ass`: appended below dialogue in italics
