VHS Digital Archive Project
===========================

This repo contains a VHS/U-matic ingest and processing pipeline.
The primary interface is `vhs.py` (single CLI).

Quick Start
-----------

1. Create/install the project environment:
   - `python setup.py`
   - Windows venv path: `.venv\\Scripts\\python.exe`
2. Run the unified CLI help:
   - `python vhs.py -h`
3. Run tests via helper script (auto-picks `.venv` on Windows when present):
   - `.\scripts\test.ps1 -q`

Main Commands
-------------

Convert source media into archive MKV:
- `python vhs.py convert avi <file1.avi> <file2.avi> ...`
- `python vhs.py convert umatic <file1.mov> <file2.mov> ...`

Build metadata + archive checksums:
- `python vhs.py metadata build`

Generate `chapters.tsv` masters from existing `chapters.ffmetadata` files:
- `python vhs.py metadata migrate-chapters [--overwrite]`
- `python scripts/migrate_chapters_tsv.py [--overwrite]`

Lint metadata and simulate TIMEBASE conversion safety:
- `python scripts/lint_metadata.py --glob "metadata/bennett*_archive" --simulate-timebase-conversion`

Embed ffmetadata into existing archive MKVs (no re-encode):
- `python vhs.py metadata embed <archive1.mkv> <archive2.mkv> ...`

Verify manifests:
- `python vhs.py verify archive [--sha3|--blake3] [manifest_path]`
- `python vhs.py verify drive [--sha3|--blake3] [manifest_path]`

Generate proxies:
- `python vhs.py proxy`

Launch the plain HTML tuner wizard (bad frames + gamma + people subtitles):
- `python vhs.py tuner`

Prefill chapter people subtitle ranges from cast face matches:
- `python vhs.py people prefill --archive <archive> --chapter "<chapter title>" --apply`

Render delivery clips/videos (forwards args to render pipeline):
- `python vhs.py render [render args]`

Build original-vs-processed chapter comparison videos:
- `python vhs.py compare [--archive ...] [--title ...] [--height ...]`

Generate drive-level checksum manifest:
- `python vhs.py checksum drive`

Interactive Tools
-----------------

Tuner wizard (plain HTML web UI):
- `python vhs.py tuner`
- optional: `python vhs.py tuner --host 127.0.0.1 --port 8092`
- usage guide: `apps/plain_html_wizard/README.md`

Tracking-loss classifier utility:
- `python tracking_loss.py -h`

Directory Notes
---------------

- `metadata/` contains per-archive metadata (`chapters.tsv` master, generated `chapters.ffmetadata`, `render_settings.json`, `people.tsv`, markers, etc.).
- `vhs_pipeline/` contains command implementations used by `vhs.py`.
- `legacy_steps/` contains legacy `step_*` entrypoints and historical step docs.
- `../modes/`, `software/`, `manuals/`, `screenshots/` contain model/data/tool references.

Platform Notes
--------------

- Windows supports full AviSynth/QTGMC paths in rendering.
- Linux/macOS use FFmpeg fallback deinterlacing where AviSynth is unavailable.
- Linux FFmpeg archives in `bin/` stay compressed for Git compatibility; `setup.py`
  extracts runtime binaries into `bin/`.

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
- During `python vhs.py render`, ranges are clipped to each chapter span.
- People labels are merged into chapter subtitle sidecars:
  - `.srt`: appended below dialogue in brackets, e.g. `[Jim | Linda]`
  - `.ass`: appended below dialogue in italics
