# AGENTS.md

Purpose: operating rules for the `vhs/` project.

## Data and Schema Migrations

Treat `vhs/metadata/*/render_settings.json` and related metadata as migratable assets.

## Directory Layout

- `vhs/metadata/` — per-archive metadata (`chapters.tsv`, `chapters.ffmetadata`, `render_settings.json`, `people.tsv`, markers, etc.)
- `vhs_pipeline/` — command implementations used by `vhs.py`
- `../models/`, `software/`, `manuals/`, `screenshots/` — model/data/tool references

## Platform Behavior

- Windows supports full AviSynth/QTGMC paths in rendering.
- Linux/macOS use FFmpeg fallback deinterlacing where AviSynth is unavailable.
- Linux FFmpeg archives in `bin/` stay compressed for Git compatibility; `scripts/bootstrap_runtime.py` extracts runtime binaries into `bin/`.

## Render Settings Schema

Each archive uses `vhs/metadata/<archive>/render_settings.json`.

- `archive_settings` — archive-wide defaults (e.g. `transcript`)
- `chapter_settings` — optional per-chapter overrides
- `bad_frames_by_chapter` — global BAD frame IDs keyed by exact chapter title

`render_settings.json` includes a `_comments` object describing each setting category.

To have a larger chapter inherit bad frames marked in overlapping subchapters, set `archive_settings.inherit_bad_frames_from_overlaps = true`.

## BAD_FRAMES → AviSynth FreezeFrame

The tuner writes metadata only; it does not modify video pixels. During render, the AviSynth script reads `BAD_FRAMES` from `render_settings.json` and converts them into `FreezeFrame(start, end, source)` calls. Source is selected automatically (next clean frame after a bad range, or previous clean frame as fallback). Contiguous/near-contiguous bad frames are merged for stable replacements.

## People Subtitle TSV Format

`vhs/metadata/<archive>/people.tsv` uses archive-global time ranges:

- Header: `start<TAB>end<TAB>people`
- Times are `HH:MM:SS.mmm`; `end` is exclusive
- During render, ranges are clipped to each chapter span
- People labels are merged into chapter subtitle sidecars:
  - `.srt`: appended below dialogue in brackets, e.g. `[Jim | Linda]`
  - `.ass`: appended below dialogue in italics
