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
- `uv run python scripts/lint_metadata.py --simulate-timebase-conversion`

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

- `vhs/metadata/` contains per-archive metadata (`chapters.tsv` master, generated `chapters.ffmetadata`, `render_settings.json`, `people.tsv`, markers, etc.).
- `vhs_pipeline/` contains command implementations used by `vhs.py`.
- `../models/`, `software/`, `manuals/`, `screenshots/` contain model/data/tool references.

Platform Notes
--------------

- Windows supports full AviSynth/QTGMC paths in rendering.
- Linux/macOS use FFmpeg fallback deinterlacing where AviSynth is unavailable.
- Linux FFmpeg archives in `bin/` stay compressed for Git compatibility;
  `scripts/bootstrap_runtime.py` extracts runtime binaries into `bin/`.

Render Settings
---------------

Each archive now uses `vhs/metadata/<archive>/render_settings.json` for render controls.

- `archive_settings`: archive-wide defaults (for example `transcript`).
- `chapter_settings`: optional per-chapter overrides.
- `bad_frames_by_chapter`: global BAD frame IDs keyed by exact chapter title.

`render_settings.json` includes a `_comments` object describing each setting category.

People Subtitle TSV
-------------------

`vhs/metadata/<archive>/people.tsv` now uses archive-global time ranges:

- Header: `start<TAB>end<TAB>people`
- Times are `HH:MM:SS.mmm` and `end` is exclusive.
- During `uv run python vhs.py render`, ranges are clipped to each chapter span.
- People labels are merged into chapter subtitle sidecars:
  - `.srt`: appended below dialogue in brackets, e.g. `[Jim | Linda]`
  - `.ass`: appended below dialogue in italics

Historical Capture Guide
------------------------

The following is the original VHS capture hardware/software reference,
preserved from the legacy step_1 documentation.

Panasonic AG-1970P VCR settings:
- Phone Level = Neutral, Picture = Neutral, Hi-fi Rec Level = Neutral
- Noise Filter = Off, TBC = On, Search Sound = Off, HiFi/NormalMix = Off, Tape Select = T120, Mono = On, MTS = MTS
- A1 connectors (back of VCR) S-Video out to Osprey 260e S-Video In, A1 connectors Right and Left Audio to Osprey 260e Unbalanced Audio In.

Osprey 260e Analog Capture Card Settings
Everything set to defaults, except:
- RefSize -> Horizontal Format: CCIR-601, Source Width: 720
- Input -> Video Input: S-Video
- Video Decoder -> Video Standard: NTSC_M
- Filters -> SimulStream: Unchecked

Software Requirements
In the software/ directory:
- Install the UT Video driver
- Install the Osprey driver
- Unzip VirtualDub

VirtualDub Capture Software Settings
1. Set capture framerate to 29.97 FPS in the lower right of the screen.
2. Set Audio -> Compression to PCM 48.000 kHz 16-bit mono if it's a home video and not stereo. (Mono avoids blank audio track noise from being mixed in.)
3. Set Video -> Compression "UtVideo YUV422 BT.601.VCM"
4. Set Capture -> Settings... -> Abort options -> Abort on left mouse button to be unchecked since it's causes errant recording stops, instead press the ESC key to stop recording.
5. Set Capture -> Timing... -> Internal capture mode synchronization -> no correction
6. Set Capture -> Stop conditions ... -> Capture time exceeds to 5400 seconds for the maximum of a VHS-C tape on EP mode.
7. Set File -> Set Capture File... to the output file every time, as that it automatically overwrites the last file by default.

See the screenshots/ directory for what the above settings look like.

Notes:

VirtualDub included in the software/ and encoding with UT Video is the only way I was able to get a reliable correct capture. Specifically it did not write video faster to storage than it could handle, and UT Video kept the color in VHS ranges. Addtionaly it didn't crash.

T2 UtVideo YUV422 BT.601.VCM is faster and also works, but VLC doesn't support the T2 version to validate the capture.

If the VCR automatically stops playing the VHS-C tape in its VHS cassette adapter, this can be due to the battery being low.

The VirtualDub software included is from here:
https://www.digitalfaq.com/guides/video/capture-avi-virtualdub.htm
https://www.digitalfaq.com/forum/video-conversion/1727-virtualdub-filters-pre.html

The FFmpeg-QTGMC Easy 2025.01.11 is from here:
https://forum.videohelp.com/threads/405720-FFmpeg-QTGMC-Easy!#post2656404
