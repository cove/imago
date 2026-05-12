# VHS Digital Archive Project

VHS/U-matic ingest and processing pipeline. The primary interface is `vhs.py` (single CLI).

## Quick Start

1. Create/update the project environment: `uv sync`
2. Bootstrap bundled runtime binaries: `uv run python scripts/bootstrap_runtime.py`
3. Run the unified CLI help: `uv run python vhs.py -h`
4. Run tests: `just vhs-test`

## Main Commands

Convert source media into archive MKV:
```
uv run python vhs.py convert avi <file1.avi> <file2.avi> ...
uv run python vhs.py convert umatic <file1.mov> <file2.mov> ...
```

Build metadata + archive checksums:
```
uv run python vhs.py metadata build
```

Generate `chapters.tsv` masters from existing `chapters.ffmetadata` files:
```
uv run python vhs.py metadata migrate-chapters [--overwrite]
```

Lint metadata and simulate TIMEBASE conversion safety:
```
uv run python scripts/lint_metadata.py --simulate-timebase-conversion
```

Embed ffmetadata into existing archive MKVs (no re-encode):
```
uv run python vhs.py metadata embed <archive1.mkv> <archive2.mkv> ...
```

Verify manifests:
```
uv run python vhs.py verify archive [--sha3|--blake3] [manifest_path]
uv run python vhs.py verify drive [--sha3|--blake3] [manifest_path]
```

Generate proxies:
```
uv run python vhs.py proxy
```

Prefill chapter people subtitle ranges from cast face matches:
```
uv run python vhs.py people prefill --archive <archive> --chapter "<chapter title>" --apply
```

Render delivery clips/videos:
```
uv run python vhs.py render [render args]
```

Build original-vs-processed chapter comparison videos:
```
uv run python vhs.py compare [--archive ...] [--title ...] [--height ...]
```

Generate drive-level checksum manifest:
```
uv run python vhs.py checksum drive
```

## Tuner Wizard

Plain HTML web UI for bad frames, gamma, and people/dialogue subtitles.

Start the server:
```
uv run python vhs.py tuner
uv run python vhs.py tuner --host 127.0.0.1 --port 8092   # optional bind
```

Open `http://127.0.0.1:8092`.

### Wizard Flow

1. **Load chapter** — choose archive and chapter, set bad batch proximity
2. **Review frames** — grid of all chapter frames; adjust IQR `k` to change the auto-threshold; click any frame to flip its manual good/bad override
3. **Gamma Correction** — set chapter-wide gamma or visible-range gamma regions
4. **People Subtitles**
   - Optional: click `Auto-fill From Cast` to pull draft ranges from `cast/data`
   - Optional: click `Generate Subtitles` to auto-transcribe chapter dialogue
   - Edit people ranges as `start<TAB>end<TAB>people` (chapter-local times; saved as archive-global in `people.tsv`)
   - Double-click a dialogue subtitle box to edit; click `x` to delete
   - Use the audio waveform strip and playhead to scrub subtitle timing
   - Edit dialogue rows in the table below the timeline; leave the final blank row empty or fill it to add a new row
5. **Summary + Save** — review settings, click `Save and Return to Chapter`; saves:
   - `vhs/metadata/<archive>/render_settings.json` (BAD_FRAMES + gamma)
   - `vhs/metadata/<archive>/people.tsv`
   - `vhs/metadata/<archive>/subtitles.tsv`

### Bad Frame Controls

- Green border = good, red border = bad
- Click a frame card to toggle its manual override
- `Force All Good` forces every loaded frame to good
- `IQR k` slider: lower marks more frames bad, higher marks fewer
- Drag the sparkline to jump through the frame list; use the play button to auto-advance

### Practical Tips

- Tune IQR first, then do manual frame-by-frame overrides.
- During frame loading, use `Cancel` on the loading overlay to stop a long load.
- This is intended for local single-user operation.

## Tracking-Loss Classifier

```
uv run python tracking_loss.py -h
```

## Capture Guide

See [CAPTURE.md](CAPTURE.md) for hardware settings, Osprey configuration, and VirtualDub capture setup.
