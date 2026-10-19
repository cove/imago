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

Bad-frame detection uses image measurements and statistical outlier detection. Its purpose is to highlight likely VHS tracking damage for review; normal scene content can also produce high scores.

### 1. Measure four signals per frame

The detector crops the frame edges before measuring the picture (50 pixels on each side by default, clamped for small images). This reduces the influence of border artifacts. It then measures:

| Signal | Measurement | What it looks for |
| --- | --- | --- |
| Color loss (`chroma`) | `1 - mean(HSV saturation) / 255` | A picture losing color and becoming gray during tracking loss. |
| Horizontal noise (`noise`) | Standard deviation of row variances divided by their mean | Rows with unusually strong noise beside relatively normal rows. |
| Row tearing (`tear`) | 95th percentile of the mean absolute brightness difference between adjacent rows | Abrupt row-to-row discontinuities caused by tearing. |
| Horizontal waviness (`wave`) | Standard deviation of each row's brightness center-of-mass after subtracting a five-row moving average | Rapid left/right displacement associated with horizontal sync instability. |

Higher values mean stronger evidence of the corresponding artifact. These are indicators, not proof: grayscale footage, stripes, sharp scene edges, and changes in composition can affect them too.

### 2. Combine the signals into one score

Each signal is normalized across the evaluated frames using a robust z-score:

```text
normalized signal = (value - median) / (1.4826 × median absolute deviation)
composite score = weighted sum of normalized signals / sum of weights
```

If the median absolute deviation is effectively zero, normalization uses the standard deviation, or a scale of 1 if that is also zero. All four signal weights default to `0.25`. Normalizing first makes their different units comparable and reduces the influence of extreme values on the baseline. The resulting score is relative to the analyzed footage, not a probability that a frame is damaged.

### 3. Flag statistical outliers

The default threshold is an upper IQR fence:

```text
Q1 = 25th percentile of scores
Q3 = 75th percentile of scores
IQR = Q3 - Q1
threshold = Q3 + k × IQR
```

The tuner flags scores at or above the threshold. Its default `k` is `3.5`; lowering `IQR k` catches more possible damage, while raising it limits suggestions to more extreme outliers. For example, if Q1 is 0 and Q3 is 2, the threshold at k = 3.5 is 9.

The tuner calculates the threshold from the currently loaded score set. The standalone classifier uses chapter-aligned windows of up to 1,000 evaluated frames by default, excluding larger overlapping umbrella chapters from window selection. It uses a global threshold for frames not covered by those windows. These different comparison sets can produce different classifications even with the same `k`.

The tuner's automatic `k` suggestion searches values from 0.5 to 8.0 for a gap between good and bad scores, favoring gaps that capture a meaningful number of suspect frames. This is a starting point for review, not a learned judgment about the picture.

### 4. Review and save repair decisions

Inspect red-bordered frames and neighboring footage, adjust `IQR k`, then use individual good/bad overrides for false positives or missed damage. `Force All Good` overrides automatic classification for the loaded frames. The tuner saves archive-global bad-frame IDs by chapter in `vhs/metadata/<archive>/render_settings.json` under `bad_frames_by_chapter`.

Saving changes metadata only. During AviSynth rendering, bad-frame ranges become `FreezeFrame` repairs that copy a clean frame into the damaged positions, preferring a following clean frame and falling back to a preceding one. This preserves frame count and timing but can create a visible pause across longer damaged runs. The original archive remains the source; these replacements are applied to rendered outputs.

### Implementation and command reference

- [Tuner signal extraction and review logic](libs/vhs_tuner_core.py): `_compute_signals`, `suggest_iqr_k`, and `_update_bad_frame_set`.
- [Shared score normalization and thresholds](common.py): `robust_zscore`, `combine_signal_scores`, and `compute_threshold`.
- [Standalone classifier](tracking_loss.py): signal extraction and chapter-window thresholding.
- [Render pipeline](vhs_pipeline/render_pipeline.py): conversion of repair ranges into AviSynth `FreezeFrame` calls.

From the repository root, inspect standalone options with:

```sh
uv run python vhs/tracking_loss.py -h
```

## Capture Guide

See [CAPTURE.md](CAPTURE.md) for hardware settings, Osprey configuration, and VirtualDub capture setup.
