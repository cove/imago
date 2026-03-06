VHS Plain HTML Wizard
=====================

This app is a plain HTML/CSS/JS wizard with a small Python HTTP server.
It avoids UI framework lock-in and reuses `libs/vhs_tuner_core.py` for core logic.

Run (Windows venv example)
--------------------------

1. Activate your environment.
2. Start the server:
   `.venv\\Scripts\\python apps\\plain_html_wizard\\server.py`
3. Open:
   `http://127.0.0.1:8092`

Wizard Flow
-----------

1. Load chapter
   - choose archive and chapter
   - set bad batch proximity
2. Review frames
   - all chapter frames are shown in a grid
   - adjust IQR `k` to change the automatic threshold
   - click any frame to flip its manual good/bad override
3. Gamma Correction
   - set chapter-wide gamma or visible-range gamma regions
4. People Subtitles
   - optional first pass: auto-fill ranges from `cast/data` matches
   - add people subtitle ranges from the currently visible frame window
   - optional first pass: click `Generate Subtitles` to auto-transcribe chapter dialogue
   - dialogue subtitle text appears as boxes on the timeline, plus an optional bulk-edit table below it
5. Summary and save
   - review settings/stats
   - save BAD frames + gamma to `render_settings.json`
   - save people subtitle rows to `people.tsv` (archive-global time ranges)
   - save dialogue subtitle rows to `subtitles.tsv` (archive-global time ranges)

How To Use The Tuner
--------------------

Step 1: Load chapter

1. Select an archive on the left.
2. Select a chapter in the chapter list.
3. Chapter frame span is loaded directly from chapter metadata.
4. Click the top `2/5 Bad Frames + IQR` step button to load frames.

Step 2: Review frames + threshold

- Frame colors:
  - green border = currently good
  - red border = currently bad
- Click a frame card to toggle its manual override.
- Use `Force All Good` to force every loaded frame status to good.
- Use the `IQR k` slider to adjust auto-threshold:
  - lower `k` marks more frames as bad
  - higher `k` marks fewer frames as bad
- Use the sparkline to inspect scores:
  - drag on the sparkline to jump through the frame list
  - use the play button to auto-advance the current viewport
- Use `Fullscreen` when you need a larger review area.

Step 3: Gamma Correction

- Use chapter-wide gamma for one value across the chapter, or region mode for visible-range bands.
- Save progress at this stage if you want to resume later.

Step 4: People Subtitles

- Scroll the frame grid to the time window where people appear.
- Optional first pass: click `Auto-fill From Cast` to pull draft ranges from `cast/data`.
- Optional dialogue draft: click `Generate Subtitles` to transcribe chapter speech.
- Double-click a dialogue subtitle box to edit its text; click `x` on the box to delete it.
- Use the audio waveform strip and its vertical playhead to scrub subtitle timing by ear.
- Click the play button beside the waveform to audition audio from the current playhead position.
- Use the `+` and `-` controls on the right side of the timeline to zoom; click the zoom value to reset.
- Edit dialogue rows in the table below the timeline (one field per cell).
- Click a row cell to edit start/end/text/speaker/confidence/source values.
- The table auto-scrolls/highlights around the current timeline cursor as you scrub or scroll frames.
- Leave the final blank row empty, or fill it to add a new subtitle row.
- Edit rows directly in the people editor:
  - `start<TAB>end<TAB>people`
  - times are chapter-local `HH:MM:SS.mmm`.
  - on save, these are converted to archive-global `start<TAB>end<TAB>people` in `people.tsv`.
- Dialogue subtitles stay synced between timeline boxes and the bulk-edit table.
- `confidence` and `source` metadata fields are still preserved in `subtitles.tsv` but hidden in tuner UI.

Step 5: Summary + save

1. Click the top `5/5 Summary + Save` step button.
2. Confirm bad-frame, gamma, and people subtitle settings.
3. Click `Save and Return to Chapters`.
4. The tuner saves:
   - `metadata/<archive>/render_settings.json` (`BAD_FRAMES` + gamma)
   - `metadata/<archive>/people.tsv` (people subtitle ranges)
   - `metadata/<archive>/subtitles.tsv` (dialogue subtitle ranges and optional metadata)

How `BAD_FRAMES` maps to AviSynth `FreezeFrame`
-----------------------------------------------

- The tuner writes metadata only; it does not modify video pixels directly.
- During render, the AviSynth script reads chapter `BAD_FRAMES` from `render_settings.json` and converts them into repair ranges.
- Each repair range becomes one or more `FreezeFrame(start, end, source)` calls.
- `source` is selected automatically from nearby clean frames:
  - prefer the next clean frame after a bad range
  - fall back to a previous clean frame if needed
- Contiguous/near-contiguous bad frames are merged so replacements are stable across bursts.
- By default BAD frames are stored per chapter title. To have a larger chapter
  (for example "01 Full Wedding") inherit bad frames that were marked in
  overlapping subchapters, set:
  - `archive_settings.inherit_bad_frames_from_overlaps = true`
    in `metadata/<archive>/render_settings.json`.

Status and cancellation
-----------------------

- During frame loading, the overlay shows progress and estimated ready time.
- Use `Cancel` on the loading overlay to stop a long load.

Practical workflow tips
-----------------------

- Tune IQR first, then do manual frame-by-frame overrides.

Notes
-----

- Render extract behavior is enabled in this plain wizard for full chapter-frame loading.
- This is intended for local single-user operation.
