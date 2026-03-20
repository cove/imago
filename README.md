# imago

Family media archive for VHS digitization, photo album preservation, and shared face IDs.

## Projects

- [`vhs/`](vhs/) - VHS tape digitization pipeline: bad-frame detection, gamma correction, subtitle generation, render wizard
- [`photoalbums/`](photoalbums/) - Scanned photo album pipeline: TIFF processing, metadata embedding, page stitching
- [`cast/`](cast/) - Shared face identity store + local web review UI (text files)
- [`viewer/`](viewer/) - Static cloud media site for public Google Drive/OneDrive photo and video links

## Setup

Create or update the repo environment from the repo root:

- `uv sync`

Optional task runner commands, if you install `just`:

- `just sync`
- `just bootstrap`
- `just test`

Bootstrap bundled runtime binaries used by VHS workflows on Windows/Linux:

- `uv run python scripts/bootstrap_runtime.py`

## Tests

Run all project tests from repo root:

- `uv run pytest -q`
- `.\scripts\test.ps1 -q` (also runs the per-project Skylos duplicate-code gate)

Enable the repo pre-push hook (runs this test suite before push):

- `git config core.hooksPath .githooks`

Run a single project test suite (same command shape in each project):

- `uv run pytest cast/tests -q`
- `uv run pytest photoalbums/tests -q`
- `uv run pytest vhs/test -q`

## AI Model Storage

Downloaded AI model weights are stored under the repo-level `modes/` directory.

## Cast quick start

1. Initialize files:
   - `uv run python cast.py init`
2. Run local web UI:
   - `uv run python cast.py web`
3. Open:
   - `http://127.0.0.1:8093`

From the Cast web UI you can:

- ingest a photo path and detect/crop faces
- bulk scan `Photo Albums/*_View` for `.jpg` photos and detect/crop faces
- ingest a VHS/video path (sampled frames) and detect/crop faces
- review queued faces and label who each person is (one-by-one naming prompt included)
- auto-prune obvious non-face detections from the pending queue
- optional "Skip Statues/Paintings" mode for stricter filtering
- click a face crop to view the full source image (with face highlight box)

Default Cast storage files:

- `cast/data/people.json`
- `cast/data/faces.jsonl`
- `cast/data/review_queue.jsonl`

## Photoalbums entrypoint

Run photoalbums workflows only via:

- `uv run python photoalbums.py ...`
- `uv run python -m photoalbums ...` (equivalent)

All `photoalbums/lib/*.py` modules are internal libraries, not user entrypoint scripts.
Also do not run `photoalbums/cli.py` or `photoalbums/commands.py` directly.

The photoalbums AI index has built-in Audrey/Leslie Cordell album hints:
`Family*` collections are described as `Family Photo Album`, geography-named collections are described as `Photo Essay`, and likely blue/white title covers with matching OCR are treated as album cover/title pages.
Canonical album titles are inferred from cover OCR and reused across later pages, and printed cover titles from `P00`/`P01` are now reused in caption prompts so descriptions can prefer the exact book name as printed, such as `Mainland China Book 11`, instead of raw filename-style identifiers.
When captions mention non-English visible text, the AI index now asks the model for explicit English translations and appends those translations in the final caption text.
When captions identify a place clearly, the AI index now asks the model for a geocoding-ready place name, resolves it online, and writes the resulting GPS into standard Exif XMP GPS fields (`exif:GPSLatitude`, `exif:GPSLongitude`, `exif:GPSMapDatum`, `exif:GPSVersionID`) so photo software can read location data directly from the sidecar.
Page segmentation is off by default during AI indexing. Set `page_split_mode` to `auto` in `render_settings.json` only for archives where sub-photo splitting is explicitly desired.
Images larger than 30 MB are processed through a temporary scaled-down copy for OCR/object/caption models to keep AI indexing stable on oversized pages.

Photoalbums OCR now uses a local Qwen vision model by default. Download the model into `models/photoalbums/hf` first, or point `QWEN_OCR_MODEL` at a local model path before running `photoalbums ai`.

## Structure

```
imago/
  .venv/         <- uv-managed virtual environment
  vhs/           <- VHS pipeline (uv run python vhs.py ...)
  photoalbums/   <- Photo album pipeline
  cast/          <- Shared face identity module + web UI
  viewer/        <- Static cloud media viewer
  cast.py        <- Cast CLI entrypoint
  photoalbums.py <- Photo Albums CLI entrypoint
```


