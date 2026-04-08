## Why

Each scanned album page lives in two sibling directories:

- `<Album>_Archive/` — high-resolution per-scan TIFFs, one or more per page (e.g. `Egypt_1975_B00_P26_S01.tif`). These are the archival masters.
- `<Album>_View/` — a single stitched JPEG per page (e.g. `Egypt_1975_B00_P26_V.jpg`). This is a colour-corrected composite of all scans for that page and is the working image for AI processing.

View images contain multiple photographs packed edge-to-edge with no background border between them. To associate per-photo ShownLocation XMP metadata and eventually split them into individual files, we must first detect the boundary of each photo within the view JPG — a task that requires vision-model reasoning, not simple edge detection.

## What Changes

- New pipeline step that sends a `_V.jpg` view image to the local LM Studio vision model (`google/gemma-4-26b-a4b`) and receives back bounding-box regions for each photo within that page
- XMP region metadata (MWG-RS `RegionList`) written to the `_View/` JPG's XMP sidecar describing each detected photo region
- Caption text from the page is associated with the closest region, or broadcast to all regions when ambiguous
- New MCP tools added to the existing `mcp_server.py` that allow another AI agent to trigger detection, review boundaries, and correct them before any destructive split step

## Capabilities

### New Capabilities

- `view-region-detection`: Detect individual photo boundaries within a stitched view JPG using the LM Studio vision API; return normalised bounding boxes with confidence scores
- `view-xmp-regions`: Write detected regions as MWG-RS `RegionList` XMP metadata on the source view JPG, including caption association per region
- `mcp-region-review`: MCP endpoint to process an album's view images and expose region data (image + boxes) for external AI validation

### Modified Capabilities

## Impact

- New dependency: LM Studio running locally with `google/gemma-4-26b-a4b` loaded (OpenAI-compatible API at `lmstudio_base_url` in `photoalbums/ai_models.toml`)
- XMP metadata written to `_View/*.jpg` XMP sidecar files (non-destructive; original pixels unchanged; `_Archive/` TIFFs are not touched)
- The `_View/*.xmp` sidecar is the single source of truth — detected regions are stored as `mwg-rs:RegionList` directly in the existing XMP file; no separate cache file is written
- New tools added to the existing `mcp_server.py` (FastMCP); no new server process required
- Affected code areas: `photoalbums/lib/` (new `ai_view_regions.py`), `photoalbums/lib/xmp_sidecar.py`, `mcp_server.py`, `photoalbums.py` CLI
