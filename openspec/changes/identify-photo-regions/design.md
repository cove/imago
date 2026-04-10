## Context

Album pages are scanned as multi-photo "view" JPGs (`*_V.jpg`). Individual photographs are packed edge-to-edge — there may be no visible background between them. The existing pipeline (`ai_page_layout.py`) handles basic layout classification but relies on geometry heuristics, not vision-model reasoning for seam detection between adjacent photos. `xmp_sidecar.py` already manages XMP write-back using Python's `xml.etree`, with a rich namespace registry. The MCP server (`mcp_server.py`) uses FastMCP and already exposes photoalbum job endpoints. LM Studio is already integrated via `_lmstudio_helpers.py`; its base URL and model selection live in `ai_models.toml`.

## Goals / Non-Goals

**Goals:**
- Add a new `ai_models.toml` entry for `google/gemma-4-26b-a4b` as the `view-region` model
- Add `ai_view_regions.py` in `photoalbums/lib/` that calls LM Studio with a vision prompt and returns normalised bounding boxes for each detected photo within a view image
- Write MWG-RS `RegionList` XMP data (namespace `http://www.metadataworkinggroup.com/schemas/regions/`) to the view JPG's sidecar or embedded XMP
- Associate page-level captions with detected regions (closest spatial match; broadcast when ambiguous)
- Expose two new MCP tools: `photoalbums_detect_view_regions` (trigger detection on a single image or album) and `photoalbums_review_view_regions` (return image + current region boxes for external AI validation)

**Non-Goals:**
- Splitting photos out of the view image (future step)
- Assigning ShownLocation data (future step — regions must exist first)
- Training or fine-tuning the vision model
- Handling non-view images (scans, derived images)

## Decisions

### Vision API call format
Use the LM Studio OpenAI-compatible `/v1/chat/completions` endpoint with a `vision` message (base64 image). The prompt asks the model to return a JSON array of bounding boxes `[{index, x, y, width, height, confidence, caption_hint}]` as **normalised 0.0–1.0 coordinates** (top-left origin). Temperature is set to `0.0` for fully deterministic output. Rationale: structured JSON output is more reliable than free-text; normalised coords are scale-invariant regardless of how the model internally resizes the input image.

The user prompt includes the original pixel dimensions of the image (e.g., `"The full image is 3840×2880 pixels."`) so the model can reason about region boundaries at the correct scale.

Alternatives considered:
- Returning pixel coordinates from the model: initially implemented, but caused a "boxes bunched in the upper-left at 1/4 size" bug because vision models internally resize input to their own patch grid (e.g. 448×448), making pixel coords unreliable regardless of what we send
- Asking the model to describe regions in prose then parsing: rejected as fragile

### XMP region schema
Use MWG-RS `mwg-rs:RegionList` with `mwg-rs:RegionInfo` → `mwg-rs:AppliedToDimensions` + `mwg-rs:RegionList/rdf:Bag/rdf:li`. Each region carries:
- `mwg-rs:Type = "Photo"`
- `mwg-rs:Name` = region index (e.g., `"photo_1"`)
- `stArea:x`, `stArea:y`, `stArea:w`, `stArea:h` (normalised 0–1, centre-point origin as per MWG spec)
- `dc:description` = associated caption

Rationale: MWG-RS is the standard used by Lightroom, digiKam, and other tools; it will survive round-trips through exiftool.

Alternative: custom `imago:` namespace regions — rejected because it would not render in standard photo tools.

### Caption association
Spatial nearest-centre heuristic: the caption whose bounding box (or page text region) is closest to a photo region's centre gets assigned to that region. If no caption geometry is available, or if the closest caption is equidistant to two regions (within 10% of image width), the page caption is broadcast to all regions with a `captionAmbiguous=true` flag.

### MCP endpoint design
Two tools added to the existing `mcp_server.py`:
1. `photoalbums_detect_view_regions(album_id, page=None, force=False)` — runs detection (async job) and returns `job_id`
2. `photoalbums_review_view_regions(album_id, page)` — returns structured JSON with image path, image dimensions, and current region list; intended for an external AI to validate boundaries

Detection results are written directly into the `_V.jpg` XMP sidecar (the existing `.xmp` file alongside the view image). The presence of a `mwg-rs:RegionList` block in the XMP is the cache signal — if it exists and `force=False`, detection is skipped. The XMP is the single source of truth; no separate JSON cache file is written.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Model returns malformed JSON | Wrap in retry loop (up to 3 attempts) with a stricter JSON-only prompt; fall back to empty region list with error logged |
| Adjacent photos with no visible seam confuse the model | Prompt includes explicit instruction to look for contextual clues (content discontinuity, perspective change) rather than borders; confidence score returned per region |
| MWG-RS normalised coordinates differ from pixel coords | Conversion utility written and unit-tested separately from the API call |
| LM Studio offline / slow | Detection is run as an async job (existing `JobRunner`); timeout configurable; review endpoint returns cached data even if model is offline |
| Writing to XMP sidecar modifies file mtime | Expected; existing pipeline already writes XMP sidecars; document this for downstream hash checks |

## Open Questions

- Should region detection run automatically as part of `scanwatch` (when a new view image appears) or only on explicit CLI/MCP request? → Default: explicit only; auto-trigger can be added later.
- Is the LM Studio model address for view-region detection always the same host as caption/OCR, or should it be separately configurable? → Use same `lmstudio_base_url` for now; can split later.
