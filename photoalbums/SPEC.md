# Photo Albums Processing Pipeline Specification

**Purpose:** Technical specification for reimplementing the photo album digitization and AI indexing pipeline.

**Scope:** Complete workflow from raw TIFF scans through AI-enriched metadata to final indexed photo archives.

---

## 1. Input & Prerequisites

### 1.1 Input Format
- **Source:** Raw TIFF scans from physical album pages
- **Location:** `{PHOTOS_ROOT}/_Archive/` directories
- **Filename Pattern:** `{Collection}_{Year}_B{Book:02d}_P{Page:02d}_S{Scan:02d}.tif`
  - Collection: alphanumeric (e.g., "Egypt", "Cordell")
  - Year: 4-digit YYYY or YYYY-YYYY range
  - Book: 2-digit number (00-99)
  - Page: 2-digit page number within album
  - Scan: 2-digit scan number per page (multiple scans per page due to oversized originals)
- **Scanner Configuration:** All preprocessing disabled—scans are raw, unrotated TIFF files

### 1.1a Scanner Hardware & Specifications
- **Device:** Epson Perfection V19 II flatbed scanner
- **Max Scan Width:** 8.5 inches (at full width for document scanning)
- **Physical Constraint:** Photo album pages are typically 11" × 14" or larger, exceeding scanner width
- **Scanning Method:** Multiple overlapping passes per album page (e.g., left portion, center, right portion)
- **Overlap Requirements:** 8-42% horizontal overlap between adjacent scans to enable stitching
- **Resolution:** Scans at native optical resolution (~600-1200 DPI depending on album condition)
- **Preprocessing:** Disabled in scanner settings; all color correction and auto-rotation handled in post-processing

### 1.2 Directory Structure

All artifacts for a single album live in three sibling directories with the same base name `{Base} = {Collection}_{Year}_B{Book}`:

```
{PHOTOS_ROOT}/
├── {Base}_Archive/
│   ├── (raw TIFF scans, one or more per album page)
│   └── (subdirectories for multi-page/multi-scan albums)
├── {Base}_Pages/
│   └── (stitched page-view JPEGs and their .xmp sidecars)
└── {Base}_Photos/
    └── (cropped individual photos and their .xmp sidecars)
```

### 1.3 File Naming Convention

The pipeline keys nearly every operation off filename structure, so consistent naming is mandatory at ingest. Detailed regexes and validation rules live in Section 11; the four canonical patterns are summarized here:

| Artifact | Pattern | Example |
|----------|---------|---------|
| Raw TIFF scan | `{Collection}_{Year}_B{Book}_P{Page:02d}_S{Scan:02d}.tif` | `Egypt_1975_B01_P05_S01.tif` |
| Stitched page view | `{Collection}_{Year}_B{Book}_P{Page:02d}_V.jpg` | `Egypt_1975_B01_P05_V.jpg` |
| Derived photo view | `{Collection}_{Year}_B{Book}_P{Page:02d}_D{Derived:02d}-{Iter:02d}_V.jpg` | `Egypt_1975_B01_P05_D00-00_V.jpg` |
| XMP sidecar | `{image_filename}.xmp` (alongside the JPEG/TIFF) | `Egypt_1975_B01_P05_V.xmp` |

**Type Tokens** — control what stage and format a file represents:

| Token | Meaning | Storage Format | Notes |
|-------|---------|---|---|
| `_S##` | Raw scan (archive) | `.tif` or `.png` | Original input from scanner; never processed |
| `_D##-##` | Derived image (intermediate) | `.tif` | Extracted crop or intermediate product; not a final view |
| `_V` | View (final rendered) | `.jpg` | Always marks a rendered/display-ready output; page views and crop views both end in `_V.jpg` |
| `_D##-##_V` | View of derived image | `.jpg` | Extracted crop rendered as JPEG; the combination of `_D##-##` (derived) + `_V` (view) |

**Invariant rules:**
- `_V` always and only marks a view output. `_S##` always and only marks an archive scan.
- `_D##-##` identifies a derived image; append `_V` for the view JPEG.
- Archive files are `.tif` and `.png`; view files are `.jpg` — no exceptions.
- `dc:source` on any view file references the archive TIF scan(s) it was derived from.
- Pages are numbered starting at P01. P00 is not a valid page number.
- XMP sidecars share the same stem as their companion image file (`.xmp` extension).

**Field semantics:**
- **Collection** — alphanumeric, no underscores (e.g., `Egypt`, `Cordell`)
- **Year** — `YYYY` or `YYYY-YYYY` range
- **Book** — two digits (`00`–`99`)
- **Page** — two digits in range `01`–`99` (P00 is not valid; leading zero required: `P05`, not `P5`)
- **Scan** — two digits (`01`, `02`, …) indicating the scan index for an oversized page
- **Derived** — two digits (`00`, `01`, …), per-page index of a derived photo extracted from the page
- **Iter** — two-digit version of the derived (`D##`) image; the first version is `01`, and increments when the derived photo is re-derived (e.g., recropped, regeometried, colorized) from the same source page region
- `_V` is a literal suffix marking a "view" (rendered/derived) artifact

### 1.4 Required Specs (What the Pipeline Needs)

The pipeline expects these properties of its environment and inputs to be provided by the operator, regardless of how the values are stored:

**Per-archive specs the system needs:**
- **Photos root:** absolute path to the `{PHOTOS_ROOT}` directory containing the `_Archive` / `_Pages` / `_Photos` siblings described in 1.2
- **People roster:** path (or null) to a CSV listing known people for face-matching; expected columns include name and reference image path
- **Album-title hint:** a human-readable album title string used to seed the metadata extractor (see Section 5.3.1)

**AI baseline specs:**
- **OCR/caption model and host:** `google/gemma-4-31b` served by an lmstudio-compatible endpoint at `http://127.0.0.1:1234/v1` (or a `localhost` equivalent)
- **Layout-analysis pipeline:** Docling configured with preset `granite_docling`, backend `auto_inline`, device `auto`, retries `3`
- **Photo-restoration pipeline:** RealRestorer at the pinned commit (Section 5.0); skipped automatically if the host has insufficient RAM

**Render specs (defaults when no per-album override is supplied):**
- JPEG quality: `95`
- Render scale: `1.0` (full resolution)
- Stitch detector strategy: the full `AFFINE_STITCH_ATTEMPTS` sequence (Section 2.4), with optional per-album overrides like `"sift"`

**External-service specs:**
- **Nominatim:** reachable HTTPS endpoint, default `https://nominatim.openstreetmap.org`, with a custom `User-Agent` (`imago-photoalbums-ai-index/1.0`)
- **Local cache directory:** writable `{PHOTOALBUMS_DIR}/data/` for `geocode_cache.json` and similar artifacts

### 1.5 Scan Acquisition & Validation (Watcher System)

Before raw TIFF scans can enter the main pipeline, they must be:
1. **Acquired:** Transferred from the scanner to a designated "incoming scans" directory
2. **Validated:** Tested to ensure stitching is possible (if multiple scans per page)
3. **Registered:** Renamed and moved into the archive directory with the correct naming convention

A long-running **ScanWatchService** monitors an `{INCOMING_NAME}` directory (typically `{PHOTOS_ROOT}/incoming/`) for new `.tif` files matching **either** of these patterns:
- `incoming_scan.tif` (default single-scan placeholder)
- `incoming_scan_NNNN.tif` (backlog: numbered 0001–9999 for multiple queued scans)

When detected:

**Per-file workflow:**
1. Log the incoming scan event with a unique ID and register it as pending
2. **Determine target filename:** Inspect the archive directory's existing TIF files and compute the next filename to use
   - List all `.tif` files in `{Archive}_Archive/`
   - Filter for valid album-naming pattern (`{Collection}_{Year}_B{Book}_P##_S##.tif`)
   - If none exist: target is `{prefix}_P01_S01.tif` (first page, first scan)
   - If some exist: sort by filename and increment the last one:
     - If last page is P01, move to P02_S01 (start a new page)
     - Otherwise, increment scan number; if scan > 2, wrap to next page P##_S01
3. **Group scans:** Collect all pending scans for the same (Collection, Year, Book, Page)
4. **Validate stitch:** Attempt to stitch multi-scan groups using SIFT 0.3 / BRISK 0.1 detectors (simplified subset, not the full `AFFINE_STITCH_ATTEMPTS`)
   - If stitch succeeds: mark the group as valid and ready for pipeline processing
   - If stitch fails: alert the operator (beep + Windows modal requesting a rescan); do NOT apply
5. **Apply on success:**
   - Rename incoming file to target filename (e.g., `incoming_scan_0001.tif` → `Egypt_1975_B01_P05_S01.tif`)
   - Move file to `{Archive}_Archive/` directory
   - Process TIFF in place (convert to standard format, ensure proper orientation, etc.)
   - Create a `ScanEvent` record with `status=applied`, recording the timestamp, target name, and `stitch_validated` flag
6. **Sync archive state:** Update the `ArchiveState` for the target album set:
   - Increment page scan count: `page_scan_counts[page] += 1`
   - If stitch failed and now retrying, remove from `needs_rescan_pages`

**Configuration:**
- **Scanning directory:** `PHOTO_SCANNING_DIR` (default: `{PHOTOS_ROOT}/scanning/`)
- **Incoming subdir:** `INCOMING_NAME` (default: `incoming`)
- **Alert behavior:** Beep + Windows messagebox on stitch failure (silent on Unix/Linux)
- **Stitch validation:** Simplified subset (SIFT 0.3, BRISK 0.1) vs. full pipeline's `AFFINE_STITCH_ATTEMPTS` (faster feedback)

**Data structures:**
- **ScanEvent:** `{id, archive_dir, incoming_path, created_at, updated_at, status, target_name, page_num, stitch_validated, note}`
  - status: `pending`, `processing`, `applied`, `failed`
  - stitch_validated: `True` if group stitched successfully, `False` if failed, `None` if single scan
- **ArchiveState:** `{archive_dir, incoming_path, pending_event_ids, page_scan_counts, needs_rescan_pages}`
  - Tracks which pages have how many scans and which pages need operator attention

**Note:** The watcher is optional — raw scans can be manually renamed and placed in `{Archive}_Archive/` without going through the watcher. The watcher is a convenience for operator feedback during the ingest phase.

---

## 2. Page Rendering (Stitch Oversized Pages)

### 2.1 Purpose & Hardware Context
Photo album pages (typically 11" × 14" or larger) exceed the maximum scan width of the Epson Perfection V19 II scanner (which has a physical scan width of ~8.5" for standard documents). Therefore:

- **Physical Constraint:** The Epson Perfection V19 II can only scan up to 8.5 inches wide at native resolution
- **Solution:** Scan each oversized album page in multiple overlapping passes (left column, center, right column, etc.)
- **Goal:** Combine these overlapping scans into a single unified page view JPEG with correct geometry

The stitching process reconstructs the full album page from 2-4 overlapping TIFF scans per page, accounting for varying overlap amounts and slight perspective differences in the scanned regions.

### 2.2 Purpose
Combine multiple TIFF scans of a single album page into a single stitched view JPEG.

### 2.2 Entry Point
- **Input:** Raw TIFF scan groups under `{PHOTOS_ROOT}/_Archive/`
- **Output:** Page view JPEG under `{PHOTOS_ROOT}/_Pages/` as `{Album}_{Page}_V.jpg`

### 2.3 Scan Grouping
Scans are grouped by (Collection, Year, Book, Page). All scans with the same page identifier belong to one stitched output.

### 2.4 Stitching Algorithm

**Affine Stitching (Primary Method)**
Try stitching attempts in order; first successful attempt is used. Each attempt uses different feature detector and confidence settings:

| Attempt | Detector | Confidence Threshold | Notes |
|---------|----------|----------------------|-------|
| 1 | SIFT | 0.3 | Most permissive: high match tolerance |
| 2 | SIFT | 0.1 | Tighter: require stronger matches |
| 3 | AKAZE | 0.3 | Switch to AKAZE detector, high tolerance |
| 4 | AKAZE | 0.1 | AKAZE with tight matching |
| 5 | BRISK | 0.1 | BRISK detector as final option |

- Feature detectors (SIFT, AKAZE, BRISK) identify characteristic points between overlapping images
- Confidence threshold determines how strong feature matches must be to accept the stitching
- Lower threshold = more permissive matching (may include false matches)
- If current attempt fails or produces invalid output, proceed to next attempt
- First successful stitch is used; remaining attempts skipped

**Linear Fallback (Secondary Method)**
When all five affine attempts fail **and exactly two input scans exist**, fall back to linear (sequential) stitching with optimized overlap detection. For three or more scans, no linear fallback is attempted—the build raises an error so the operator can investigate.

**Linear Stitching Parameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| target_width | 640 px | Resize images to this width for faster analysis |
| min_overlap | 0.08 | Minimum acceptable overlap between scans (8% of width) |
| max_overlap | 0.42 | Maximum acceptable overlap (42% of width) |
| max_vertical_shift | 0.08 | Allow up to 8% vertical misalignment |
| min_shared_height | 0.6 | Overlapping regions must share 60% of height |
| min_detail | 0.05 | Require 5% detail/features in overlap for validation |
| overlap_search_step | 12 | Search for optimal overlap in 12-pixel increments |
| vertical_search_step | 4 | Search for optimal vertical alignment in 4-pixel increments |
| refine_overlap_radius | 12 px | Fine-tune overlap within ±12 pixels of best guess |
| refine_vertical_radius | 4 px | Fine-tune vertical alignment within ±4 pixels |
| expansion_ratio | 1.02 | Expand final canvas by 2% to prevent edge clipping |

**Algorithm:**
1. Resize scans to target width (640 px) for speed
2. Search for optimal overlap amount (8%–42% range) in 12-pixel steps
3. Search for optimal vertical alignment in 4-pixel steps
4. Validate that overlapping regions share enough detail and height
5. Refine best result with tighter search radius (±12 px overlap, ±4 px vertical)
6. Expand output canvas by 2% to ensure no clipping of edges
7. Composite aligned images into final stitched output

### 2.5 Image Format Handling
- **Input Formats:** TIFF (with auto-orient), JPEG, PNG
- **Read Methods:** 
  - Primary: Pillow (with EXIF transpose)
  - Fallback: ImageMagick (`magick` command) for problematic formats
- **Output Format:** JPEG (quality=95, RGB color space)

### 2.6 Output Validation
- Check file size > 0
- Validate JPEG/TIFF integrity with Pillow
- Skip existing valid outputs (unless `--force`)
- Re-render if existing output fails validation

### 2.7 Large Image Handling
- Lift PIL decompression bomb limits: `allow_large_pillow_images()`
- Set `OPENCV_IO_MAX_IMAGE_PIXELS = 2^40` for OpenCV/stitching library

---

## 3. Photo Region Detection (Docling Pipeline)

### 3.1 Purpose & Overview
Identify individual photographs within a stitched page view JPEG using Docling's document layout analysis pipeline. Docling analyzes the visual structure of the page to detect and extract photo regions.

### 3.2 Entry Point
- **Input:** Page view JPEG (`{Album}_P{Page:02d}_V.jpg`)
- **Output:** Bounding boxes as MWG-RS region metadata in XMP sidecar

### 3.3 Docling Pipeline Configuration

**Library:** `docling` **2.88.0**
**Pipeline Type:** Standard image pipeline (image layout analysis, NOT document text extraction, NOT OCR)

**Configuration Settings (Embedded):**
| Setting | Value | Purpose |
|---------|-------|---------|
| **preset** | `granite_docling` | Pipeline tuning optimized for document layout analysis |
| **backend** | `auto_inline` | Auto-select best inference backend (fallback options: "transformers" or "mlx") |
| **device** | `auto` | Auto-detect available hardware (GPU/MPS/CPU) |
| **retries** | `3` | Retry up to 3 times if first run detects no photo regions |
| **do_ocr** | `False` | Explicitly disable OCR—layout analysis only, no text extraction |

### 3.4 Docling Pipeline Processing Steps

**Step 1: Initialize Document Converter**
Create a DocumentConverter instance configured with:
- Input format: IMAGE
- Backend: ImageDocumentBackend (image-optimized, not document-optimized)
- Pipeline options: do_ocr=False (CRITICAL: layout analysis only, no text extraction)
- Accelerator: Use specified device (GPU/MPS/CPU as configured)

**Step 2: Run Pipeline with Retry Loop**
For each attempt (up to max_attempts):
1. Call converter.convert(image_path)
2. This internally runs Docling's standard image layout analysis
3. Docling outputs a document object with detected items and metadata
4. If regions detected → break from retry loop
5. Else if attempts remaining → log warning and retry
6. Else if no more attempts → log final warning, return empty list

**Step 3: Extract Picture Items from Document**
Iterate through all items in the document:
1. Filter for items with label = "PICTURE" (Docling's classification for photos/images)
2. Skip items without provenance metadata (provenance contains layout information)
3. Use first provenance entry from the list
4. Collect all matching picture items

**Step 4: Extract Bounding Box from Provenance**
For each picture item:
1. Get page height (from document.pages if available, else use image_height)
2. Extract bbox from provenance metadata
3. Convert bbox coordinate system to top-left origin (if necessary)
4. Extract four values: left, top, right, bottom (in pixels)
5. Calculate dimensions: width = right - left, height = bottom - top
6. Clamp to valid ranges: x ≥ 0, y ≥ 0, width ≥ 1, height ≥ 1

**Step 5: Extract Caption Hints (Optional, from Docling)**
If item has associated captions:
1. Get first caption reference
2. Caption reference contains a "cref" value (reference string, format: "path/to/text/INDEX")
3. Extract the numeric index from the end of cref
4. Look up that index in document.texts array
5. If found, extract and clean the text value as a caption hint
6. If not found or parsing fails, use empty string (silent failure is acceptable)
7. **Note:** This is just a hint. The actual region caption comes from **Gemma-4 metadata extraction** (see Section 5.3.1), not from Docling.

**Step 6: Create Region Result Object**
For each picture, record a frozen dataclass with these fields:
- `index` (int): sequential count (0, 1, 2, ...)
- `x`, `y` (int): top-left corner pixel coordinates
- `width`, `height` (int): dimensions in pixels
- `confidence` (float, default 1.0): detector confidence; Docling does not expose a per-item score, so this is 1.0 unless filled later
- `caption_hint` (str, default ""): hint extracted from Docling captions (may be empty)
- `location_hint` (str, default ""): optional location hint (usually empty from Docling)
- `location_payload` (dict, default {}): structured location result added later in the pipeline (empty when produced by Docling)
- `person_names` (list[str], default []): list of person names (empty from Docling)
- `photo_number` (int, default 0): optional photo identifier

**Important:** The actual caption text that becomes `mwg-rs:Name` in XMP comes from the **Gemma-4 metadata extraction** (lmstudio model), not from Docling. Docling only provides bounding boxes and optional hints.

**Step 7: Coordinate Conversion to MWG-RS Normalized Format**
After extracting pixel coordinates, convert to normalized center-point format for XMP storage:
- Calculate center point: cx = (x + width/2) / img_width, cy = (y + height/2) / img_height
- Calculate normalized dimensions: nw = width / img_width, nh = height / img_height
- Result: cx, cy, nw, nh (all values in range 0.0–1.0)
- Store in XMP as MWG-RS region with this normalized format

### 3.5 Region Validation

Validate detected regions before persisting to XMP:

**Validation Rules:**
- **Zero Area:** If `width <= 0` or `height <= 0` → reject with severity `hard`, reason `zero_area`
- **Full Page:** If region occupies ≥90% (`_MAX_SINGLE_REGION_PAGE_FRACTION = 0.90`) of page area → reject with severity `hard`, reason `full_page`
- **Bounds Clamping:** Coordinates outside the image are silently clamped to `[0, img_w] × [0, img_h]`; no warning is emitted

**Validation Output:**
Return a validation result object containing:
- valid: boolean flag (true if all regions passed validation, false if any rejections)
- kept: list of RegionResult objects that passed validation
- failures: list of RegionFailure objects describing rejected regions with reason codes

### 3.6 Retry Logic

If no regions found on attempt N:
1. Log warning: `"no picture items found for {path} on attempt N/MAX, retrying"`
2. Record error: `{"attempt": N, "error": "no_regions"}`
3. Retry (up to `retries` times)
4. If all retries exhausted with no regions: log final warning and return empty list

---

## 4. Photo Cropping from Regions

### 4.1 Purpose
Extract individual cropped photos from page view JPEG using detected region bounding boxes.

### 4.2 Entry Point
- **Input:** Page view JPEG + XMP region metadata
- **Output:** Individual crop JPEGs in `{PHOTOS_ROOT}/_Photos/`
- **Naming:** `{Album}_P{Page:02d}_D{Derived:02d}-00_V.jpg`

### 4.3 Region Extraction
For each region in XMP sidecar:
1. Read MWG-RS normalized center-point coordinates: cx (center x), cy (center y), width, height (all in 0.0–1.0 range)
2. Convert normalized to pixel coordinates:
   - Calculate float pixel positions: left_f = (cx - width/2) × img_w, top_f = (cy - height/2) × img_h, right_f = (cx + width/2) × img_w, bottom_f = (cy + height/2) × img_h
   - Round and clamp to image bounds: left = max(0, round(left_f)), top = max(0, round(top_f)), right = min(img_w, round(right_f)), bottom = min(img_h, round(bottom_f))
3. Extract rectangular region from image using bounds (top:bottom, left:right)

### 4.4 Photo Restoration

**Model:** RealRestorer diffusion pipeline
- **Repo:** `https://github.com/yfyang007/RealRestorer.git`
- **Model Name:** `RealRestorer/RealRestorer`
- **Availability Check:** Try importing from `diffusers` library; skip if unavailable
- **RAM Requirement:** Model repo size ≈41.8 GB; skip if installed RAM < repo size

**Restoration Inference Process:**
Call the RealRestorer pipeline with the following parameters:
- Input: crop_image (PIL Image object or equivalent)
- Prompt: "Please restore this low-quality image, recovering its normal brightness and clarity."
- Inference steps: 28 (diffusion steps)
- Guidance scale: 3.0 (strength of prompt adherence)
- Seed: 42 (fixed seed for reproducibility)
- Size level: 1024 (output resolution)
- Output: First image from results array

**Runtime Configuration:** Automatically detect hardware and select precision:
- If CUDA GPU available → Use bfloat16 precision with "cuda" device, enable CPU offload for memory efficiency
- Else if MPS (Metal Performance Shaders) available → Use float32 precision with "mps" device
- Else → Use float32 precision with "cpu" device

**Fallback Behavior:**
- If RealRestorer unavailable (not installed): return original crop unchanged, log "restoration_unavailable"
- If inference fails: return original crop unchanged, log error with exception details

### 4.5 Output Handling
- **Format:** JPEG (quality=95, RGB)
- **Skipping Existing:** Skip crop output if already exists (unless `--force`)
- **Force Restoration Flag:** Re-run restoration on existing crops without forcing full crop step

---

## 5. AI Processing Pipeline

### 5.0 Current Baseline (Starting Point for Reimplementation)

**Important:** This section documents the exact AI models, services, and configurations used in the current implementation. This baseline is self-contained—no external files needed to understand the system.

**Current AI Stack:**

**1. OCR/Caption Extraction Engine**
- **Service Type:** Local inference server (lmstudio)
- **Server URL:** `http://127.0.0.1:1234/v1` (TOML config) or `http://localhost:1234/v1` (code default fallback)
- **Model Name:** `google/gemma-4-31b` (Google Gemma 4, 31 billion parameters)
- **Model Source:** Hugging Face / lmstudio model registry
- **Per-purpose model lists:** `pc = ["google/gemma-4-31b"]` (people-count), `primary = ["google/gemma-4-31b"]` (general)

**2. View Region Detection (Photo Detection)**
- **Service Type:** Docling layout analysis pipeline
- **Pipeline Preset:** `granite_docling`
- **Backend:** `auto_inline` (automatically selects best backend: transformers or mlx)
- **Device:** `auto` (automatically detects GPU/MPS/CPU)
- **Retry Attempts:** 3 (if no regions found, retry up to 3 times)

**3. Photo Restoration**
- **Model:** RealRestorer diffusion pipeline
- **Repository:** https://github.com/yfyang007/RealRestorer.git
- **Pinned Commit:** `fa2a3e3c23768eb94748c5855d83cc2e340ab13b`
- **Inference Parameters:**
  - num_inference_steps = 28
  - guidance_scale = 3.0
  - seed = 42
  - size_level = 1024
- **Restoration Prompt:** "Please restore this low-quality image, recovering its normal brightness and clarity."

**4. Face Recognition (Optional)**
- **Service Type:** Cast (internal face matching service)
- **Model:** buffalo_l (via InsightFace)
- **Purpose:** Match detected faces against people roster

**5. Object Detection (Optional)**
- **Model:** YOLOv11 nano (Ultralytics)
- **Model Path:** `models/yolo11n.pt` (resolved relative to the photoalbums package)
- **Default Confidence Threshold:** `0.30` (configurable)

### 5.1 Overview
For each page view, run a series of AI analyses on the original page and extracted crops, writing results to XMP sidecars.

### 5.2 Processing Modes
- **default:** Process all pages/crops without XMP
- **gps:** Reprocess only GPS/location estimation (re-runnable)
- **force:** Force full reprocessing even if XMP exists
- **skip-existing:** Skip pages/crops with valid existing XMP

### 5.3 Page-Level Processing

#### 5.3.1 OCR & Caption Extraction

This step extracts a structured metadata record per visible photograph (caption, location, estimated date, OCR'd scene text, people count) from the full page view image.

**Baseline:**
- **Service:** lmstudio-compatible local HTTP server at `http://127.0.0.1:1234/v1`
- **Model:** `google/gemma-4-31b`
- **Endpoint:** `/v1/chat/completions` with `response_format` set to `json_schema` for structured output

**Prompt Source:** `photoalbums/prompts/ai-index/metadata/` (`system.md`, `user.md`, `schema.json`, `params.toml`)

**System Prompt** (`system.md`):
```
- You extract metadata from scanned photo album pages.
- Return only valid JSON matching the response_format schema.
- Return one entry in `photos` for each photograph visible on the page.
- `photo_number`: The number shown in the bounding box label on the image for this photo. Use this to identify which photo you are describing.
- `location`: Nominatim-queryable location string for the photo. Always include a country name. If the country is not visible, infer it from the album title.
- `location_name`: Famous or well-known location name only if it can be confidently identified from visible evidence. Always include a country name, and city if possible. Use the country name from the album title if unsure.
- `est_date`: Estimated date or date range the photo was taken. First use explicit date evidence visible on the album page near the photo, in page titles, captions, handwritten notes, or scene text, preserving the best supported precision (`1988 Aug.` -> `1988-08`). If the page gives month/day but no year, use the album title as the year. If no date is visible on the page, use the album title year only. Do not infer a different year from clothing, vehicles, or visual style when an album title year is available.
- `scene_ocr`: Any text visible within the photo itself (signs, labels, banners). Empty string if none.
- `caption`: The handwritten or typed caption text written by the album owner near this photo on the album page (beneath, beside, or above it). Copy it verbatim — keep the original wording, casing, and punctuation. **Always populate this field whenever any caption text exists on the page near a photo, even if that text also describes a location and was used to derive `location` or `location_name`.** A single caption may apply to several adjacent photos; in that case repeat the same caption text in each of those photos' entries. Empty string only when there is no caption text written near this photo at all.
- `corrected_caption`: Correct obvious spelling errors in `caption` only when the caption appears to be a place/location name and the corrected spelling would make the place queryable. Keep this empty when there is no confident correction. Do not rewrite normal prose captions.
- `people_count`: Number of clearly visible people in the photo.
```

**User Prompt** (`user.md`):
```
Analyze this album page.
Album title: {album_title}
```

**Inference Parameters (Embedded):**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_tokens | 2048 | Maximum length of model response |
| temperature | 0.1 | Low temperature for deterministic output (0.1 = very focused) |
| max_image_edge | 0 | Image size limit (0 = no limit) |
| timeout_seconds | 300.0 | API timeout for inference (5 minutes) |

**Output Schema** (`schema.json`):
```json
{
  "type": "object",
  "properties": {
    "photos": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "photo_number": {"type": "integer"},
          "location": {"type": "string"},
          "location_name": {"type": "string"},
          "est_date": {"type": "string"},
          "scene_ocr": {"type": "string"},
          "caption": {"type": "string"},
          "corrected_caption": {"type": "string"},
          "people_count": {"type": "integer"}
        }
      }
    }
  }
}
```

#### 5.3.2 People Count (Per-Crop Refinement)
For each detected crop, optionally run a separate people-counting request to refine estimates.

**Baseline:** same lmstudio server and `google/gemma-4-31b` model as 5.3.1.

**Prompt Source:** `photoalbums/prompts/ai-index/people-count/` (`system.md`, `user.md`, `output.md`, `params.toml`)

**System Prompt** (`system.md`):
```
- You count visible people in photographs.
- Return only valid JSON matching the response_format schema.
- Count clearly visible real people only.
- Do not include reasoning or extra fields.
```

**User Prompt** (`user.md`):
```
- Count the number of clearly visible real people.
```

**Output Format** (`output.md`):
```json
{"count": 0}
```

**Inference Parameters (Embedded):**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_tokens | 48 | Very short response (just a count number) |
| temperature | 0.0 | Maximum determinism (no sampling, greedy decoding) |
| max_image_edge | 0 | Image size limit (0 = no limit) |
| timeout_seconds | 300.0 | API timeout for inference (5 minutes) |

**Output Schema:** JSON with single field:
```json
{"count": 0}
```

#### 5.3.3 People Matching (Optional)
Use Cast service (external face matching system) to match detected faces against roster.

**Inputs:**
- Detected faces from YOLO (optional)
- People roster from album set configuration
- Cast service embeddings

**Output:** Person name + confidence score per detection

#### 5.3.4 Object Detection (Optional)
Use YOLO to detect visual objects in page/crop images.

**Output:** Object class labels + bounding boxes + confidence scores

#### 5.3.5 GPS Location Estimation
Estimate GPS coordinates from image content and page context.

**Inputs:**
- Page/crop image content
- Page caption/description from OCR
- Album context (collection, year, page number)
- People names (matched or manually entered)

**Output:** Latitude/longitude coordinates with confidence metadata

#### 5.3.6 Geocoding (Nominatim Reverse Lookup)
Convert GPS coordinates to human-readable location names.

**Service:** Nominatim OpenStreetMap API
**Base URL:** `https://nominatim.openstreetmap.org`
**User Agent:** `imago-photoalbums-ai-index/1.0`
**Timeout:** 20.0 seconds per request
**Min Interval:** 1.0 second between requests (rate limiting)
**Cache:** `{PHOTOALBUMS_DIR}/data/geocode_cache.json`

**Query Structure:**
- Input: lat, lon, zoom (optional)
- Output: JSON with `address` dict containing: `country`, `state`, `city`, `town`, `village`, `road`, `postcode`, etc.

**Fallback Behavior:**
- Network timeout → retry with exponential backoff (up to 3 attempts)
- No result found → store empty location, continue processing
- Rate limited (HTTP 429) → pause and retry
- Connection refused → skip geocoding, store estimated coords only

### 5.4 Crop-Level Processing

Individual cropped photos inherit page-level metadata (caption, date, location) and may be refined with:
- Crop-specific people counting
- Crop-specific location grounding
- Crop XMP verification (see Section 7)

---

## 6. XMP Sidecar Metadata Schema

### 6.1 File Structure
**Naming:** `{SourceImageName}.xmp` (parallel to JPEG/TIFF)
**Format:** XML with RDF namespaces and custom schema

### 6.2 Namespace Declarations

The sidecar declares the following prefixes. Standard interop prefixes (dc, exif, xmpMM, etc.) carry data that downstream tools (Bridge, Lightroom, exiftool) can read; `imago:` and the `stArea`/`stEvt`/`stRef` structured-type namespaces carry pipeline-specific state.

| Prefix | Namespace URI | Purpose |
|--------|---------------|---------|
| `x` | `adobe:ns:meta/` | XMP packet container |
| `rdf` | `http://www.w3.org/1999/02/22-rdf-syntax-ns#` | RDF/XML structure |
| `dc` | `http://purl.org/dc/elements/1.1/` | Dublin Core |
| `xmp` | `http://ns.adobe.com/xap/1.0/` | XMP core |
| `xmpMM` | `http://ns.adobe.com/xap/1.0/mm/` | Media management / provenance |
| `xmpDM` | `http://ns.adobe.com/xmp/1.0/DynamicMedia/` | Dynamic media (rarely written) |
| `exif` | `http://ns.adobe.com/exif/1.0/` | EXIF (dates, GPS, dimensions) |
| `Iptc4xmpExt` | `http://iptc.org/std/Iptc4xmpExt/2008-02-29/` | IPTC extension (people, location) |
| `crs` | `http://ns.adobe.com/camera-raw-settings/1.0/` | Camera Raw (rarely written) |
| `mwg-rs` | `http://www.metadataworkinggroup.com/schemas/regions/` | MWG region list |
| `stArea` | `http://ns.adobe.com/xap/1.0/sType/Area#` | Region coordinate attributes |
| `stEvt` | `http://ns.adobe.com/xap/1.0/sType/ResourceEvent#` | xmpMM:History events |
| `stRef` | `http://ns.adobe.com/xap/1.0/sType/ResourceRef#` | xmpMM:DerivedFrom references |
| `imago` | `https://imago.local/ns/1.0/` | Custom imago fields and JSON detections |

```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
      xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
      xmlns:stArea="http://ns.adobe.com/xap/1.0/sType/Area#"
      xmlns:stEvt="http://ns.adobe.com/xap/1.0/sType/ResourceEvent#"
      xmlns:stRef="http://ns.adobe.com/xap/1.0/sType/ResourceRef#"
      xmlns:imago="https://imago.local/ns/1.0/">
```

### 6.3 Dublin Core (dc:) Metadata

Note: `dc:creator` is **not** written. People are recorded under `Iptc4xmpExt:PersonInImage` (Section 6.5).

```xml
<dc:title>
  <rdf:Alt>
    <rdf:li xml:lang="x-default">{title}</rdf:li>
  </rdf:Alt>
</dc:title>

<dc:description>
  <rdf:Alt>
    <rdf:li xml:lang="x-default">{description}</rdf:li>
  </rdf:Alt>
</dc:description>

<dc:date>
  <rdf:Seq>
    <rdf:li>{YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD or YYYY-MM or YYYY}</rdf:li>
  </rdf:Seq>
</dc:date>

<dc:subject>
  <rdf:Bag>
    <rdf:li>{keyword}</rdf:li>
  </rdf:Bag>
</dc:subject>

<dc:source>{source_page_filename}</dc:source>
```

### 6.4 EXIF Metadata
```xml
<exif:DateTimeOriginal>{ISO8601 datetime}</exif:DateTimeOriginal>
<exif:GPSLatitude>{lat in DMS or signed-decimal string}</exif:GPSLatitude>
<exif:GPSLongitude>{lon in DMS or signed-decimal string}</exif:GPSLongitude>
<exif:GPSMapDatum>WGS-84</exif:GPSMapDatum>
<exif:GPSVersionID>2.3.0.0</exif:GPSVersionID>
```

`exif:ImageWidth`/`exif:ImageLength` are not written by the pipeline; consumers should derive dimensions from the JPEG itself.

### 6.5 IPTC-Ext (Iptc4xmpExt:) Metadata

```xml
<Iptc4xmpExt:PersonInImage>
  <rdf:Bag>
    <rdf:li>{person_name}</rdf:li>
  </rdf:Bag>
</Iptc4xmpExt:PersonInImage>

<Iptc4xmpExt:LocationShown>
  <rdf:Bag>
    <rdf:li rdf:parseType="Resource">
      <Iptc4xmpExt:LocationName>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{display_name}</rdf:li>
        </rdf:Alt>
      </Iptc4xmpExt:LocationName>
      <Iptc4xmpExt:WorldRegion>{region}</Iptc4xmpExt:WorldRegion>
      <Iptc4xmpExt:CountryCode>{ISO 3166-1 alpha-2}</Iptc4xmpExt:CountryCode>
      <Iptc4xmpExt:CountryName>{country}</Iptc4xmpExt:CountryName>
      <Iptc4xmpExt:ProvinceState>{state}</Iptc4xmpExt:ProvinceState>
      <Iptc4xmpExt:City>{city}</Iptc4xmpExt:City>
      <Iptc4xmpExt:Sublocation>{sublocation}</Iptc4xmpExt:Sublocation>
      <exif:GPSLatitude>{lat}</exif:GPSLatitude>
      <exif:GPSLongitude>{lon}</exif:GPSLongitude>
    </rdf:li>
  </rdf:Bag>
</Iptc4xmpExt:LocationShown>

<Iptc4xmpExt:Sublocation>{sublocation}</Iptc4xmpExt:Sublocation>
<Iptc4xmpExt:LocationCreated>{formatted location string}</Iptc4xmpExt:LocationCreated>
```

### 6.6 Imago Custom Schema (imago:)
**Namespace:** `https://imago.local/ns/1.0/`

The imago schema has two parts:

1. **Discrete XML elements** for editorial / OCR text that benefits from staying queryable as XML.
2. **A single `imago:Detections` JSON blob** for evolving operational state (people, objects, captions, location, processing/pipeline). This avoids RDF schema churn as new pipeline steps are added.

#### 6.6.1 Discrete imago: Elements

```xml
<imago:AlbumTitle>{album_title}</imago:AlbumTitle>
<imago:OCRText>{full_page_ocr}</imago:OCRText>
<imago:ParentOCRText>{ocr_from_parent_page}</imago:ParentOCRText>
<imago:OCRLang>{language_code}</imago:OCRLang>
<imago:AuthorText>{handwritten_author_annotations}</imago:AuthorText>
<imago:SceneText>{text_visible_in_photo_scene}</imago:SceneText>
<imago:TitleSource>{origin_of_dc:title}</imago:TitleSource>
<imago:OCRAuthoritySource>{authoritative_ocr_provider}</imago:OCRAuthoritySource>
```

#### 6.6.2 imago:Detections JSON

A single text element whose value is a JSON object:

```xml
<imago:Detections>{...JSON object...}</imago:Detections>
```

JSON top-level keys (each optional):

| Key | Type | Contents |
|-----|------|----------|
| `people` | array | People records (name, confidence, source, bbox, etc.) |
| `objects` | array | Object detections (class, confidence, bbox) from YOLO |
| `caption` | object | AI caption record (text, source model, confidence) |
| `location` | object | Resolved location: `{city, state, country, sublocation, ...}` |
| `processing` | object | Per-stage flags such as `people_detected`, `people_identified`, `ocr_ran` |
| `pipeline` | object | Step records keyed by step name (see below) |

**Pipeline step records** live under `pipeline.{step_name}` and contain stage-specific keys. Common keys observed:

| Key | Meaning |
|-----|---------|
| `timestamp` | ISO-8601 when the step completed |
| `result` | Status string (e.g., `"ok"`, `"no_regions"`, `"failed"`) |
| `input_hash` | Hash of inputs used to detect re-run requirement |
| `artifact_path` | Path to a debug or output artifact, if any |
| Stage-specific fields | e.g., `concerns`, `human_inference`, `needs_another_pass`, `needs_human_review` for `verify_crops` |

There is **no** `imago:ProcessingStatus` XML attribute, no `imago:PipelineSteps` rdf:Seq, and no separate `imago:PersonCount`/`imago:PersonInImage`/`imago:ObjectDetections`/`imago:LocationPayload`/`imago:LocationReverse` elements. All of that data lives inside `imago:Detections`.

### 6.7 MWG-RS Region Metadata (mwg-rs:)
**Use Case:** Photo region bounding boxes on a page view sidecar
**Schema:** MWG (Metadata Working Group) regions, using Adobe `stArea:` structured-type attributes

Coordinates are written as **attributes** on the `rdf:li` element using the `stArea:` namespace, **not** as nested `mwg-rs:x` / `mwg-rs:y` child elements. `mwg-rs:Type` and `mwg-rs:Name` are also attributes. Per-region imago-namespace fields (`imago:PhotoNumber`, `imago:CaptionHint`) are likewise attributes; structured location data and person lists are written as child elements.

```xml
<mwg-rs:Regions rdf:parseType="Resource">
  <mwg-rs:RegionList>
    <rdf:Bag>
      <rdf:li rdf:parseType="Resource"
              mwg-rs:Type="Photo"
              mwg-rs:Name="{caption_text_from_gemma4_metadata}"
              stArea:x="{cx}"
              stArea:y="{cy}"
              stArea:w="{width}"
              stArea:h="{height}"
              stArea:unit="normalized"
              imago:PhotoNumber="{1-based_index}"
              imago:CaptionHint="{docling_caption_hint}">
        <imago:PersonNames>
          <rdf:Bag>
            <rdf:li>{person_name}</rdf:li>
          </rdf:Bag>
        </imago:PersonNames>
        <imago:LocationAssigned rdf:parseType="Resource">
          <!-- city, state, country, GPS, etc. -->
        </imago:LocationAssigned>
        <imago:LocationOverride rdf:parseType="Resource">
          <!-- manual override of LocationAssigned -->
        </imago:LocationOverride>
      </rdf:li>
    </rdf:Bag>
  </mwg-rs:RegionList>
</mwg-rs:Regions>
```

- `stArea:x`, `stArea:y`: normalized **center-point** coordinates (0.0–1.0)
- `stArea:w`, `stArea:h`: normalized width and height (0.0–1.0)
- `stArea:unit`: always `"normalized"`
- Coordinates are formatted with six decimal places (e.g., `0.500000`)

### 6.8 XMP Standard Fields (xmp:)
```xml
<xmp:CreatorTool>imago-photoalbums</xmp:CreatorTool>
<xmp:CreateDate>{ISO8601_datetime}</xmp:CreateDate>
<xmp:MetadataDate>{ISO8601_datetime}</xmp:MetadataDate>
<xmp:ModifyDate>{ISO8601_datetime}</xmp:ModifyDate>
```

### 6.9 XMP Media Management (xmpMM:)

Tracks document identity and provenance. `xmpMM:DerivedFrom` uses `stRef:` structured type to link to source files. `xmpMM:Pantry` is an rdf:Bag of cross-pipeline tracking entries.

```xml
<xmpMM:DocumentID>uuid:{unique-id}</xmpMM:DocumentID>
<xmpMM:DerivedFrom rdf:parseType="Resource">
  <stRef:documentID>uuid:{source-document-id}</stRef:documentID>
  <stRef:filePath>{optional source path}</stRef:filePath>
</xmpMM:DerivedFrom>
<xmpMM:Pantry>
  <rdf:Bag>
    <rdf:li rdf:parseType="Resource">
      <xmpMM:DocumentID>uuid:{tracked-id}</xmpMM:DocumentID>
    </rdf:li>
  </rdf:Bag>
</xmpMM:Pantry>
```

### 6.10 Date Format Normalization
- **Dates on file:** ISO 8601 preferred (e.g., `2024-05-08T14:30:00Z`)
- **Partial dates:** `YYYY`, `YYYY-MM`, `YYYY-MM-DD` accepted
- **EXIF format:** `YYYY:MM:DD HH:MM:SS` (colon separator)
- **Parsing:** Normalize all formats to canonical ISO 8601 + remove microseconds
- **Timezone:** All times in UTC (Z suffix)

### 6.11 Example Page View XMP
```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:imago="https://imago.local/ns/1.0/"
      xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
      xmlns:stArea="http://ns.adobe.com/xap/1.0/sType/Area#">

      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Egypt 1975, Book B, Page 05</rdf:li>
        </rdf:Alt>
      </dc:title>

      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Vacation photos from Cairo. Two photos: 1. Temple entrance with tourists. 2. Local market scene.</rdf:li>
        </rdf:Alt>
      </dc:description>

      <dc:date>
        <rdf:Seq>
          <rdf:li>1975-08</rdf:li>
        </rdf:Seq>
      </dc:date>

      <imago:AlbumTitle>Egypt 1975</imago:AlbumTitle>
      <imago:OCRText>Cairo temple 1975 ...</imago:OCRText>

      <mwg-rs:Regions rdf:parseType="Resource">
        <mwg-rs:RegionList>
          <rdf:Bag>
            <rdf:li rdf:parseType="Resource"
                    mwg-rs:Type="Photo"
                    mwg-rs:Name="Cairo temple 1975"
                    stArea:x="0.250000"
                    stArea:y="0.300000"
                    stArea:w="0.350000"
                    stArea:h="0.400000"
                    stArea:unit="normalized"
                    imago:PhotoNumber="1"
                    imago:CaptionHint="Cairo temple"/>
          </rdf:Bag>
        </mwg-rs:RegionList>
      </mwg-rs:Regions>

      <imago:Detections>{
        "processing": {"people_detected": true, "ocr_ran": true},
        "pipeline": {
          "view_regions": {
            "timestamp": "2024-05-08T14:30:00Z",
            "result": "ok",
            "input_hash": "sha256:..."
          }
        },
        "caption": {"text": "Cairo temple 1975", "source": "google/gemma-4-31b"}
      }</imago:Detections>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

### 6.12 Example Crop XMP
```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
      xmlns:imago="https://imago.local/ns/1.0/">

      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Temple entrance with tourists</rdf:li>
        </rdf:Alt>
      </dc:title>

      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Temple entrance. Handwritten caption: "Cairo temple 1975"</rdf:li>
        </rdf:Alt>
      </dc:description>

      <dc:date>
        <rdf:Seq>
          <rdf:li>1975-08</rdf:li>
        </rdf:Seq>
      </dc:date>

      <dc:source>Egypt_1975_B01_P05_V.jpg</dc:source>

      <exif:DateTimeOriginal>1975-08-01T00:00:00</exif:DateTimeOriginal>
      <exif:GPSLatitude>30,1,43.68N</exif:GPSLatitude>
      <exif:GPSLongitude>31,14,58.20E</exif:GPSLongitude>
      <exif:GPSMapDatum>WGS-84</exif:GPSMapDatum>
      <exif:GPSVersionID>2.3.0.0</exif:GPSVersionID>

      <Iptc4xmpExt:PersonInImage>
        <rdf:Bag>
          <rdf:li>Jane Doe</rdf:li>
          <rdf:li>John Doe</rdf:li>
        </rdf:Bag>
      </Iptc4xmpExt:PersonInImage>

      <imago:OCRText>Cairo temple 1975</imago:OCRText>
      <imago:AuthorText>Cairo temple 1975</imago:AuthorText>

      <imago:Detections>{
        "people": [
          {"name": "Jane Doe", "confidence": 0.91, "source": "cast"},
          {"name": "John Doe", "confidence": 0.88, "source": "cast"}
        ],
        "objects": [
          {"class": "person", "confidence": 0.97}
        ],
        "caption": {"text": "Temple entrance with tourists", "source": "google/gemma-4-31b"},
        "location": {
          "city": "Cairo", "country": "Egypt",
          "gps_latitude": 30.0288, "gps_longitude": 31.2495,
          "source": "ai_caption"
        },
        "processing": {"people_detected": true, "people_identified": true, "ocr_ran": true},
        "pipeline": {
          "crop":        {"timestamp": "2024-05-08T14:35:00Z", "result": "ok",       "input_hash": "sha256:..."},
          "restoration": {"timestamp": "2024-05-08T14:36:00Z", "result": "ok",       "input_hash": "sha256:..."},
          "verify_crops":{"timestamp": "2024-05-08T14:40:00Z", "result": "ok",
                          "concerns": {
                            "caption":        {"verdict": "good", "reasoning": "matches handwritten note"},
                            "gps":            {"verdict": "good", "reasoning": "Nominatim lookup matches"},
                            "shown_location": {"verdict": "good", "reasoning": "landmark visible"},
                            "date":           {"verdict": "good", "reasoning": "album title year"},
                            "overall":        {"verdict": "good", "reasoning": "all concerns clear"}
                          },
                          "needs_another_pass": [],
                          "needs_human_review": []}
        }
      }</imago:Detections>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

---

## 7. Crop Verification & Validation

### 7.1 Purpose
Validate crop metadata (caption, date, location) against page context before finalizing.

### 7.2 Entry Point
Verify crops after region detection and AI processing.

### 7.3 Verification Prompt

**System Prompt:**
```
- You verify already-generated crop metadata against family photo album page context.
- Return only valid JSON matching the response_format schema.
- Treat the crop image as the primary review target and the page image as supporting context.
- Use the supplied page XMP text and crop XMP text as evidence, not as unquestioned truth.
- Mark a concern `good` only when the metadata is supported by what a careful human would infer from the page.
- Mark a concern `bad` when the metadata conflicts with the page context.
- Mark a concern `uncertain` when the page context does not support a confident human judgment.
- Keep each concern `reasoning` to one sentence.
- Keep each concern `failure_reason` empty when verdict is `good`.
- When any concern is `bad` or `uncertain`, include `human_inference` describing what a person would actually infer from the page.
- `needs_another_pass` and `needs_human_review` must use readable concern names from: `caption`, `gps`, `shown_location`, `date`.
- `overall` is summary only and must not appear in retry-routing arrays.
```

**User Prompt:**
```
- Review target: one crop image plus its crop XMP metadata, using the full album page image and page XMP metadata as supporting context.
- Judge whether the crop caption belongs to that crop on the page, including nearby-caption carry-over when a human would read neighboring captions together.
- Judge whether shown location matches the crop image, `dc:description`, or visible landmarks.
- Judge whether the GPS coordinates are generally consistent with the crop image and `dc:description`.
- For GPS and shown-location concerns, use the supplied location verification evidence: GPS coordinates plus Nominatim reverse-lookup name. OCR text is intentionally not supplied for location verification.
- If the reverse-lookup name and GPS coordinates align with what is visible in the image or stated in `dc:description`, consider the location grounded and mark it as good.
- Judge whether date preserves the best supported precision from the page, including month-plus-year evidence such as `AUG. 1988` -> `1988-08` that is non-standard but human-readable. If no date is visible in the page context, accept the album title year as the correct fallback year. Do not mark an album-title-year-only date as uncertain merely because the page itself lacks a visible date.
- If page image context is missing, do not pretend a full review happened.
- If page XMP text context is missing, do not pretend a full metadata-context review happened.
- Base your verdicts on what a human flipping through the album would accept as belonging together.
```

### 7.4 Verification Concerns

The pipeline tracks five concerns, but only four are routable for retry:

- `VERIFICATION_CONCERNS = ("caption", "gps", "shown_location", "date", "overall")`
- `ROUTABLE_CONCERNS = ("caption", "gps", "shown_location", "date")` — eligible to appear in `needs_another_pass` / `needs_human_review`

`overall` is a summary-only verdict and never appears in retry-routing arrays.

**Verdict Values:** `good`, `bad`, `uncertain`
- `good`: Metadata supported by page context
- `bad`: Metadata conflicts with page context
- `uncertain`: Insufficient page context for confident judgment

**Inference Parameters (defaults):**
| Parameter | Value |
|-----------|-------|
| max_tokens | 512 |
| temperature | 0.0 |
| max_image_edge | from settings |

**Per-concern record:** `{verdict, reasoning, failure_reason}` plus top-level `human_inference`, `needs_another_pass`, `needs_human_review`.

### 7.5 Retry & Human Review Routing
If verification finds concerns:
- **Needs Another Pass:** Re-run AI processing with adjusted prompts or models (specific failing fields)
- **Needs Human Review:** Metadata cannot be auto-corrected; flag for manual inspection

---

## 8. Location Resolution & Fallbacks

### 8.1 Location Payload Structure
Internal representation of resolved location metadata. Fields are flat at the top level (no `inferred_location` / `confidence` / `explicit_gps` wrappers); the raw Nominatim response, when present, lives under the `nominatim` key:

```json
{
  "query": "Cairo, Egypt",
  "display_name": "Cairo, Egypt",
  "gps_latitude": 30.0288,
  "gps_longitude": 31.2495,
  "map_datum": "WGS-84",
  "source": "ai_caption",
  "city": "Cairo",
  "state": "Cairo Governorate",
  "country": "Egypt",
  "sublocation": "Temple district",
  "nominatim": {
    "lat": "30.0288",
    "lon": "31.2495",
    "display_name": "...",
    "address": { "country": "...", "city": "...", "...": "..." }
  }
}
```

Optional sub-keys (`city`, `state`, `country`, `sublocation`, `nominatim`) are omitted when not available.

### 8.2 Location Extraction
**Input:** AI-generated location string from page OCR
**Query:** Pass string to Nominatim for coordinate resolution

**Nominatim Request Parameters:**
- `q`: location query string (e.g., "Cairo, Egypt")
- `format`: "json"
- `limit`: 1 (top result only)
- `timeout`: 20.0 seconds
- `User-Agent`: "imago-photoalbums-ai-index/1.0"

### 8.3 Reverse Geocoding
**Input:** GPS coordinates from AI estimation
**Query:** Nominatim reverse lookup for human-readable name

**Request Parameters:**
- `lat`, `lon`: coordinates
- `format`: "json"
- `zoom`: `18` (building / address-level detail; the most precise zoom Nominatim accepts)
- `timeout`: 20.0 seconds

### 8.4 Cache Management
**File:** `{PHOTOALBUMS_DIR}/data/geocode_cache.json`
**Format:** JSON dict mapping query string → result
**Behavior:**
- Check cache before network request
- Store successful results
- Skip failed requests (timeout, 404, rate limit)

### 8.5 Rate Limiting
- **Min Interval:** 1.0 second between Nominatim requests, enforced by a simple `_throttle()` sleep before each request
- **Backoff:** None. The current implementation does **not** retry with exponential backoff—a network or HTTP failure raises a `RuntimeError` immediately
- **Max Retries:** 0 (single attempt per query)

### 8.6 Fallback Behavior by Scenario

**Scenario: Nominatim unavailable (network error)**
- The geocoder raises `RuntimeError` on the first failure
- Calling code is responsible for catching it; the resolved location is left unfilled and processing continues with whatever string evidence is available

**Scenario: Location string not found**
- Return empty/null coordinates
- Preserve original location string
- Continue processing

**Scenario: Rate limited (HTTP 429)**
- Treated as a request failure; raised to the caller (no automatic backoff/retry)

**Scenario: Invalid coordinates**
- Skip reverse lookup
- Preserve original coordinates in XMP

The geocode cache (Section 8.4) absorbs most repeat traffic, so repeated failures typically don't compound across runs.

---

## 9. Pipeline State Tracking

Pipeline state is **not** modeled as discrete RDF elements. Both the per-stage flags and the per-step records are stored as JSON inside the single `imago:Detections` element (see Section 6.6.2). There is no `imago:ProcessingStatus` attribute and no `imago:PipelineSteps` rdf:Seq.

### 9.1 Per-Stage Flags
Stored under `imago:Detections.processing` as boolean flags. Stage names observed in code include:
- `people_detected`
- `people_identified`
- `ocr_ran`
- (other stage-specific keys added as new pipeline steps come online)

### 9.2 Pipeline Step Records
Stored under `imago:Detections.pipeline.{step_name}` as a JSON object per step. Common step names: `view_regions`, `crop`, `restoration`, `ai_caption`, `ai_metadata`, `verify_crops`, `geocode`.

**Common per-step keys:**
| Key | Meaning |
|-----|---------|
| `timestamp` | ISO 8601 when step completed |
| `result` | Outcome string (e.g., `"ok"`, `"no_regions"`, `"validation_failed"`, `"failed"`) |
| `input_hash` | Hash of inputs used to detect re-run requirement |
| `artifact_path` | Path to a debug/output artifact, if produced |

**Step-specific extras (examples):**
- `verify_crops` adds: `concerns` (per-concern dict of `{verdict, reasoning, failure_reason, provenance}`), `human_inference`, `needs_another_pass`, `needs_human_review`, `page_verification_ran`

### 9.3 Skip Existing Logic
When determining whether to reprocess a stage:
1. Read the sidecar and parse `imago:Detections` JSON
2. Look up `pipeline.{step_name}`
3. If the stored `input_hash` matches the current inputs and `result` is a terminal state (e.g., `ok`, `no_regions`, `validation_failed`, `failed`), skip
4. If the result claims success but downstream metadata is missing, force reprocess
5. If no record exists, run the step

---

## 10. Storing the Specs (Reference File Layouts)

Section 1.4 describes *what* the system needs. This section shows one concrete way to persist those specs on disk so the pipeline can read them; a reimplementation may use any equivalent storage. Filenames here are descriptive, not required.

### 10.1 Album-Set Spec (TOML example)
```toml
[archive_set_name]
photos_root = "/path/to/Photo Albums"
people_roster_path = "people.csv"
```

Maps an archive-set identifier to its `{PHOTOS_ROOT}` directory and its people roster.

### 10.2 AI-Models Spec (TOML example)
```toml
[archive_set_name.docling]
preset  = "granite_docling"
backend = "auto_inline"
device  = "auto"
retries = 3
lmstudio_base_url = "http://127.0.0.1:1234/v1"

[archive_set_name.lmstudio]
primary = ["google/gemma-4-31b"]
pc      = ["google/gemma-4-31b"]

[archive_set_name.restoration]
enabled    = true
model_name = "RealRestorer/RealRestorer"
```

Per-archive overrides for the AI baseline. Values shown here are the defaults the current codebase ships with.

### 10.3 Render Spec (JSON example)
```json
{
  "archive_settings": {
    "render_scale": 1.0,
    "jpeg_quality": 95
  },
  "chapter_settings": {
    "Egypt_1975_B01": {
      "stitch_detector": "sift"
    }
  }
}
```

Per-album rendering overrides. With no override, stitching uses the full `AFFINE_STITCH_ATTEMPTS` sequence (Section 2.4).

---

## 11. Naming & Identification

### 11.1 File Naming Patterns

**Album Directory Structure:**
All album content is organized in three sibling directories with the same base name:
- `{Base}_Archive`: Raw TIFF scans
- `{Base}_Pages`: Stitched page JPEGs
- `{Base}_Photos`: Cropped individual photos

Where Base = `{Collection}_{Year}_B{Book}`

**Complete Naming Pattern:**

| File Type | Pattern | Example |
|-----------|---------|---------|
| **TIFF Scan (raw)** | `{Collection}_{Year}_B{Book}_P{Page:02d}_S{Scan:02d}.tif` | `Egypt_1975_B01_P05_S01.tif` |
| **Page View (stitched)** | `{Collection}_{Year}_B{Book}_P{Page:02d}_V.jpg` | `Egypt_1975_B01_P05_V.jpg` |
| **Derived photo view** | `{Collection}_{Year}_B{Book}_P{Page:02d}_D{Derived:02d}-{Iter:02d}_V.jpg` | `Egypt_1975_B01_P05_D00-00_V.jpg` |
| **XMP Sidecar** | `{source_filename}.xmp` | `Egypt_1975_B01_P05_V.xmp` |

**Type Tokens** — control what stage and format a file represents:

| Token | Meaning | Storage | Notes |
|-------|---------|---------|-------|
| `_S##` | Raw scan | `.tif` or `.png` | Original input; never processed |
| `_D##-##` | Derived image | `.tif` | Extracted crop or intermediate; not a final view |
| `_V` | View (rendered) | `.jpg` | Always marks display-ready output; page views and crop views end in `_V.jpg` |
| `_D##-##_V` | View of derived | `.jpg` | Extracted crop rendered as JPEG |

**Field Definitions:**
- **Collection:** Alphanumeric string (no underscores), e.g., "Egypt", "Cordell", "Hawaii"
- **Year:** 4-digit year or year range, e.g., "1975" or "1975-1976"
- **Book:** 2-digit book number (00-99)
  - Valid: `00` through `99`
- **Page:** 2-digit page number within book in range `01` through `99` (P00 is not valid). Leading zero required, e.g., "05"
- **Scan:** 2-digit scan index per page, e.g., "01", "02" (multiple scans per page due to scanner width limit)
- **Derived:** 2-digit per-page index of a derived photo extracted from the page, e.g., "00", "01"
- **Iter:** 2-digit version of the derived (`D##`) image, e.g., "00" (first version); increments when the derived photo is re-derived (recropped, regeometried) from the same source page region

### 11.2 Validation & Parsing Rules

**TIFF Scan File Validation:**
- Must match: `{Collection}_{Year}_B{Book}_P{Page:02d}_S{Scan:02d}.tif`
- Collection: any alphanumeric characters except underscore
- Year: exactly 4 digits, or 4 digits + hyphen + 4 digits
- Book: exactly 2 digits (00-99)
- Page: exactly 2 decimal digits in range 01-99 (P00 is invalid)
- Scan: exactly 2 decimal digits
- Extension: `.tif`
- Valid example: `Egypt_1975_B01_P05_S01.tif` ✓
- Invalid examples: `Egypt_1975_B01_P5_S1.tif` ✗ (no leading zeros), `Egypt_1975_B01_P00_S01.tif` ✗ (P00 not allowed)

**Page View File Validation:**
- Must end with: `_P{Page:02d}_V.jpg`
- Capture base album identifier: `{Collection}_{Year}_B{Book}`
- Page: exactly 2 decimal digits in range 01-99 (P00 is invalid)
- Type token: literal `_V` (view)
- Extension: `.jpg`
- Valid example: `Egypt_1975_B01_P05_V.jpg` ✓
- Invalid example: `Egypt_1975_B01_P00_V.jpg` ✗ (P00 not allowed)

**Derived Photo File Validation:**
- Must match: `{Collection}_{Year}_B{Book}_P{Page:02d}_D{Derived:02d}-{Iter:02d}_V.jpg`
- Page: exactly 2 decimal digits in range 01-99 (P00 is invalid)
- Derived index: exactly 2 decimal digits (00-99)
- Iter index: exactly 2 decimal digits (00-99)
- Type token: literal `_V` (view)
- Extension: `.jpg`
- Valid example: `Egypt_1975_B01_P05_D00-00_V.jpg` ✓
- Invalid examples: `Egypt_1975_B01_P05_D0-0_V.jpg` ✗ (single digit), `Egypt_1975_B01_P00_D00-00_V.jpg` ✗ (P00 not allowed)

**XMP Sidecar Validation:**
- Must be parallel to image file with `.xmp` extension
- Examples:
  - `Egypt_1975_B01_P05_V.jpg` → `Egypt_1975_B01_P05_V.xmp`
  - `Egypt_1975_B01_P05_D00-00_V.jpg` → `Egypt_1975_B01_P05_D00-00_V.xmp`

---

## 12. Error Handling & Recovery

### 12.1 Validation Failures
**Region validation fails:**
- Log region as invalid (zero area, full page, etc.)
- Continue with remaining regions
- Mark page as "validation_failed" in pipeline state

**XMP parsing fails:**
- Log error with path
- Retry parse (up to 2 times)
- If persists: skip image, log to error list

**Image reading fails (Pillow):**
- Try ImageMagick fallback
- If both fail: skip image

### 12.2 Network Failures

**Nominatim failure (timeout, HTTP error, 429):**
- Baseline behavior: the geocoder raises and the calling step records the failure; no automatic retry/backoff (see Section 8.5)
- The cached entry, if any, is still served on later runs

**lmstudio (Gemma-4) failure:**
- Log error with the request payload identifier
- Mark the AI step's pipeline record with a non-`ok` `result`
- Continue with remaining images; downstream steps that depend on the missing metadata are skipped

A reimplementation that swaps the AI provider (e.g., a hosted vision API) is free to add retry policies — but the baseline does not.

### 12.3 Processing Locks
To prevent concurrent processing of same image:
- Acquire lock file before processing: `{image_path}.processing`
- Release lock on completion (success or failure)
- If lock exists and stale (>1 hour old), allow reacquisition

---

## 13. Output Artifacts

### 13.1 Directory Structure After Full Processing

```
{PHOTOS_ROOT}/
├── {Album}_Archive/
│   ├── (original raw TIFFs)
├── {Album}_Pages/
│   ├── {Album}_P01_V.jpg
│   ├── {Album}_P01_V.xmp
│   ├── {Album}_P02_V.jpg
│   ├── {Album}_P02_V.xmp
│   └── ...
└── {Album}_Photos/
    ├── {Album}_P01_D00-00_V.jpg
    ├── {Album}_P01_D00-00_V.xmp
    ├── {Album}_P01_D01-00_V.jpg
    ├── {Album}_P01_D01-00_V.xmp
    ├── {Album}_P02_D00-00_V.jpg
    ├── {Album}_P02_D00-00_V.xmp
    └── ...
```

### 13.2 XMP Sidecar Metadata
Every JPEG (page view or crop) has companion `.xmp` file with:
- OCR text & captions
- Detected faces & objects
- GPS coordinates
- Estimated dates
- People names & confidence
- Processing pipeline history
- Region detection metadata (page views only)

### 13.3 Optional: Debug Artifacts
When `--debug` flag enabled:
- Docling pipeline debug JSON: `{image_path}.view-regions.debug.json`
- AI processing logs: stdout/stderr redirection
- Verification results: `{image_path}.verify-crops.debug.json`

---

## 14. Maintaining the Baseline Snapshot

This document is a **labeled snapshot of a working implementation**, not a forward-looking design. A reimplementation is free to improve on the baseline, but the values recorded here must continue to describe what the current code actually does so the working example is never lost.

### 14.1 What Counts as Baseline (Update When Code Changes)

The following are load-bearing baseline values. If the code drifts from any of them, update the spec in the same change:

| Category | Examples of baseline values to keep in sync |
|----------|---------------------------------------------|
| Library versions | `opencv-contrib-python`, `pillow`, `stitching`, `docling`, `torch` (Section 15.1) |
| Models | `google/gemma-4-31b`, RealRestorer commit hash, YOLO weights path (Section 5.0) |
| Inference parameters | `max_tokens`, `temperature`, `timeout_seconds` per prompt (Sections 5.3.1, 5.3.2, 7.4) |
| Stitching | `AFFINE_STITCH_ATTEMPTS` ordering, linear-fallback constants (Section 2.4) |
| Docling | `preset`, `backend`, `device`, `retries`, `do_ocr` (Section 3.3) |
| Restoration | Prompt text, `num_inference_steps`, `guidance_scale`, `seed`, `size_level` (Section 4.4) |
| Geocoder | User-Agent, timeout, min interval, reverse-lookup `zoom`, retry policy (Section 8) |
| Format contracts | XMP namespaces, MWG-RS attribute layout, file-naming patterns (Sections 6, 11) |
| Prompt text | Exact `system.md`/`user.md` contents under `photoalbums/prompts/` (Sections 5.3.1, 5.3.2, 7.3) |

### 14.2 What Is Not Baseline (Reimplementation Is Free to Change)

These are accidents of the current implementation; a reimplementation may diverge without updating the baseline section, as long as functional behavior in Sections 1–9 still holds:

- Specific Python library choices (e.g., using `xml.etree.ElementTree` vs. `lxml`)
- Internal helper names, exception class names, and module layout
- Retry policies for network calls (the baseline has none for Nominatim — adding sensible retries is an *improvement*, not a deviation)
- Storage format of the operator-supplied specs (TOML vs. JSON vs. env vars; see Section 10)

### 14.3 Update Checklist

When any code change affects a baseline value above:

1. Update the literal value in the relevant SPEC section.
2. If the change alters re-run semantics (e.g., new `input_hash` inputs), note it in Section 9.
3. Record the date and commit hash of the new baseline at the top of the document.
4. In the commit message, state which baseline value moved and why — this is the audit trail for the snapshot.

---

## 15. Key Dependencies & External Services

### 15.1 Python Libraries & Versions
- `opencv-contrib-python` **4.10.0.84**: Image stitching (AffineStitcher), feature detection (SIFT, AKAZE, BRISK), image manipulation
- `pillow` **12.1.1**: Image I/O (JPEG, TIFF, PNG), EXIF handling, image format conversion
- `stitching` **0.6.1**: High-level affine stitcher wrapper (uses OpenCV backend)
- `docling` **2.88.0**: Document layout analysis and image region detection (photo detection from page layouts)
- `diffusers` (RealRestorer fork): RealRestorer pipeline for photo restoration (from https://github.com/yfyang007/RealRestorer.git)
- `torch` **2.10.0**: Hardware/precision detection for RealRestorer and other ML models

The XMP sidecar reader/writer uses Python's standard-library `xml.etree.ElementTree`; that's an implementation choice and a reimplementation may use any equivalent XML library.

### 15.2 External Services
- **lmstudio (local):** Hosts `google/gemma-4-31b` for OCR/caption/metadata extraction at `http://127.0.0.1:1234/v1`
- **Nominatim (OpenStreetMap):** Forward and reverse geocoding at `https://nominatim.openstreetmap.org`
- **Cast Service:** Face recognition / people matching (internal/custom)
- **YOLO (Ultralytics):** Optional object detection via local `models/yolo11n.pt`

### 15.3 System Tools
- **ImageMagick (`magick` command):** Fallback image reading for problematic formats
- **Git:** Not required at runtime, used for version control only

---

## 16. Performance Characteristics

### 16.1 Processing Time Estimates (per page)
- **Stitching:** 2-10 seconds (varies by detector, image size)
- **Docling region detection:** 1-5 seconds
- **Cropping + restoration:** 0.5-2 seconds per crop
- **AI caption extraction:** 5-20 seconds (network latency dominated)
- **Geocoding:** 1-5 seconds per location query (network latency + cache hits)

### 16.2 Memory Usage
- Page view JPEG in memory: ~50-500 MB (depending on resolution)
- RealRestorer model: ~42 GB loaded (uses CPU offload if needed)
- Docling converter: ~200 MB (cached per session)
- lmstudio host process: governed by the loaded Gemma-4 model (separate process; not counted against the pipeline's RSS)

### 16.3 Disk Space
- Each album page JPEG: ~2-20 MB
- Each crop JPEG: ~0.1-2 MB
- XMP sidecar: ~5-50 KB per image
- Geocoding cache: ~100-500 KB

---

## 17. Testing & Validation

### 17.1 Unit Test Entry Points
- `photoalbums/tests/` directory contains test suites
- Test data in `photoalbums/data/` and `photoalbums/evals/`

### 17.2 Integration Validation
- **Stitch rendering:** Validate output JPEG exists and is readable
- **Docling regions:** Check region count > 0 and bounding boxes within image bounds
- **XMP sidecars:** Validate XML structure and required namespace declarations
- **Crop images:** Check dimensions and format
- **Metadata round-trip:** Read XMP, parse all fields, ensure no data loss

### 17.3 Known Limitations
- **Stitching:** May fail on pages with non-overlapping scans or extreme perspective distortion
- **Docling:** Struggles with ornate page borders or text-heavy layouts
- **Location inference:** Relies on visible landmarks or context; isolated photos may have low confidence
- **Restoration:** Slow on CPU; benefits from CUDA acceleration
- **Nominatim:** Occasional rate limiting; geographical biases in coverage

---

**End of Specification**
