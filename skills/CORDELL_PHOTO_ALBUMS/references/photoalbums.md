# Cordell Photo Albums — Reference Documentation

## Pipeline Overview

Each image is processed by `_run_image_analysis()`, which runs one of two branches depending on engine configuration, then converges into a shared synthesis phase.

**Branch selection:** Use Combined mode when OCR is not overridden and both OCR and caption engines are `local` (local HF). Use Separate mode for any other engine configuration (lmstudio, none, or any mix).

### Combined Mode (single GLM inference for OCR + caption)

1. People detection: match faces against Cast embeddings. No prior context.
2. Object detection: run YOLO on the model-prepared image. Independent of people.
3. OCR + Caption (single inference): extract visible text and generate a one-sentence description simultaneously. Receives people names, object labels, and spatial positions computed from face bounding boxes.
4. People recovery (conditional): if recovery mode is not off, and the first pass finds any evidence of people (faces, matched names, YOLO `person` objects, or caption people-count output), re-run face detection using rembg background removal and a looser IOU threshold (0.30 vs 0.55 normally). If the recovered people list differs from the original, regenerate the caption with the updated names.

### Separate Mode (OCR and caption as independent steps)

1. OCR: extract visible text (or use provided override). No prior context.
2. People detection: match faces using OCR text as an additional hint alongside any metadata hint.
3. Object detection: run YOLO. Independent.
4. Cover page detection (if processing a page scan): determine whether the image is an album cover or title page before captioning.
5. Caption: describe the image using OCR text, people names, object labels, and spatial positions.
6. People recovery: same logic as Combined mode.

### Post-Analysis Synthesis (both branches)

- Extract keywords from OCR text (up to 15, filtered for stopwords).
- Merge object labels and OCR keywords into a subjects list.
- Estimate people count: take the maximum of faces detected, matched names, and YOLO person count.
- If the caption engine is LMStudio: run a separate people-count inference and merge with local estimate using the higher value.
- If the caption engine is LMStudio: extract GPS coordinates from OCR text first (explicit lat/lon patterns), then run a location inference if no explicit GPS was found. Prefer explicit GPS over model-inferred GPS.
- Geocode the resolved location name to GPS coordinates via Nominatim (if location name is non-empty and GPS is not already set).

### Output

Assemble an `ImageAnalysis` object with: image path, people names, object labels, OCR text, OCR keywords, subjects, caption, faces detected count, and a full detection payload (serialized people matches, object scores, OCR metadata, caption metadata, and optional location block).

### Fallback and Skip Rules

- Any step with a missing matcher or detector returns an empty result and continues.
- If the caption engine is set to `none`, captioning and location inference are skipped.
- If OCR override is provided, the OCR step is skipped.
- If caption generation fails, the fallback flag is set and an empty caption is returned; the pipeline continues.
- People recovery can be forced on (`always`), forced off (`off`), or set to `auto` (default). For `auto`, trigger whenever the first pass finds any evidence of people.

---

## Use Cases

Model configuration is loaded from `ai_models.json` (see `selected_ocr_model` and `selected_caption_model`). "Combined engine" means the same model handles both OCR and captioning in one inference. "Separate engine" means OCR and captioning run independently.

**Use Case 1: Combined OCR + Caption**
Trigger: Both OCR and caption engines are `local` AND no OCR text override provided.
Steps: People detection → object detection → single model inference for OCR text + caption.
Output: `Combined` format (`ocr_text`, `caption`, `location_name`).

**Use Case 2: Describe — Caption Only (separate engine)**
Trigger: Caption or OCR engine is not `local` (e.g. lmstudio), or an OCR text override was provided.
Steps: OCR first (text used as hint for people matching) → people detection → object detection → caption inference.
Output: `Describe` format (`caption` only).

**Use Case 3: People Count Estimation (lmstudio engine only)**
Trigger: Called after caption generation to validate or refine the people count. Only runs when caption engine is `lmstudio`.
Steps: Single inference with People Count preamble, all available context passed in.
Output: `People Count` format (`people_present`, `estimated_people_count`).

**Use Case 4: Location Inference (lmstudio engine only)**
Trigger: Called after caption when no explicit GPS found in OCR text. Only runs when caption engine is `lmstudio`.
Steps: Try GPS extraction from OCR patterns first; if not found, run location inference.
Output: `Location` format (`location_name`, `gps_latitude`, `gps_longitude`).

**Use Case 5: Cover Page / Title Page**
Trigger: `looks_like_album_cover()` returns true before the caption step.
Steps: Same as Describe or Combined, with Cover Page preamble prepended.
Output: `Describe` format (caption describes the cover, not a photo scene).

---

## Examples

**Example 1: Family group photo (Combined mode)**
Image: Scanned page with several people outdoors, handwritten names below each photo.
Known people: Leslie, Tommy, Robert (matched by face embeddings).
Expected output: `{"ocr_text": "leslie-tommy-robert", "caption": "Leslie, Tommy, and Robert pose together outdoors on a bench.", "location_name": ""}`

**Example 2: Album cover page**
Image: First page with printed title "BOOK 11 — EUROPE 1962" and a decorative border.
Note: `ocr_text` reproduces "BOOK 11 — EUROPE 1962" exactly; caption may reason it as Book II.
Expected output: `{"caption": "The cover page of a photo album titled \"BOOK 11\" documenting a trip to Europe in 1962.", ...}`

**Example 3: Travel photo with visible location text (Describe + Location, separate engine)**
Image: Street scene with sign reading "Via Roma, Napoli" visible in the background.
Pipeline: OCR extracts "Via Roma, Napoli" → used as location-matching hint → caption inference → location inference (no explicit GPS found).
Expected caption output: `{"caption": "A street scene in Naples, Italy, with a sign for Via Roma visible."}`
Expected location output: `{"location_name": "Naples, Italy", "gps_latitude": "", "gps_longitude": ""}`

**Example 4: Portrait with printed GPS coordinates**
Image: Photo with a data strip reading "Lat: 48.8566 Lon: 2.3522".
Pipeline: OCR extracts data strip → GPS regex matches directly → location inference step skipped.
Expected location output: `{"location_name": "", "gps_latitude": "48.8566", "gps_longitude": "2.3522"}`

---

## Common Issues

**Caption names a person as "a man" / "a woman" instead of using a matched name**
Cause: Face match confidence fell below threshold, or face was not detected (occluded, turned away, cluttered background).
Solution: People recovery (rembg) fires automatically in `auto` mode whenever the first pass finds any evidence of people. If recovery still fails, the person may need a better reference embedding in Cast. Force recovery with `always` mode to bypass the auto evidence check.

**OCR text is empty or garbled on a page with visible handwriting**
Cause: Low-contrast, faded, or angled handwriting; dark or sepia-toned scans reduce OCR accuracy.
Solution: Check scan quality. Consider OCR override to pass corrected text manually. GLM Combined mode generally handles handwriting better than standalone OCR engines.

**Location returns empty despite clear place context**
Cause 1: Interior shots have no place evidence — this is correct behaviour.
Cause 2: LMStudio location inference was not called (only runs for LMStudio engine; GLM Combined does not run a separate location step).
Cause 3: Place name is too obscure or ambiguous to meet the "well-documented" threshold.
Solution: For GLM mode, accept empty location as expected. For LMStudio mode, verify the location inference step ran by checking the payload location block.

**Cover page detected incorrectly (false positive or false negative)**
Cause: `looks_like_album_cover()` uses OCR text patterns and aspect ratio — can misfire on title pages inside an album, or miss covers with minimal text.
Solution: Use the `is_page_scan` flag and page index context when processing. For known cover pages, OCR override or a manual album classification hint can force the correct preamble.

**People count is higher than the number of named people**
Cause: YOLO detected person silhouettes in the background, or LMStudio people-count inference returned a higher estimate than the face matcher found. This is expected — count includes unidentified people.
Solution: If the count seems wrong, check the objects list for false "person" detections (posters, paintings, etc.).
