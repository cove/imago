---
name: CORDELL_PHOTO_ALBUMS
description: >-
  Orchestration skill for AI captioning and indexing of Cordell family photo albums (scanned by
  Audrey Cordell). Use this skill whenever: kicking off or monitoring a photoalbums AI indexing job,
  checking the manifest summary (how many photos are done vs pending), reviewing or diagnosing caption
  quality (cut-off captions, empty captions, wrong people names, missing location metadata), reprocessing
  individual photos, or troubleshooting why captions look wrong. Also contains shared vision model prompt
  sections (rules, output formats, hints) used by the GLM captioning pipeline. Album-type-specific
  preambles live in CORDELL_PHOTO_ALBUMS_TRAVEL and CORDELL_PHOTO_ALBUMS_FAMILY skills. Invoke any time
  the user mentions photo albums, Audrey's albums, AI index, manifest summary, caption problems, or
  specific photo filenames — even if they don't say "skill".
compatibility: >-
  Requires local GPU with GLM vision model (zai-org/glm-4.6v-flash). Model selection configured in
  ai_models.json. Object detection requires YOLO. Face matching requires InsightFace embeddings from Cast.
  Nominatim geocoding requires network access. MCP server: imago.
metadata:
  author: Cove Schneider
  version: 1.1.0
  mcp-server: imago
  documentation: references/photoalbums.md
  ocr-model: zai-org/glm-4.6v-flash
  caption-model: zai-org/glm-4.6v-flash
---

# Cordell Photo Albums AI Skill

## Overview

This skill serves two purposes that live in the same file:

**1. Claude workflow** — instructions for using the `imago` MCP tools to run and monitor AI indexing jobs
on Cordell family photo albums. See the Workflow and Quality Monitoring sections below.

**2. Shared vision model prompt sections** — rules, output formats, and hint templates loaded at runtime
by `photoalbums/lib/_caption_prompts.py`. These sections are parsed by exact `## Section Name` heading —
do not rename them. Read `references/photoalbums.md` for full pipeline documentation.

Album-type-specific preambles (`Preamble Describe`, `Preamble Combined Travel`, `Preamble Combined
Family`) live in the `CORDELL_PHOTO_ALBUMS_TRAVEL` and `CORDELL_PHOTO_ALBUMS_FAMILY` skills.

The pipeline runs four inference modes per image: Combined (OCR + caption in one GLM pass), Describe
(caption only, separate engine), People Count, and Location. Engine selection and mode branching are
controlled by `ai_models.json` — the skill templates apply to all modes.

---

## Workflow

Use the `imago` MCP tools to drive the AI indexing pipeline. The typical flow:

### Check what needs processing
Call `photoalbums_manifest_summary` first to see counts by state (pending, done, errored). This tells
you whether there's real work to do and how large the job will be.

### Start a job
Call `photoalbums_ai_index` to launch a background job. It returns a `job_id` immediately.
- Omit all filters to process all pending photos.
- Use `album` to scope to a single album directory (substring match, case-insensitive).
- Use `photo` for a single file (also forces reprocessing).
- Use `max_images` to cap the run during testing or spot-checks.
- Use `process_all_photos=true` to reprocess images already in the manifest.

### Monitor progress
Poll `job_status(job_id)` periodically (every 30–60 seconds for large runs). It returns status and a
recent log tail. When the job finishes, call `job_logs(job_id)` to retrieve the full output for
quality review. Use `job_list` to see all recent jobs if you've lost track of an ID.

### Cancel if needed
Call `job_cancel(job_id)` to terminate a running job gracefully.

---

## Quality Monitoring

After a job completes, review the logs for these common problems. When issues are found, report them
clearly: which photo, what the symptom is, and what the likely cause is.

### Cut-off captions
**Symptom:** Caption ends abruptly without terminal punctuation (`.`, `!`, `?`), ends mid-sentence,
or ends mid-word.
**Cause:** The `max_tokens` limit (default 96) was reached before the model finished. This is a
prompt or config issue, not a model failure.
**What to look for in logs:** The raw caption text before JSON parsing — if it lacks a closing
sentence, it was truncated.

### Empty or fallback captions
**Symptom:** `caption` is an empty string; log shows `fallback=True` or "returned empty output".
**Cause:** Model inference failed or returned nothing usable.
**Action:** Check if the image file is corrupt or the GPU ran out of memory. A single retry with
`photo=<filename>` will reprocess just that image.

### People not named
**Symptom:** Caption says "a man" or "a woman" instead of a matched name from Cast.
**Cause:** Face match fell below confidence threshold, or face was occluded. People recovery
(rembg + looser IOU) fires automatically in `auto` mode whenever the first pass finds any evidence
of people — if it still failed, the person likely needs a better reference embedding in Cast.

### OCR garbled or empty
**Symptom:** `ocr_text` is nonsense or empty despite visible text in the scan.
**Cause:** Low-contrast, faded, angled, or sepia-toned handwriting. GLM Combined mode handles
handwriting better than standalone OCR engines — confirm engine selection in `ai_models.json`.

### Location empty despite clear context
**Symptom:** `location_name` is empty for a photo with a visible place name or landmark.
**Cause (GLM Combined mode):** Expected — Combined mode does not run a separate location step.
Location inference only runs with the LMStudio engine.
**Cause (LMStudio mode):** Verify the location inference step ran by checking the payload
location block in the logs.

---

## Global Style & Behavior Rules (apply to every mode)
- State only supported facts.
- If evidence is insufficient, omit the detail or use the empty string, false, or 0 required by the output schema.
- Never reference file names, folder names, internal IDs (B02, P01, Archive, View, etc.), scan artifacts, or processing details.
- Never reference the medium itself: do not use "photograph(s)", "picture(s)", "image(s)", "a collection of photographs", "scanned album page", "this photograph shows", "this image depicts", "this photo", or any similar meta-references in captions. Describe the subjects and scenes directly.
- Always try to identify the location, as that this often provides the most value and context for the photo. If the location is uncertain, provide a reason (e.g. "the sign is too blurry to read" or "the architecture suggests it could be in either France or Italy").
- Write captions in a descriptive first-person voice explaining what's happening in the scene (e.g. "A nice road in the English [assuming the album is about England] country side" not "There appears to be a road in the country side of some country").
- When any subject in a photo is identifiable by visual appearance — whether a landmark, an artwork, an iconographic figure, an architectural style, a cultural object, or clothing — refer to it by its recognized proper name or description rather than a generic physical description. Apply this at every level of specificity the evidence supports: "the Parthenon" not "ancient ruins on a hill"; "a Buddhist Bodhisattva" not "a figure with a halo"; "a Dunhuang cave mural" not "a painting in red and green tones"; "Byzantine imperial regalia" not "traditional attire".
- Never use "traditional" as a standalone modifier without naming the tradition — say "Buddhist devotional figures" not "traditional figures", "Mogul architectural details" not "traditional details". When the specific tradition cannot be identified visually, use geographic or cultural context from visible text instead (e.g., "Chinese figures", "Moroccan tilework") rather than falling back to "traditional".
- When quoting visible text, reproduce it exactly as printed.
- Describe what things are, not how old they appear — avoid temporal qualifiers like "vintage", "historic", "antique", or "old".
- Think step-by-step internally if needed, but output only the final JSON.
  
## Text Handling & Correction Rules
- Copy all visible text into `ocr_text` exactly as printed: preserve spelling, capitalization, punctuation, spacing, and line breaks. Do not translate, normalize, or correct.
- Include only clearly legible portions of blurry or illegible text. Use corrected or translated understanding only in caption or location reasoning when confidence exceeds 95%.
- Infer completion only for words visibly truncated at scan edges when the intended word is obvious.
- In `ocr_text`, reproduce `BOOK 11` exactly as printed. In caption or location reasoning, interpret it as Book II.
- Never correct proper names, dates, personal captions, or ambiguous text unless visual evidence is unambiguous.
- For non-English text: preserve exactly in `ocr_text`; use English translation only in caption or location reasoning.
- If OCR text reads as a printed caption for the photo (a label or note written under or beside the image in the album), incorporate it naturally into the description rather than ignoring it.

## Location Rules (strict)
- Infer location only from visible text and unmistakable visual landmarks.
- Use the most specific well-documented place name (landmark, city, province/state, country) the evidence supports.
- Return `location_name` as an empty string if evidence is low, uncertain, or conflicting.
- Never use a generic place-type as the location query (e.g. "a beach", "a park", "a field", "a city street"). If no named specific place is identifiable, return an empty string.
- Do not infer obscure villages, townships, or precise sites without explicit evidence.
- Output GPS coordinates only when both values are literally visible in the image text; otherwise leave `gps_latitude` and `gps_longitude` empty.
- Use the established modern name for any place — do not add qualifiers like "now known as" or "formerly called" unless a name change is directly evidenced by something visible in the image.

## People Rules
- Count only clearly visible real people in the main photo.
- Exclude statues, dolls, paintings, posters, reflections, and tiny indistinct background figures.
- Hyphen-separated lowercase names in visible text (for example `leslie-tommy-robert`) indicate left-to-right order.
- Names printed below or centered on a photo refer to the person or people shown.
- In captions, use identified names directly and naturally when the mapping is clear.
- Use `child` or `baby` only when the person is clearly young.
- Do not guess identities or relationships.

## Album Classification Rules (apply in this order)

- Treat album title hints and classification hints as supporting context only (do not show them as visible text).
- Fix Roman numeral typo in album names: replace accidental "1" with "I" (e.g., Book 1 → Book I, Book 11 → Book II).
- Use the printed cover title (not a normalized version) when naming the album in captions.
- For album covers or title pages, describe the image as "the cover" or "the title page" of the photo album.
- Quote any visible cover labels exactly as they appear.
- Cordell family albums typically feature blue or white leathery covers with gold trim and the title printed in the lower-right corner.
- Classify albums by title:
  - Family albums contain "Family" in the title.
  - Travel albums contain one or more country names in the title.
  - Travel albums focus on a specific place and time; family albums span many years and locations.

## Preamble People Count
Count the number of clearly visible real people in this photo.

## Preamble Location
Determine the most useful location metadata for this photo.

## Preamble Cover Page
This image is an album cover or title page.

## Album Title Hint
Album title hint: {album_title}.

## Canonical Title Hint
Canonical album title hint: {canonical_title}.
Prefer the printed cover title over the normalized title when naming the album in a caption.

## Album Classification Hint
Album classification hint: {album_label}.

## Album Focus Hint
Album focus hint: {album_focus}.

## Output Format – Describe (full caption)
{"title": "...", "caption": "..."}

title: short title for the photo or page — if there is a printed caption or visible title text on the page (over 15 words counts as a page caption), use that verbatim; otherwise write a brief visual description of the scene.
caption: detailed description in first-person family voice using only supported facts.

## Preamble Page Photo Regions
This image is a scanned album page containing multiple photographs.
Identify each distinct photograph as a rectangle.
Do not describe the page itself as a "scanned album page" or similar in captions — describe the people and scenes directly.
Use the internationally recognized proper name for any famous landmark in descriptions — do not use a generic visual label (e.g., "Stari Most" not "stone bridge", "the Acropolis" not "ancient ruins on a hill").

## Output Format – Describe Page (with photo regions)
{"caption": "...", "location_name": "...", "photo_regions": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "description": "..."}]}

caption: detailed description in first-person family voice using only supported facts.
location_name: concise geocoding query or empty string.
photo_regions: list each distinct photograph; x/y/w/h are normalized rectangle coordinates (0–1, top-left origin, relative to full image); description is one sentence per photograph. Return an empty list if there are no clearly distinct photographs.

## Output Format – People Count
{"people_present": false, "estimated_people_count": 0}

people_present: true if one or more clearly visible real people are present, otherwise false.
estimated_people_count: best integer count of clearly visible real people.

## Output Format – Location
{"location_name": "...", "gps_latitude": "...", "gps_longitude": "..."}

location_name: concise geocoding query or empty string.
gps_latitude: decimal degrees if explicitly visible in image text, else empty string.
gps_longitude: decimal degrees if explicitly visible in image text, else empty string.

## People Hint
These people have been identified in this photo: {people_hint}.
Use their names directly in the caption when describing any person in the scene.
Do not replace provided names with generic phrases like "a man", "a woman", "two people", "a couple", or "tourists".

## People Hint With Positions
These people have been identified in this photo (deduplicate before referencing): {people_hint}.
Use their names directly in the caption when describing any person in the scene.
Do not replace provided names with generic phrases like "a man", "a woman", "two people", "a couple", or "tourists".

## People Count Hint
Known identified people: {people_hint}.

## People Count Hint With Positions
Known identified people (deduplicate before referencing): {people_hint}.

## Objects Hint
Detected objects: {object_list}.

## OCR Hint
OCR text hint: "{ocr_snippet}". If this reads as a printed caption or label for the photo, incorporate it naturally into the description.

