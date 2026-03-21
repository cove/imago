---
name: CORDELL_PHOTO_ALBUMS
description: >-
  Workflow and prompt templates for AI captioning and indexing of Cordell family photo albums (scanned by
  Audrey Cordell). Use this skill whenever: kicking off or monitoring a photoalbums AI indexing job,
  checking the manifest summary (how many photos are done vs pending), reviewing or diagnosing caption
  quality (cut-off captions, empty captions, wrong people names, missing location metadata), reprocessing
  individual photos, or troubleshooting why captions look wrong. Also contains all vision model prompt
  templates used by the GLM captioning pipeline (combined OCR+caption, describe, people count, location
  inference). Invoke any time the user mentions photo albums, Audrey's albums, AI index, manifest summary,
  caption problems, or specific photo filenames — even if they don't say "skill".
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

**2. Vision model prompt templates** — sections loaded at runtime by `photoalbums/lib/_caption_prompts.py`
to assemble prompts sent to the GLM vision model. These sections are parsed by exact `## Section Name`
heading — do not rename them. Read `references/photoalbums.md` for full pipeline documentation.

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
(rembg + looser IOU) fires automatically in `auto` mode — if it still failed, the person likely
needs a better reference embedding in Cast.

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
- Never use phrases like "scanned album page", "this photograph shows", "this image depicts", "this photo", or any similar meta-references in captions.
- Write captions in a descriptive first-person voice explaining what's happening in the scene (e.g. "A nice road in the English [assuming the album is about England] country side" not "There appears to be a road in the country side of some country").
- When quoting visible text, reproduce it exactly as printed.
- Think step-by-step internally if needed, but output only the final JSON.

## Text Handling & Correction Rules
- Copy all visible text into `ocr_text` exactly as printed: preserve spelling, capitalization, punctuation, spacing, and line breaks. Do not translate, normalize, or correct.
- Include only clearly legible portions of blurry or illegible text. Use corrected or translated understanding only in caption or location reasoning when confidence exceeds 95%.
- Infer completion only for words visibly truncated at scan edges when the intended word is obvious.
- In `ocr_text`, reproduce `BOOK 11` exactly as printed. In caption or location reasoning, interpret it as Book II.
- Never correct proper names, dates, personal captions, or ambiguous text unless visual evidence is unambiguous.
- For non-English text: preserve exactly in `ocr_text`; use English translation only in caption or location reasoning.

## Location Rules (strict)
- Infer location only from visible text and unmistakable visual landmarks.
- Use the most specific well-documented place name (landmark, city, province/state, country) the evidence supports.
- Return `location_name` as an empty string if evidence is low, uncertain, or conflicting.
- Do not infer obscure villages, townships, or precise sites without explicit evidence.
- Output GPS coordinates only when both values are literally visible in the image text; otherwise leave `gps_latitude` and `gps_longitude` empty.

## People Rules
- Count only clearly visible real people in the main photo.
- Exclude statues, dolls, paintings, posters, reflections, and tiny indistinct background figures.
- Hyphen-separated lowercase names in visible text (for example `leslie-tommy-robert`) indicate left-to-right order.
- Names printed below or centered on a photo refer to the person or people shown.
- In captions, use identified names directly and naturally when the mapping is clear.
- Use `child` or `baby` only when the person is clearly young.
- Do not guess identities or relationships.

## Album Classification Rules (apply in this order)
- Treat album title hints and classification hints as supporting context, not as visible text.
- There's a typo in album names where the number of the Book is written in Roman Numerals, but 1 was used instead of I by accident for one. Replace this 1 with an I in the album names. (e.g. Book 11 is really Book II)
- Prefer the printed cover title over a normalized album title when naming the album in a caption.
- If the image is an album cover or title page, describe it as the cover or title page of the photo album.
- Preserve visible cover labels exactly as shown when quoting text.
- Cordell family album covers are typically blue or white with a leathery texture, gold trim, and the album title printed in the lower-right corner.

## Preamble Describe
Describe this photo in exhaustive detail.

## Preamble Combined
Analyze this photo. Perform both tasks:
1. Extract all visible text exactly as it appears. If none, output an empty string.
2. Write one precise sentence describing the scene.

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
{"caption": "..."}

caption: detailed description in first-person family voice using only supported facts.

## Preamble Page Photo Regions
This image is a scanned album page containing multiple photographs.
Identify each distinct photograph as a rectangle.
Do not describe the page itself as a "scanned album page" or similar in captions — describe the people and scenes directly.

## Output Format – Describe Page (with photo regions)
{"caption": "...", "location_name": "...", "photo_regions": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "description": "..."}]}

caption: detailed description in first-person family voice using only supported facts.
location_name: concise geocoding query or empty string.
photo_regions: list each distinct photograph; x/y/w/h are normalized rectangle coordinates (0–1, top-left origin, relative to full image); description is one sentence per photograph. Return an empty list if there are no clearly distinct photographs.

## Output Format – Combined
{"ocr_text": "...", "caption": "...", "location_name": "...", "album_title": "..."}

ocr_text: all visible text exactly as shown, or empty string.
caption: one sentence describing the scene in first-person family voice using only supported facts.
location_name: concise geocoding query or empty string.
album_title: canonical album title derived solely from visible cover text — only populate for cover or title pages. Romanize book numbers exactly as printed (BOOK 11 → Book II, BOOK II → Book II). Include the year only if it appears on the cover (e.g. "England Book II 1983"). Do not copy, combine, or extend the album title hint; derive only from what is visible. Empty string for all other pages.

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
Known people: {people_hint}.
Refer to these people by name in the caption wherever they appear.

## People Hint With Positions
Known people in this image (deduplicate before referencing): {people_hint}.
Refer to these people by name in the caption wherever they appear.

## People Count Hint
Known identified people: {people_hint}.

## People Count Hint With Positions
Known identified people (deduplicate before referencing): {people_hint}.

## Objects Hint
Detected objects: {object_list}.

## OCR Hint
OCR text hint: "{ocr_snippet}".

