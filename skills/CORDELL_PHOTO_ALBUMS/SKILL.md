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
  version: 1.1.2
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
you whether there's real work to do and how large the job will be. Then use
`photoalbums_manifest_query(album="...")` when you need concrete filenames, cover pages, or sidecar status.

### Ensure the cover page is processed first
Before processing any album pages, the cover page (P00 or P01) must be processed so the album title is available to all subsequent pages. Always check:

1. Call `photoalbums_album_status(album="...")` and inspect `cover_candidates` plus `cover_ready`.
2. If the cover page has not been processed yet (state is `pending` or absent), run a targeted job first:
   `photoalbums_ai_index(photo="<AlbumName>_B<book>_P00")` — wait for it to complete before continuing.
3. If the cover was previously processed but predates this change (its `xmpDM:album` field may be empty or missing the year), reprocess it:
   `photoalbums_ai_index(photo="<AlbumName>_B<book>_P00", process_all_photos=true)`
4. Once the cover is done, proceed with the full album job — non-cover pages will pick up the title from the cover's XMP sidecar automatically.

This step is especially important when processing a single page (e.g. `photo=...P25`) — always run the cover page first if the title is unknown.

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
quality review. Use `photoalbums_job_artifacts(job_id)` to inspect XMP outputs and prompt-debug artifacts for
specific photos. Use `job_list` to see all recent jobs if you've lost track of an ID.

### Cancel if needed
Call `job_cancel(job_id)` to terminate a running job gracefully.

### Audit what needs repair
Call `photoalbums_reprocess_audit(album="...")` to find files that look stale or incomplete before starting a
repair pass. Use this when you suspect missing stitched OCR authority, missing sidecars, stale sidecars, or
Cast-driven people-name refreshes.

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
- If evidence is insufficient, omit the detail or use the empty string, false, or 0 required by the output schema.
- When quoting visible text, reproduce it exactly as printed.
- Think step-by-step internally if needed, but output only the final JSON.
- Write captions as a single complete sentence ending with terminal punctuation. If length is a concern, write a shorter complete sentence — never truncate mid-sentence or mid-word.
  
## Text Handling & Correction Rules
- Copy all visible text into `ocr_text` exactly as printed: preserve spelling, capitalization, punctuation, spacing, and line breaks. Do not translate, normalize, or correct.
- Do not emit literal escape sequences like `\n`, `\r`, or `\t` inside text field values. If a field needs line breaks, use normal line breaks; if a field is defined as single-line, collapse line breaks to spaces.
- Include only clearly legible portions of blurry or illegible text. Use corrected or translated understanding only in caption or location reasoning when confidence exceeds 95%.
- Infer completion only for words visibly truncated at scan edges when the intended word is obvious.
- Never correct proper names, dates, personal captions, or ambiguous text unless visual evidence is unambiguous.
- For non-English text: preserve exactly in `author_text` or `scene_text`; use English translation only in caption or location reasoning. Set `ocr_lang` to the BCP-47 code of that language (e.g. `"zh"`, `"fr"`, `"ar"`). The pipeline will store the original-language text under its proper `xml:lang` code in `dc:description` alongside the English AI caption under `x-default`, so both versions are preserved in the XMP.
- Typed text on white paper strips or typed labels on the album page is album-authored annotation text.
- In this archive, album-authored annotation text is typed, often on white paper strips in a typewriter-style Courier-like font. Treat that text as high-authority archival evidence.
- Do not assign album-authored annotation text to a photo based on position alone. Use both spatial cues (proximity, centering, alignment, grouping, borders, page layout) and content cues (whether the text semantically matches what is visible in the candidate photo or photos).
- An annotation may apply to one photo, a group of photos, or the whole page. If one annotation fits multiple nearby photos, treat it as group-level rather than forcing it onto a single photo.
- Do not assume annotation text names people. It may identify a place, landmark, building, event, date, or short narrative note.
- If the target is ambiguous, preserve the text verbatim in `ocr_text` but do not force it into a specific photo description or location.

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
- Hyphen-separated names in visible text (for example `leslie-tommy-robert`) indicate left-to-right order.
- When typed annotation text clearly names people for a specific photo, use those names only when both the page layout and the visible photo content support that match.

## Album Classification Rules (apply in this order)
- Fix Roman numeral typo in album names: replace accidental "1" with "I" (e.g., Book 1 → Book I, Book 11 → Book II).
- Use the printed cover title (not a normalized version) when naming the album.
- Albums feature blue or white faux leathery covers with gold trim and the title printed in the lower-right corner and has a year and often a book number.
- Titles can be multiple lines, and have mulitple countries and dates in them (for example, a first line `Europe 1973` and a second line `Egypt 1974`). Preserve real line breaks in raw text fields and include all lines in the title as printed.
- When the title has mulitple countries and dates, you'll need to match them with the right photos. For example if you see Egypt 1974 and Europe 1973, you'll need to look at the contents of the photos to determine if it's Europe or Egypt for the album name; you can assume the photos aren't intertwined, but sometimes they are mixed as in the case of taking a boat from Egypt to Europe, pictures from both regions will appear on the same page, in which case you'd combine both into the album name.

## System Prompt - People Count
You count visible people in photographs.
Return only valid JSON matching the response_format schema.
Count clearly visible real people only.
Do not include reasoning or extra fields.

## System Prompt - Location
You extract location metadata for photographs.
Return only valid JSON matching the response_format schema.
Only return GPS coordinates when exact coordinates are explicitly visible in the image or OCR text.
If exact coordinates are not explicit, leave GPS fields empty.
Do not include reasoning or extra fields.

## System Prompt - OCR
You are an OCR engine.
Return only valid JSON matching the response_format schema.
Put the extracted text in the text field.
Do not describe the image, show reasoning, or add extra fields.

## Preamble People Count
Count the number of clearly visible real people.

## Preamble Location
Determine the most useful location metadata supported by visible evidence.

## Preamble Cover Page
This is an album cover or title page.
Read the full album title exactly as printed on the cover, including all countries, years, and book numbers if present.
Output `album_title` as a single-line storage title: preserve the printed words and order, but replace line breaks with spaces.
Do not output literal `\n` sequences inside `album_title`.
Do not normalize, romanize book numbers, or otherwise rewrite the title text.

## Output Format – Describe (full caption)
`{"author_text": "...", "scene_text": "...", "annotation_scope": "...", "location_name": "...", "album_title": "", "ocr_lang": ""}`

- `author_text`: typed album-authored annotation text that clearly applies to this photo. Otherwise empty string.
- `scene_text`: readable text visible inside the photographed scene itself, preserved verbatim in any language. Otherwise empty string.
- `annotation_scope`: one of `photo`, `group`, `page`, `none`, or `unknown`.
- `location_name`: concise geocoding query for GPS lookup when supported strongly enough by visible evidence; otherwise empty string.
- `album_title`: for cover pages only — the full album title as a single-line storage string, with any printed line breaks replaced by spaces (e.g. `"Egypt 1975"`, `"Mainland China Book 11"`, `"Europe 1973 Egypt 1974"`). Empty string for all other pages.
- `ocr_lang`: BCP-47 language code of the primary non-English text in `author_text` or `scene_text` (e.g. `"zh"` for Chinese, `"fr"` for French, `"ar"` for Arabic). Use `"en"` for English-only text. Empty string when there is no visible text.

## Output Format – Describe Page (with photo regions)
`{"author_text": "...", "scene_text": "...", "annotation_scope": "...", "location_name": "...", "photo_regions": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "author_text": "...", "scene_text": "...", "annotation_scope": "..."}]}`

- `author_text`: typed album-authored annotation text that's typed on a typewriter on strips of white paper, otherwise empty string.
- `scene_text`: readable text visible inside photographs, otherwise empty string.
- `annotation_scope`: one of `photo`, `group`, `page`, `none`, or `unknown`.
- `location_name`: concise geocoding query or empty string.
- `photo_regions`: list each distinct photograph; x/y/w/h are normalized rectangle coordinates (0–1, top-left origin, relative to full image)

## Output Format – People Count
`{"people_present": false, "estimated_people_count": 0}`

- `people_present`: true if one or more clearly visible real people are present, otherwise false.
- `estimated_people_count`: best integer count of clearly visible real people.

## Output Format – Location
`{"location_name": "...", "gps_latitude": "...", "gps_longitude": "..."}`

- `location_name`: concise geocoding query or empty string.
- `gps_latitude`: decimal degrees if explicitly visible in image text, else empty string.
- `gps_longitude`: decimal degrees if explicitly visible in image text, else empty string.

## Preamble Page Photo Regions Compact
- This page contains multiple photographs.
- Identify each distinct photograph as a rectangle.
- Do not invent visual descriptions for the photographs.
- Use typed album-page annotations only after deciding whether they belong to one photo, multiple photos, or the whole page based on both layout and photo contents.
- Return `author_text` for applicable typed album annotations and `scene_text` for readable in-photo text. Do not synthesize any other descriptive prose.

