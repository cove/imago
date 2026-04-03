---
name: cordell-photo-albums
description: >-
  Orchestration skill for AI captioning and indexing of Cordell family photo albums (scanned by
  Audrey Cordell). Use this skill whenever: kicking off or monitoring a photoalbums AI indexing job,
  checking the manifest summary (how many photos are done vs pending), reviewing or diagnosing caption
  quality (cut-off captions, empty captions, wrong people names, missing location metadata), reprocessing
  individual photos, or troubleshooting why captions look wrong. Also contains shared vision model prompt
  sections (rules, output formats, hints) used by the GLM captioning pipeline. Invoke any time the user
  mentions photo albums, Audrey's albums, AI index, manifest summary, caption problems, or specific photo
  filenames — even if they don't say "skill".
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

Legacy album-type-specific preambles are deprecated. The runtime pipeline now uses the base skill's
generic prompt sections directly.

## Requirements

- Local GPU with GLM vision model `zai-org/glm-4.6v-flash`
- Model selection configured in `photoalbums/ai_models.toml`
- YOLO for object detection
- InsightFace embeddings from Cast for face matching
- Network access for Nominatim geocoding
- MCP server: `imago`

---

## File Naming Convention

All album files follow the pattern `{Collection}_{Year}_B{book}_P{page}_{type}.{ext}`:

| Type token | Role | Archive ext | View ext |
|------------|------|-------------|----------|
| `_S##` | Raw scan | `.tif` | — |
| `_D##-##` | Derived image | `.tif` | — |
| `_V` | View page (any scan count) | — | `.jpg` |
| `_D##-##_V` | View derived image | — | `.jpg` |

- Pages start at P01. P00 is not valid.
- `_V` always marks a view output; `_S##` always marks an archive scan.
- Every `_V` file derives from one or more `_S##.tif` archive scans.
- XMP sidecars share the same stem as their companion image (`.xmp` extension).
- `dc:source` always names the archive TIF scan files, never the view page filename.
- Legacy suffixes `_VC`, `_VR`, `_stitched` are still recognised but not produced.

---

The default pipeline runs three inference modes per image: Describe, People Count, and Location.
The Describe response produces OCR text and caption together in one combined call, and the
Describe response must always include verbatim visible text in `ocr_text`. Engine selection and
mode branching are controlled by `photoalbums/ai_models.toml` and render settings — the skill
templates apply to all modes.

---

## Workflow

Use the `imago` MCP tools to drive the AI indexing pipeline. The typical flow:

### Select the archive set first
For routine Cordell work, omit `album_set` and use the server default archive set.
Only call `photoalbums_list_sets(kind="archive")` when you need to choose among multiple archive sets.
If you do pass `album_set`, use the exact short `album_set` value returned by the server, such as
`cordell`. Do not pass the description text.

### Check what needs processing
Call `photoalbums_manifest_summary()` first to see counts by state (pending, done, errored). This tells
you whether there's real work to do and how large the job will be. Then use
`photoalbums_manifest_query(album="...")` when you need concrete filenames, cover pages, or sidecar status.

### Ensure the cover page is processed first
Before processing any album pages, the cover page (P01) must be processed so the album title is available to all subsequent pages. Always check:

1. Call `photoalbums_album_status(album="...")` and inspect `cover_candidates` plus `cover_ready`.
2. If the cover page has not been processed yet (state is `pending` or absent), run a targeted job first:
   `photoalbums_ai_index(photo="<AlbumName>_B<book>_P01")` — wait for it to complete before continuing.
3. If the cover was previously processed but predates this change (its `xmpDM:album` field may be empty or missing the year), reprocess it:
   `photoalbums_ai_index(photo="<AlbumName>_B<book>_P01", reprocess_mode="all")`
4. Once the cover is done, proceed with the full album job — non-cover pages will pick up the title from the cover's XMP sidecar automatically.

This step is especially important when processing a single page (e.g. `photo=...P25`) — always run the cover page first if the title is unknown.

### Start a job
Call `photoalbums_ai_index` to launch a background job. It returns a `job_id` immediately.
- Omit all filters to process all pending photos.
- Use `album` to scope to a single album directory (substring match, case-insensitive).
- Use `photo` for a single file (also forces reprocessing).
- Use `max_images` to cap the run during testing or spot-checks.
- Use `reprocess_mode="all"` to reprocess images already in the manifest.
- Pass `album_set` only when you need a non-default archive set.

### Monitor progress
Poll `job_status(job_id)` periodically (every 30–60 seconds for large runs). It returns status metadata
only. When you need output, call `job_logs(job_id)` to retrieve the log text for
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


---

## Global Style & Behavior Rules (apply to every mode)
- If evidence is insufficient, omit the detail or use the empty string, false, or 0 required by the output schema.
- When quoting visible text, reproduce it exactly as printed.
- Think step-by-step internally if needed, but output only the final JSON.

## People Rules
- Count only clearly visible real people in the main photo.
- Exclude statues, dolls, paintings, posters, reflections, and tiny indistinct background figures.
- Hyphen-separated names in visible text (for example `leslie-tommy-robert`) indicate left-to-right order.
- When typed annotation text clearly names people for a specific photo, use those names only when both the page layout and the visible photo content support that match.

## System Prompt - People Count
- You count visible people in photographs.
- Return only valid JSON matching the response_format schema.
- Count clearly visible real people only.
- Do not include reasoning or extra fields.

## System Prompt - Location
- You extract location metadata for photographs.
- Return only valid JSON matching the response_format schema.
- When the response schema asks for `location_name`, only return GPS coordinates when exact coordinates are explicitly visible in the image or OCR text.
- If exact coordinates are not explicit, leave GPS fields empty.
- When the response schema asks for `locations_shown`, only include famous locations that can be confidently identified from visible evidence.
- If no famous locations are identifiable, return an empty array.
- Do not include reasoning or extra fields.

## System Prompt - OCR
- You are an OCR engine.
- Return only valid JSON matching the response_format schema.
- Put the extracted text in the text field.
- If there is no readable text, return an empty text field.
- Do not describe the image, show reasoning, or add extra fields.

## System Prompt - Date Estimate
- You estimate a photo date for XMP `dc:date`.
- Return only valid JSON matching the response_format schema.
- Use OCR text as the primary source of truth.
- If OCR text does not support a date, fall back to the album title.
- Return the most precise supported W3C date value: `YYYY-MM-DD`, `YYYY-MM`, `YYYY`, or an empty string.
- Do not invent a month or day unless the supplied text supports it.
- If the source only supports an approximate or rounded date, return the nearest honest precision instead of inventing missing parts.
- Never use placeholder components like `00` for month or day.
- If the day is unknown, return `YYYY-MM` instead of `YYYY-MM-00`.
- If the month is unknown, return `YYYY` instead of `YYYY-00` or `YYYY-00-00`.
- Do not include reasoning or extra fields.

## Preamble Describe
- Use `author_text` for typewriter-written Courier text on white paper strips.
- Use `scene_text` only for readable text inside the photographed scene itself, not the page itself.
- Return empty strings when no applicable text exists for a field.

## Preamble Upstream OCR Context
- The following OCR text comes from the parent album page XMP and is context only: {context_ocr_text}
- Use it only to disambiguate cropped annotations, partial captions, or place names visible in the current image.
- Do not copy words from this context into `ocr_text`, `author_text`, or `scene_text` unless they are visible in the current image.

## Preamble People Count
- Count the number of clearly visible real people.

## Preamble Location
- If the response schema asks for `location_name`, determine the most useful location metadata supported by visible evidence.
- When returning `location_name`, prefer a disambiguating geocoding query that includes the country, state/province, or broad region when that context is supported by the visible evidence, OCR text, or album title.
- Avoid ambiguous bare place names like `Oxford` when a more specific query like `Oxford, England` or `Oxford, United Kingdom` is supported.
- Album titles can have multiple country or region names in them, including a year, but you need to return just one country or region name only. For example "Spain and Morocco 1988", you'd look at the picture and OCR text to determine if the country is "Spain" or "Morocco" but not both, and "Egypt 1975" you'd pick "Egypt" but not "1975".
- If the response schema asks for `locations_shown`, identify distinct famous locations visible in the photographs on this page.
- Examples: landmarks, monuments, city skylines, famous natural features, iconic buildings.
- OCR text hints about the general location are provided to help identify specific famous locations, often these hints are the exact name of the location or landmark and little other work is needed.
- Only include locations that are well-known and can be identified with reasonable confidence.
- Include `name` when the landmark or place itself can be named, not just the city or country.
- Use `name` for the full location name, for example `Tower Bridge`, `Westminster Abbey`, or `St. Mark's Square`.
- Leave `name` empty when only the broader city or country can be identified.
- The album name is: {album_title}
- The OCR text on the page is: {ocr_text}

## Preamble Date Estimate
- Estimate a single photo date for XMP `dc:date`.
- First look for an explicit or strongly implied date in the OCR text.
- If the OCR text does not support a date, use the album title as the fallback date range hint.
- When only a year is supported, return the year only.
- When a month and year are supported, return `YYYY-MM`.
- When a full calendar date is supported, return `YYYY-MM-DD`.
- If the visible text implies an approximate date, round to the nearest supported precision without adding unsupported detail.
- Example: `about January 1988` -> `1988-01`.
- Example: `early 1988` -> `1988`.
- Example: `winter 1988` -> `1988`.
- Never return `00` for an unknown month or day; reduce precision instead.
- Example: `January 1988` -> `1988-01`, not `1988-01-00`.
- Example: `1988` -> `1988`, not `1988-00` or `1988-00-00`.
- If neither OCR text nor album title supports any date estimate, return the empty string.

## Preamble Cover Page
- This is an album cover or title page.
- Use the OCR text from this page as the source of truth for `album_title`.
- Read the full album title exactly as printed on the cover, including all countries, years, and book numbers if present.
- Output `album_title` as a single-line storage title: preserve the printed words and order, but replace line breaks with spaces.
- Do not output literal `\n` sequences inside `album_title`.
- Do not normalize, romanize book numbers, or otherwise rewrite the title text.

## Output Format – Describe Page (with photo regions)
`{"ocr_text": "", "author_text": "", "scene_text": "", "location_name": "", "photo_regions": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "author_text": "", "scene_text": ""}]}`

- `ocr_text`: you're an OCR engine when processing this, look for all clearly legible visible text on the page, copied verbatim with original capitalization, punctuation, spacing, and real line breaks.
- `author_text`: you're an OCR engine when processing this, typed album-authored annotation text that's typed on a typewriter on strips of white paper, otherwise empty string.
- Recover the full `author_text` when the strip is visibly present but cropped in this scan and the supplied `ocr_text` contains the missing words.
- `scene_text`: you're an OCR engine when processing this, readable text visible inside photographs, otherwise empty string.
- `author_text` and `scene_text` are classified subsets of `ocr_text`, not replacements for it. Fill them whenever the classification is supported by the page.
- The example JSON uses empty strings as placeholders. Do not copy literal `...` from any example or schema text.
- `location_name`: concise geocoding query for GPS lookup when supported strongly enough by visible evidence; otherwise empty string.
- Prefer `place, city/state, country` style queries over ambiguous short names when the broader geography is supported by the page or album context.
- `photo_regions`: list each distinct photograph; x/y/w/h are normalized rectangle coordinates (0–1, top-left origin, relative to full image); round each value to at most 3 decimal places (e.g. `0.297`, not `0.2978260869...`)
- `album_title`: for album title pages or cover pages — the full album title as a single-line storage string, with any printed line breaks replaced by spaces (e.g. `"Egypt 1975"`, `"Mainland China Book 11"`, `"Europe 1973 Egypt 1974"`). Empty string for all other pages.
- `ocr_lang`: BCP-47 language code of the primary non-English text in `author_text` or `scene_text` (e.g. `"zh"`, `"fr"`, `"ar"` for Chinese, French, Arabic). Use `"en"` for English-only text. Empty string when there is no visible text.
- Just return the JSON without any extra text or explanation.

## Output Format – People Count
`{"people_present": false, "estimated_people_count": 0}`

- `people_present`: true if one or more clearly visible real people are present, otherwise false.
- `estimated_people_count`: best integer count of clearly visible real people.
- Just return the JSON without any extra text or explanation.

## Output Format – Location
`{"location_name": "", "gps_latitude": "", "gps_longitude": ""}`

- `location_name`: concise geocoding query or empty string.
- Include country, state/province, or broad region when supported so Nominatim is less likely to resolve to the wrong place.
- `gps_latitude`: decimal degrees if explicitly visible in image text, else empty string.
- `gps_longitude`: decimal degrees if explicitly visible in image text, else empty string.
- Just return the JSON without any extra text or explanation.

## Output Format – Location Shown
`{"locations_shown": [{"name": "", "world_region": "", "country_name": "", "country_code": "", "province_or_state": "", "city": "", "sublocation": ""}]}`

- Return empty array if no famous locations are identifiable.
- Each location should include as many hierarchical levels as known.
- country_code: 2-letter ISO 3166-1 alpha-2 code if known.
- Just return the JSON without any extra text or explanation.

## Preamble Page Photo Regions Compact
- This page contains multiple photographs.
- Photos can be directly next to eachother, so you have to look for the seam between them sometimes, if there isn't an obvious border on the page.
- Identify each distinct photograph as a rectangle.
- Do not invent visual descriptions for the photographs.
- Use typed album-page annotations only after deciding whether they belong to one photo, multiple photos, or the whole page based on both layout and photo contents.

## Output Format - Date Estimate
`{"date": ""}`

- `date`: estimated W3C date string for `dc:date`, using one of `YYYY-MM-DD`, `YYYY-MM`, `YYYY`, or `""`.
- Prefer OCR evidence over album-title fallback.
- Just return the JSON without any extra text or explanation.


