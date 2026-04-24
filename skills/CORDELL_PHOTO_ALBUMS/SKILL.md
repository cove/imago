---
name: cordell-photo-albums
description: >-
  Orchestration skill for AI captioning and indexing of Cordell family photo albums (scanned by
  Audrey Cordell). Use this skill whenever: kicking off or monitoring a photoalbums AI indexing job,
  checking the manifest summary (how many photos are done vs pending), reviewing or diagnosing caption
  quality (cut-off captions, empty captions, wrong people names, missing location metadata), reprocessing
  individual photos, or troubleshooting why captions look wrong. Runtime prompt text and tuning live under
  photoalbums/prompts/. Invoke any time the user
  mentions photo albums, Audrey's albums, AI index, manifest summary, caption problems, or specific photo
  filenames - even if they don't say "skill".
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

This skill is operator workflow documentation for using the `imago` MCP tools to run and monitor AI
indexing jobs on Cordell family photo albums. Runtime prompt text, output-format guidance, and tunable
model-call parameters are owned by `photoalbums/prompts/` and adjacent `params.toml` files.

Do not edit this skill to change production OCR, caption, people-count, location, date-estimate, or
verify-crops prompt behavior. Edit the corresponding file under `photoalbums/prompts/` instead.


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
| `_S##` | Raw scan | `.tif` | - |
| `_D##-##` | Derived image | `.tif` | - |
| `_V` | View page (any scan count) | - | `.jpg` |
| `_D##-##_V` | View derived image | - | `.jpg` |

- Pages start at P01. P00 is not valid.
- `_V` always marks a view output; `_S##` always marks an archive scan.
- Every `_V` file derives from one or more `_S##.tif` archive scans.
- XMP sidecars share the same stem as their companion image (`.xmp` extension).
- `dc:source` always names the archive TIF scan files, never the view page filename.

---

The default pipeline runs three inference modes per image: Describe, People Count, and Location.
The Describe response produces OCR text and caption together in one combined call, and the
Describe response must always include verbatim visible text in `ocr_text`. Engine selection and
mode branching are controlled by `photoalbums/ai_models.toml`, render settings, and prompt assets
under `photoalbums/prompts/`.

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
   `photoalbums_ai_index(photo="<AlbumName>_B<book>_P01")` - wait for it to complete before continuing.
3. If the cover was previously processed but predates this change (its `xmpDM:album` field may be empty or missing the year), reprocess it:
   `photoalbums_ai_index(photo="<AlbumName>_B<book>_P01", reprocess_mode="all")`
4. Once the cover is done, proceed with the full album job - non-cover pages will pick up the title from the cover's XMP sidecar automatically.

This step is especially important when processing a single page (e.g. `photo=...P25`) - always run the cover page first if the title is unknown.

### Start a job
Call `photoalbums_ai_index` to launch a background job. It returns a `job_id` immediately.
- Omit all filters to process all pending photos.
- Use `album` to scope to a single album directory (substring match, case-insensitive).
- Use `photo` for a single file (also forces reprocessing).
- Use `max_images` to cap the run during testing or spot-checks.
- Use `reprocess_mode="all"` to reprocess images already in the manifest.
- Pass `album_set` only when you need a non-default archive set.

### Monitor progress
Poll `job_status(job_id)` periodically (every 30-60 seconds for large runs). It returns status metadata
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


