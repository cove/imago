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

## Paths

When searching for XMP sidecar files, these are the locations they can be found:

- Top level directory `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums`
- Archive directory: `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums\*_Archive`
- Pages directory: `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums\*_Pages`
- Photos/crops directory: `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums\*_Photos`

