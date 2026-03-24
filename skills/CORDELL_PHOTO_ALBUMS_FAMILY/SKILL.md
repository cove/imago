---
name: CORDELL_PHOTO_ALBUMS_FAMILY
description: >-
  Family album captioning skill for Cordell photo albums. Contains family-specific preamble prompt
  templates used by the GLM pipeline for family albums (albums with "Family" in the title). Loaded at
  runtime by `photoalbums/lib/_caption_prompts.py`. Use the base CORDELL_PHOTO_ALBUMS skill for job
  management, manifest checking, and quality monitoring.
compatibility: >-
  Requires local GPU with GLM vision model (zai-org/glm-4.6v-flash). Model selection configured in
  ai_models.json. Object detection requires YOLO. Face matching requires InsightFace embeddings from Cast.
  Nominatim geocoding requires network access. MCP server: imago.
metadata:
  author: Cove Schneider
  version: 1.0.0
  mcp-server: imago
  documentation: ../CORDELL_PHOTO_ALBUMS/references/photoalbums.md
  ocr-model: zai-org/glm-4.6v-flash
  caption-model: zai-org/glm-4.6v-flash
---

# Cordell Photo Albums - Family Skill

## Overview

Family album variant of `CORDELL_PHOTO_ALBUMS`. This file contains the prompt template sections loaded
by `photoalbums/lib/_caption_prompts.py` for family albums (albums whose title contains "Family").
Sections are parsed by exact `## Section Name` heading - do not rename them.

For Workflow, Quality Monitoring, and shared rules (`Global Style & Behavior Rules`, `Location Rules`,
etc.), see the base `CORDELL_PHOTO_ALBUMS` skill.

---

## Preamble Describe
Describe the people, event, and actions directly.
If OCR provides a printed caption for the page or photo, use it as the title when it is short enough; otherwise shorten it to a concise title.
Do not use the album name as the title.
Use identified names naturally when they are provided.
Family references:
"Daddy" = Oliver Dennison
"Mommy" = Maude Dennison
"Gilbert" = Audrey's brother
"Leslie Cordell" = Audrey's husband
Frequent family locations:
San Marino, California
Woodhaven, Winnipeg, Canada
Indianapolis, Indiana
