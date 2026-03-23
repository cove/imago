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

# Cordell Photo Albums — Family Skill

## Overview

Family album variant of `CORDELL_PHOTO_ALBUMS`. This file contains the prompt template sections loaded
by `photoalbums/lib/_caption_prompts.py` for family albums (albums whose title contains "Family").
Sections are parsed by exact `## Section Name` heading — do not rename them.

For Workflow, Quality Monitoring, and shared rules (`Global Style & Behavior Rules`, `Location Rules`,
etc.), see the base `CORDELL_PHOTO_ALBUMS` skill.

---

## Preamble Describe
- Audrey Cordell assembled these albums and frequently appears in photos with her husband Leslie Cordell; identify and name them whenever they are recognizable, but don't say a "women named Audrey Cordell [reset of sentence]", just say "Audrey Cordell [rest of sentence]".
- Contain "Family" in the title; span many years and locations.
- Captions are written in the first person by Audrey Cordell; reflect this voice in descriptive captions.
- Focus on people present, the event or occasion, and their actions.
- Key family references:
  - "Daddy" = Oliver Dennison (Audrey's father)
  - "Mommy" = Maude Dennison (Audrey's mother)
  - Gilbert = Audrey's brother
  - Leslise Cordell = Audrey's husband
- Important locations:
  - San Marino, California (family home purchased 1958; frequent setting for holidays, Christmas, dining-room gatherings)
  - Woodhaven, Winnipeg, Canada (Audrey's childhood home)
  - Indianapolis, Indiana (Leslie's childhood home; many photos with relatives there in the 1980s and 1990s)

