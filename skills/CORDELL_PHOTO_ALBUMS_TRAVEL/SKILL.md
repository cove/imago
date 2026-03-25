---
name: CORDELL_PHOTO_ALBUMS_TRAVEL
description: >-
  Travel album captioning skill for Cordell photo albums. Contains travel-specific preamble prompt
  templates used by the GLM pipeline for photo-essay / travel albums (albums with country names in the
  title). Loaded at runtime by `photoalbums/lib/_caption_prompts.py`. Use the base
  CORDELL_PHOTO_ALBUMS skill for job management, manifest checking, and quality monitoring.
compatibility: >-
  Requires local GPU with GLM vision model (zai-org/glm-4.6v-flash). Model selection configured in
  ai_models.json. Object detection requires YOLO. Face matching requires InsightFace embeddings from Cast.
  Nominatim geocoding requires network access. MCP server: imago.
metadata:
  author: Cove Schneider
  version: 1.0.1
  mcp-server: imago
  documentation: ../CORDELL_PHOTO_ALBUMS/references/photoalbums.md
  ocr-model: zai-org/glm-4.6v-flash
  caption-model: zai-org/glm-4.6v-flash
---

# Cordell Photo Albums - Travel Skill

## Overview

Travel album variant of `CORDELL_PHOTO_ALBUMS`. This file contains the prompt template sections loaded
by `photoalbums/lib/_caption_prompts.py` for travel / photo-essay albums (albums whose title contains
country names). Sections are parsed by exact `## Section Name` heading - do not rename them.

For Workflow, Quality Monitoring, and shared rules (`Global Style & Behavior Rules`, `Location Rules`,
etc.), see the base `CORDELL_PHOTO_ALBUMS` skill.

---

## Preamble Describe
Classify visible text into `author_text` and `scene_text`.
Use `author_text` typewriter written text on white paper strips.
Use `scene_text` only for readable text inside a the photographed scene itself, not the page itself.
Use page layout, OCR text, and visible contents only to decide whether typed annotation text applies to the photo.
Return empty strings when no applicable text exists for a field.
