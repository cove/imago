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
  version: 1.0.0
  mcp-server: imago
  documentation: ../CORDELL_PHOTO_ALBUMS/references/photoalbums.md
  ocr-model: zai-org/glm-4.6v-flash
  caption-model: zai-org/glm-4.6v-flash
---

# Cordell Photo Albums — Travel Skill

## Overview

Travel album variant of `CORDELL_PHOTO_ALBUMS`. This file contains the prompt template sections loaded
by `photoalbums/lib/_caption_prompts.py` for travel / photo-essay albums (albums whose title contains
country names). Sections are parsed by exact `## Section Name` heading — do not rename them.

For Workflow, Quality Monitoring, and shared rules (`Global Style & Behavior Rules`, `Location Rules`,
etc.), see the base `CORDELL_PHOTO_ALBUMS` skill.

---

## Preamble Describe
- Primarly focus on identifing the location, since many are famous locations and tell a lot about the photo.
- If there's an existing capation for a page, use that as the title, but if it's more than 15 words, use it as the caption and not the title.
- When Audrey Cordell or Leslie Cordell appear on a page, incorperate their names in to the caption, but don't say a "women named Audrey Cordell [reset of sentence]", just say "Audrey Cordell [rest of sentence]".
- When describing a page or a photo:
  - Use strong, direct statements without phrases like "suggesting," "implying," or "indicating." Just state what you see.
  - Emphasize clearly visible architectural style rather than age — use specific words like ruin, gothic, castle, palace, temple, Buddhist, Mulsim, Christian etc.
  - Use vivid, colorful language; incorporate appropriate metaphors/similes to convey mood and atmosphere.
- When identified people are present, name them in the caption using the People Hint.

