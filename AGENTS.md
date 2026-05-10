# AGENTS.md

Purpose: repository-wide operating rules for AI coding agents working on this project.

## Core Policy

- You use `just dupes`, `just deadcode`, `just complexity`, `just lint` and `just test` to validate the quality of the code changes.
- Prefer stateless and reconstructing state from ground truth and rather than storing data in a database when possible.
- Limit code file sizes to about 500 lines.
- Do not use brittle regex and string replaacments to edit AI model responses, improve the prompt instead to get the correct output in JSON.
- Always bubble up the underlining errors when error reporting, don't interpet the errors or discard low level errors, for example if you try to write to a file and you get a permission error, you would buble up the error to the user or write the log as: <intention or process failed due to>:<OS permission error output>.
- Documentation: Don't add docstrings, comments, or type annotations to code you didn't change. Only add comments where the logic isn't self-evident.
- Defensive coding: Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs).
- Abstractions: Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task.

## Project Skills

- `photoalbums/prompts/` is the runtime prompt source of truth for the `photoalbums/` pipeline.

## Photo Album File Naming Convention

All photo album files use a structured naming scheme:

```
{Collection}_{Year}_B{book}_P{page}_{type}.{ext}
```

| Type token | Role | Archive ext | View ext |
|------------|------|-------------|----------|
| `_S##` | Raw scan | `.tif` | — |
| `_D##-##` | Derived image | `.tif` | — |
| `_V` | View page (any scan count) | — | `.jpg` |
| `_D##-##_V` | View derived image | — | `.jpg` |

Rules:
- `_V` always and only marks a view output. `_S##` always and only marks an archive scan.
- `_D##-##` identifies a derived image; append `_V` for the view JPEG.
- Archive files are `.tif` and `.png`; view files are `.jpg` — no exceptions.
- `dc:source` on any view file references the archive TIF scan(s) it was derived from.
- Pages are numbered starting at P01. P00 is not a valid page number.
- XMP sidecars share the same stem as their companion image file (`.xmp` extension).

## Data and Schema Migrations

- Treat `vhs/metadata/*/render_settings.json` and related metadata as migratable assets.
- When renaming keys or structures:
  - update readers/writers in code,
  - run migration across existing metadata files,
  - remove old keys/paths unless explicitly told to keep them.
- Keep a single canonical schema in code and on disk.
- Prefer standard XMP schema fields over custom namespaces for `imago`.
- Photo Albums are located in `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums`

## UI and Terminology

- Prefer precise naming that matches behavior.
- Current preferred term: `Gamma Correction` (not `Brightness`).
- Keep user-facing labels aligned with metadata/renderer semantics.

## Preview Render Behavior

- Preview should apply only the current wizard stage effects by default:
  - Review step: bad-frame repair only.
  - Gamma Correction step: gamma correction only.
  - Summary step: combined output.

## Engineering Defaults

- Make focused, minimal changes.
- Avoid introducing extra abstraction unless it reduces real maintenance cost.
- Update docs/help text when behavior or naming changes.

## Duplicate Code

When `just dupes` reports a duplicate-code finding (SKY-C401):

- **Do not** make superficial edits (rename variables, reorder statements, split into slightly different forms) to make the code look different to the detector. That is evading the problem, not solving it.
- **Do** refactor the duplicated logic into a shared function, helper, or class that both call sites use. The goal is genuine reuse — one canonical implementation, multiple callers.
- If the duplicated code serves genuinely different purposes and sharing would introduce harmful coupling, explain why before leaving it as-is. This is the exception, not the default.

## Python Environment

- For Python commands in this repo, do not rely on PATH-resolved `python`.
- Use `uv sync` from the repo root to create or update the project environment.
- From the repo root use  `just ...` for test, lint, and validation commands.
- Validate changed Python modules with `uv run python -m py_compile` when possible.
- To run ad-hoc python scripts use `uv run python -c`

## Git Hygiene

- Do not revert unrelated local changes.
- Commit only files related to the requested task unless asked otherwise.
- Use clear commit messages that describe behavior change.
- Git hooks are configured to run tests, don't use `--no-verify` to skip tests, instead fix the errors if there are any.


<claude-mem-context>
# Memory Context

# [imago] recent context, 2026-05-08 7:30pm PDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (21,094t read) | 504,894t work | 96% savings

### Apr 29, 2026
S23 Investigating `just photoalbums-watch` recipe failure at line 88 with exit code 1 (Apr 29, 6:23 PM)
### Apr 30, 2026
S24 photoalbums-map showing San Marino, Italy instead of San Marino, CA — debugging GPS geocoding bug in XMP pipeline (Apr 30, 2:52 PM)
### May 1, 2026
58 12:24p 🔵 map_server.py _handle_geocode_and_update Has idx-Branching Logic That Skips Main GPS Update
59 " 🔵 XMP File Family_1989-1993_B10_P05_D01-00_V.xmp Not Found Under imago Project Directory
63 12:25p 🔵 Move Function (searchLocation JS) Never Sends loc_idx — Always Targets Primary GPS (idx=-1 Branch)
64 12:27p 🔵 geocode_cache.json "san marino" Entry Lacks "raw" Field — Causes Cache Bypass and Live Nominatim Call Every Time
65 12:28p 🔵 ai_location.py Architecture: _serialize_geocode_result Omits raw Field If Empty, Explains Cache Entries Without raw
66 " 🔵 propagate-to-crops Pipeline Step Wrote European San Marino Coordinates Into XMP via _build_detections_payload
67 " 🔵 resolve_crop_location Priority Chain Explains How European Coordinates Propagated to Crop XMP
68 12:30p 🔵 _fallback_geocode_queries Generates No Fallbacks for Single-Part Names Like "San Marino"
69 " 🔵 map_server.py Has Been Refactored: Router Simplified, _send_markers Replaces _handle_list_photos With _XMP_PATHS Module Variable
70 12:31p 🔵 _coalesce_gps Returns New Value If Non-Empty; Falls Back to Existing XMP GPS Only When New Value Is Blank
71 " 🔵 run_server() Accepts Flexible Path Arguments to Populate _XMP_PATHS — Enables OneDrive XMP Discovery
72 12:32p 🔵 write_xmp_sidecar Skips location_city/_state/_country Coalescing for Crop XMP Files
73 12:33p 🔵 _description_role_for_sidecar_path Determines XMP Role by Directory Name: Photos Dir → DESCRIPTION_ROLE_CROP
S25 photoalbums-map GPS location bug: map pin showing San Marino, Italy instead of San Marino, CA — investigated root cause and fixed Move/drag operations to properly update Detections location blob (May 1, 12:33 PM)
74 12:36p 🔵 Map Drag-Drop Uses /api/update Endpoint With Raw Lat/Lon (Separate from Move Function's /api/geocode_and_update)
75 " 🔵 _handle_update Architecture: Reverse Geocodes Dropped Coordinates, Supports delete_shown_location Action and Undo Bypass
76 " 🔵 read_ai_sidecar_state GPS Priority: EXIF:GPSLatitude First, Then locations_shown[0] Fallback for Crop Files — Detections Location GPS Not Used
77 12:38p 🔵 undoLastMove Sends undoing:true to /api/update, Bypassing Reverse Geocode; _merge_xmp_tree Updates Detections Payload via _with_location_detections
78 " 🔵 _with_location_detections Only Fills Empty Detections Location Fields — Will NOT Overwrite Existing "San Marino" City/Country After Move
79 " 🔵 San Marino GPS Bug: EXIF Shows Italy Coords Despite "San Marino, CA" City Label
80 12:49p 🔴 map_server: Move/drag operations now overwrite stale Detections location blob
81 " ✅ map_server.py Detections-overwrite fix passes compile check
82 12:50p 🔴 Family_1989-1993_B10_P05_D01-00_V.xmp manually corrected from European to California GPS
83 12:51p 🔵 Test suite blocked by Windows WinError 448 on vhs/metadata mount point
84 12:52p ✅ photoalbums test suite passes 525/525 after map_server Detections fix
S26 Investigate why Docling caption_hints are mismatched between Family_1994_B00_P03_D03-00_V.xmp and D07-00_V.xmp — each XMP holds the other photo's caption (May 1, 12:52 PM)
85 1:15p 🔵 Docling Caption Mismatch Between XMP Files (D03/D07 Swapped)
86 " 🔵 Docling Caption Resolution Uses cref Index Into doc.texts — Likely Source of Caption Swap Bug
87 " 🔵 Caption/dc:description Handling Spread Across ~30 Python Files in imago/photoalbums
88 1:16p 🔵 XMP dc:description Write Pipeline Uses Description Roles Derived From Sidecar Path
89 1:17p 🔵 Affected XMP Files Located in OneDrive Photo Album Directory
90 1:18p 🔵 XMP Caption Swap Confirmed: D03 Has D07's Caption and Vice Versa, Plus Additional Data Errors
91 1:25p 🔵 Caption-to-Region Assignment in ai_view_regions.py Uses Geometric Distance — Can Produce Swaps
92 " 🔵 associate_captions() Ambiguity Threshold Is 10% of Image Width — Nearby Photos Trigger Silent Wrong-Winner Assignment
93 1:26p 🔵 Full caption_hint Pipeline Traced: Docling → Page XMP Region → propagate-to-crops → Crop dc:description
94 1:28p 🔵 Page Scan XMP Detections Payload Contains Zero Regions — MWG-RS XML Regions Are Stored Separately
95 " 🔵 Page Scan XMP Files Have No MWG-RS Regions At All — Region Data Not Present in S01/S02
96 " 🔵 Page Scan S01 XMP Has Full OCR Text Including Both Swapped Captions, No Regions, and Bogus GPS Matching D07
97 1:29p 🔵 Smoking Gun Found: Page View XMP Contains All 9 Region CaptionHints — D03/D07 Captions Match Spatial Positions But Are Swapped Relative to Physical Photos
98 1:30p 🔵 Docling Debug JSON Files Not Present for P03 — Raw Docling Output Unavailable for Root Cause Analysis
99 " 🔵 Debug Root for Photo Album Images Is `_debug/` Subdirectory Under the Album Root
S27 Configure statusline from shell PS1 configuration (May 1, 1:32 PM)
100 1:43p 🔵 Architecture Clarification: Docling Only Draws Bounding Boxes — LM Studio (Gemma4) Assigns Captions to Photos
### May 3, 2026
101 2:52p 🔵 No PS1 statusline configuration found in shell config files
S28 User requested "ntt" - no session activity or work output recorded (May 3, 2:53 PM)
102 7:06p 🔵 No PS1 configuration found in shell files
S32 Troubleshoot why OSC 777 notification command works over SSH but fails when run in Zellij with Nushell (May 3, 8:38 PM)
### May 6, 2026
105 10:38p 🔵 Ripgrep crashes with heap corruption when searching Codex configuration directories
107 10:39p 🔵 PowerShell heap corruption resolved by explicitly granting escalated sandbox permissions
108 10:48p 🔵 Hook fallback mechanism uses direct OSC escape codes for terminal notifications
S34 Add Windows-only environment variable to Nushell startup configuration while excluding macOS (May 6, 11:09 PM)
### May 7, 2026
S36 Change nushell prompt colors based on the machine being used (May 7, 1:01 AM)
109 1:14a ✅ Codex goals feature enabled via chezmoi-managed config
110 " 🔵 Chezmoi apply timeout with interactive state lock contention
113 9:30a 🔵 Codebase complexity analysis reveals high-complexity hotspots across photoalbums, vhs, and cast modules
114 3:57p 🔵 Ripgrep search for hook files failed with critical error
115 3:58p ✅ Added complexity check to git pre-push hook
### May 8, 2026
S37 Refactored photoalbums just recipes to improve naming clarity and wrap internal implementation details (May 8, 6:32 PM)
**Investigated**: Examined justfile to understand current photoalbums recipe structure and identified five recipes needing renaming: photoalbums-list-render-steps, photoalbums-list-watcher-steps, photoalbums-render, photoalbums-watch. Searched for references to these recipes across the codebase to understand scope of changes.

**Learned**: The justfile uses convention of separating user-facing recipe names from internal implementation details. Original names conflated the pipeline processing step with the public recipe interface. The refactoring introduces wrapper recipes that delegate to pipeline-specific implementations (photoalbums-render-pipeline, photoalbums-watch-scans, etc.), allowing future expansion without breaking the public API.

**Completed**: Modified justfile with six recipe renames deployed to main: (1) photoalbums-list-render-steps → photoalbums-list-render-pipeline-steps, (2) photoalbums-list-watcher-steps → photoalbums-list-scan-pipeline-steps, (3) photoalbums-render refactored as wrapper delegating to photoalbums-render-pipeline, (4) photoalbums-watch refactored as wrapper delegating to photoalbums-watch-scans. All quality checks (dupes, deadcode, complexity) passed. Commit pushed to origin/main after pre-push hook validation.

**Next Steps**: Session is complete. The refactored recipes are live on main branch. User can now update references in documentation or code that may reference the old recipe names if needed in future sessions.


Access 505k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>
