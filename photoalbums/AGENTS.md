# AGENTS.md

Purpose: operating rules for the `photoalbums/` project.

## Project Skills

- Do not use brittle regex or string replacements to edit AI model responses. Improve the prompt instead so the model produces the right JSON.
- `photoalbums/prompts/` is the runtime prompt source of truth for the `photoalbums/` pipeline.

## Spec Maintenance

`SPEC.md` is the authoritative description of the photo album pipeline and must be updated in the same change that alters its underlying contracts. Use the table below to decide whether a code change is spec-affecting; if any row applies, edit the listed `SPEC.md` sections in the same commit (or explain in the PR why it does not apply).

**Spec writing style — RFC, not code tour.** `SPEC.md` describes contracts and behavior, not the implementation. When editing it:
- Do not reference internal file paths, module names, function names, or class names (e.g., `lib/pipeline.py`, `_run_pipeline_immich_face_refresh_step`). Refer to concepts instead: pipeline steps, services, sidecar fields, env vars.
- Refer to pipeline steps by their public step id (e.g., `immich-face-refresh`), services by their service name (e.g., Immich, lmstudio, Docling), and env vars by their exact name.
- A reader who has never seen the source tree should be able to reimplement the pipeline from the spec alone. The file/module pointers in the table below are aids for the *agent deciding whether a change is spec-affecting* — they do not belong in the spec text.

| If you change… | Update these SPEC.md sections |
| --- | --- |
| Pipeline step list, ids, labels, or dependencies in `lib/pipeline.py` | §5 (AI Processing Pipeline), §9.2 (Pipeline Step Records) |
| Pipeline step implementation in `commands.py` (new step, removed step, changed inputs/outputs, new required env var) | §1.4 (Required Specs), §5, §9.2, §15 (Key Dependencies & External Services) |
| Required env vars or service endpoints in `lib/ai_model_settings.py`, `lib/_caption_lmstudio.py`, or any new external integration (Immich, etc.) | §1.4, §10.2 (AI-Models Spec), §15 |
| XMP sidecar schema, namespaces, or `imago:Detections` keys in `lib/xmp_sidecar.py` | §6 (Sidecar Structure), §7 (XMP Elements), §9 (Pipeline State Tracking) |
| File naming, scan ingest naming, or stitch rules in `naming.py`, `scanwatch.py`, `bennett.py` | §1 (Naming), §2 (Pre-Pipeline Ingest), §11 (Naming Regexes) |
| Docling or RealRestorer configuration | §3 (Photo Region Detection), §4 (Restoration), §5 |
| Dependency version bumps (root `pyproject.toml`, `uv.lock`) or AI model identifiers/pinned commits (`ai_models.toml`, model names in code) | §1.4 (Required Specs), §5 (model versions per step), §15 (Key Dependencies & External Services — record the new version/commit so we have a history of what was used) |

Drive-by code edits (refactors, lint fixes, bug fixes that do not alter the contracts above) do not require spec updates.

## Photo Album File Naming Convention

All photo album files use this naming scheme:

```text
{Collection}_{Year}_B{book}_P{page}_{type}.{ext}
```

| Type token | Role | Archive ext | View ext |
| ------------ | ------ | ------------- | ---------- |
| `_S##` | Raw scan | `.tif` | - |
| `_D##-##` | Derived image | `.tif` | - |
| `_V` | View page (any scan count) | - | `.jpg` |
| `_D##-##_V` | View derived image | - | `.jpg` |

Rules:

- `_V` always and only marks a view output. `_S##` always and only marks an archive scan.
- `_D##-##` identifies a derived image; append `_V` for the view JPEG.
- Archive files are `.tif` and `.png`; view files are `.jpg` with no exceptions.
- `dc:source` on any view file references the archive TIF scan(s) it was derived from.
- Pages are numbered starting at `P01`. `P00` is not valid.
- XMP sidecars share the same stem as their companion image file.

## Data and Schema

- Photo albums are located in `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums`.
- Prefer standard XMP schema fields over custom namespaces for `imago`.

## UI and Terminology

- Prefer precise naming that matches behavior.
- The current preferred term is `Gamma Correction`, not `Brightness`.
- Keep user-facing labels aligned with metadata and renderer semantics.

## Preview Render Behavior

- Preview should apply only the current wizard stage effects by default:
  - Review step: bad-frame repair only.
  - Gamma Correction step: gamma correction only.
  - Summary step: combined output.
