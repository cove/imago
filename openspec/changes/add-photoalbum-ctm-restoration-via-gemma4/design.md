## Context

Imago already supports stitched photoalbum views, LM Studio-backed AI workflows, XMP sidecar updates, and MCP-driven async jobs. Recent work added region-detection and review flows that write structured metadata next to stitched photoalbum assets. This change extends the same metadata-first approach to chromatic restoration: derive a reusable 3×3 Color Transformation Matrix (CTM) from a stitched image, store it in standards-compatible XMP, and apply it deterministically during render.

The requested restoration target is red-shift degradation caused by cyan pigment failure. The user specifically wants CTMs generated from stitched images using a Gemma 4 model and stored using Adobe Camera Raw-compatible metadata (`crs:ColorMatrix1`) inside a master XMP manifest that can also carry stitch provenance data.

## Goals / Non-Goals

**Goals:**
- Add a Gemma 4-backed CTM estimation workflow for stitched photoalbum images.
- Persist CTMs in XMP using `crs:ColorMatrix1` while preserving existing photoalbum metadata.
- Support manifest-style metadata that can include `xmpMM:DocumentID`, `crs:HasSettings`, and optional stitch ingredient / homography data.
- Expose CTM generation and review through job-capable CLI and MCP entrypoints.
- Apply stored CTMs as deterministic linear transforms during render/export without modifying archival masters.

**Non-Goals:**
- Destructively rewriting `_Archive/*.tif` master scans.
- Full color-management redesign beyond the requested 3×3 matrix workflow.
- Training or fine-tuning a model for restoration.
- Replacing existing region-detection or face-metadata workflows.

## Decisions

### Use stitched images as CTM inputs
Estimate CTMs from stitched page images (`*_V.jpg` or equivalent stitched render inputs), not per-tile archival masters. This matches operator review surfaces and avoids having to reconcile tile-specific matrices into one page-level transform.

Alternative considered: estimate per-tile CTMs and merge later. Rejected because it complicates the workflow and does not match the user's requested stitched-image flow.

### Use Gemma 4 via LM Studio with structured JSON output
Use the local LM Studio OpenAI-compatible endpoint with a Gemma 4 model and a structured prompt that requests JSON containing matrix coefficients, confidence, and warnings. The application will validate that the model returns exactly 9 finite numeric coefficients before accepting the result.

Alternative considered: allow free-text matrix responses. Rejected as too fragile for a job pipeline.

### Store CTMs in Adobe Camera Raw-compatible XMP
Persist CTMs in XMP using `crs:ColorMatrix1`, with `crs:HasSettings=True`, and support adjacent manifest metadata such as `xmpMM:DocumentID`, `xmpMM:Ingredients`, `stRef:filePath`, and `archive:HomographyMatrix`. This keeps the restoration recipe portable across Imago and external imaging software.

Alternative considered: use a fully custom Imago-only namespace. Rejected because the user explicitly wants institutional-grade future-proofing and cross-platform compatibility.

### Apply CTMs deterministically at render time
Implement CTM application as a deterministic 3×3 linear transform in the Python render path before final stitched/rendered output is written. This ensures the same stored recipe can be applied consistently without needing Adobe software in the render pipeline.

Alternative considered: store metadata only and require external tools to apply it. Rejected because Imago itself needs to render restored outputs.

### Reuse existing job patterns for CTM generation and review
Expose generation/review through the existing CLI and `JobRunner` / MCP patterns used elsewhere in photoalbums. CTM generation can be launched for a page, album, or album set, and review surfaces should expose stored matrix values, confidence, and warnings.

Alternative considered: make CTM generation synchronous only. Rejected because model-backed generation may be slow and bulk album runs need job tracking.

## Risks / Trade-offs

- **Model returns malformed or unstable CTM values** → Use strict JSON prompting, retries, numeric validation, and reject invalid matrices.
- **CTM causes clipping or overcorrection** → Run sanity checks on coefficients and preview clipping metrics; persist warnings with the result.
- **Camera Raw metadata conflicts with existing photoalbum XMP fields** → Extend namespace handling carefully and test coexistence with current XMP structures.
- **Bulk reruns may overwrite operator-approved CTMs** → Require explicit `force=True` semantics for overwrite paths.
- **Render implementation diverges from external tools** → Treat `crs:ColorMatrix1` as the source of truth and keep Imago's transform logic deterministic and well-tested.
