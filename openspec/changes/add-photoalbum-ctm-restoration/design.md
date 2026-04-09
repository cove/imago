## Context

Imago already manages stitched photoalbum view images, XMP sidecars, LM Studio-backed AI jobs, and MCP-triggered async jobs. Recent OpenSpec work added view-region detection and review workflows that store structured metadata alongside photoalbum assets. That gives us the scaffolding for another metadata-first workflow: deriving a chromatic-restoration recipe from a stitched image and storing it as reusable XMP.

The requested restoration target is a 3×3 Color Transformation Matrix (CTM) that corrects red-shift caused by cyan pigment failure. The CTM must be computed from a stitched image rather than raw tile scans, then stored in a standards-friendly format that remains separate from archival masters. The requested schema aligns with Adobe Camera Raw metadata (`crs:ColorMatrix1`) inside an XMP manifest and may also include archive-specific stitch references.

## Goals / Non-Goals

**Goals:**
- Add a Gemma 4-backed CTM estimation workflow for stitched photoalbum images.
- Store CTM values in XMP using `crs:ColorMatrix1`.
- Support master-manifest style metadata that can also include stitch ingredients / homography data alongside the CTM recipe.
- Expose job-based CLI and MCP entrypoints for running CTM generation across albums/pages.
- Apply the stored CTM in Imago's render path as a non-destructive linear transform before final stitched/rendered output is written.
- Keep archival masters unchanged and keep the restoration recipe portable across tools.

**Non-Goals:**
- Destructively rewriting `_Archive/*.tif` master scans.
- Training or fine-tuning a vision/color model.
- Full ICC/profile management beyond the requested 3×3 matrix workflow.
- Solving every color-restoration problem; this change is specifically for CTM-based red-shift correction.

## Decisions

### CTM estimation input
Use the stitched page image (`*_V.jpg` or the stitched render precursor) as the input to CTM estimation. The model sees the page in the same visual composition operators review, which is what the user requested and avoids having to reconcile per-tile estimates later.

### Model integration
Use LM Studio with a Gemma 4 model as the CTM estimator. The model prompt should request structured JSON containing:
- `matrix`: 9 numeric values in row-major order
- `confidence`: scalar confidence
- `reasoning_summary`: short human-readable summary
- optional `warnings`: list of concerns (heavy clipping, low confidence, mixed illuminants)

The application should validate the response numerically before accepting it.

### XMP storage format
Store the CTM in XMP under `crs:ColorMatrix1`, formatted as a comma-separated 3×3 matrix string following Adobe Camera Raw / DNG conventions. Extend XMP handling so the same XMP document can also include archive-manifest information such as:
- `xmpMM:DocumentID`
- `crs:HasSettings=True`
- optional archive manifest structure with `xmpMM:Ingredients`, `stRef:filePath`, and `archive:HomographyMatrix`

The CTM recipe should live in a manifest/master XMP associated with the stitched page or stitched-photo output, not embedded as destructive pixel edits.

### Render-time application
Apply the stored CTM as a linear transform before final image render/export. Implementation should multiply each pixel's RGB vector by the 3×3 matrix in a deterministic Python path so renders remain reproducible without needing Adobe software.

### Job surface
Add a job-based MCP tool and CLI command, likely along the lines of:
- `photoalbums_generate_ctm(album_id, page=None, force=False, album_set=None)`
- `photoalbums_review_ctm(album_id, page, album_set=None)`
- optional bulk mode over an album or album set

Generation should run through the existing async `JobRunner` patterns because model inference and manifest writes may take time.

### Safety and validation
Before persisting a CTM:
- verify there are exactly 9 finite numeric coefficients,
- clamp or reject obviously broken outputs,
- optionally run a preview transform and compute simple sanity metrics (channel clipping percentage, extreme gain checks),
- record warnings when confidence is low rather than silently applying a risky matrix.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Model outputs malformed or unstable matrix values | Use strict JSON schema prompting, retries, numeric validation, and reject invalid matrices |
| CTM over-corrects and clips channels | Run preview sanity checks and persist warnings/confidence with the job output |
| XMP interoperability issues with Camera Raw fields plus custom archive fields | Register namespaces explicitly and write integration tests against representative XMP output |
| Operators need to rerun CTM generation after improvements | Support `force=True` and make CTM generation a repeatable job |
| Render pipeline divergence between Imago and Adobe tools | Treat `crs:ColorMatrix1` as source of truth and implement deterministic row-major matrix application in Python |

## Open Questions

- Should CTMs be stored per page-view image, per stitched individual photo, or both? Initial recommendation: per stitched page or stitched-photo asset that is actually rendered, with room to expand later.
- Should the pipeline automatically seed `archive:StitchManifest` from existing stitch metadata, or only write CTM-related fields initially?
- Do we want a separate MCP tool to preview before/after CTM application, or is generation + render sufficient for the first pass?
