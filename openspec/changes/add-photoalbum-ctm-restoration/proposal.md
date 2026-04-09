## Why

Some archival scans exhibit red-shift degradation caused by cyan dye failure. Today Imago can stitch, index, and render album pages, but it does not have a first-class way to compute and persist a non-destructive chromatic restoration recipe that can be applied consistently during rendering. We need an institutional-grade mechanism that:

- derives a 3×3 Color Transformation Matrix (CTM) from the stitched page image,
- stores that CTM as durable metadata rather than baking it destructively into source scans,
- and lets operators run the workflow as a repeatable job across photo albums.

Using XMP keeps the restoration recipe decoupled from raw archival scans while remaining readable by both Imago's Python render pipeline and external imaging tools that understand Adobe Camera Raw metadata.

## What Changes

- Add a photoalbum chromatic-restoration workflow that computes a 3×3 CTM from a stitched `*_V.jpg` or equivalent stitched page image using the local LM Studio `gemma-4` model family.
- Persist the CTM in the album's master/manifests XMP using `crs:ColorMatrix1`, following the Adobe DNG / Camera Raw schema.
- Add a job-capable CLI + MCP entrypoint so operators can run CTM generation over one page, one album, or an album set.
- Teach the render/stitch pipeline to read the stored CTM recipe and apply it as a non-destructive linear color transform before final stitched output is rendered.
- Preserve source archival scans unchanged; CTM data lives in metadata/manifests, not baked into `_Archive/*.tif` masters.

## Capabilities

### New Capabilities

- `photoalbum-ctm-restoration`: Generate, persist, inspect, and apply chromatic-restoration CTMs for stitched photo album images using a Gemma 4 model-driven workflow.

### Modified Capabilities

- `view-xmp-regions`: existing photoalbum XMP flows must coexist cleanly with new Camera Raw / archive manifest fields in the same XMP documents.

## Impact

- New LM Studio model/config entry for the Gemma 4 CTM workflow.
- New photoalbum library module(s) for CTM estimation, validation, and matrix application.
- New XMP write/read support for `crs:ColorMatrix1` and archive stitch-manifest metadata.
- New MCP job/tool surface for launching CTM generation jobs and reviewing status/results.
- Render pipeline changes so stitched outputs can consume stored CTM recipes.
- Affected areas likely include `photoalbums/lib/`, `photoalbums.py`, `mcp_server.py`, and XMP helper code.
