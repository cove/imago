## Context

`photoalbums_review_view_regions` already returns raw region data (pixel + normalised coords) from the XMP sidecar. What's missing is a visual sanity-check: a rendered image with boxes drawn over the actual photograph so a reviewer can immediately see whether the detected boundaries are right. Pillow is already present in the virtualenv (used by `_caption_lmstudio.py` for image resizing).

## Goals / Non-Goals

**Goals:**
- Add `photoalbums_render_view_regions(album_id, page)` to `mcp_server.py`
- Draw coloured, labelled bounding boxes on a downscaled copy of the view JPG
- Save the annotated image to `<Album>_View/_debug/<stem>_regions_debug.jpg`
- Return the annotated image as a `data:image/jpeg;base64,...` string so MCP clients that support image rendering can display it inline

**Non-Goals:**
- Modifying the original view JPG or its XMP sidecar
- Rendering at full archive resolution (this is a preview tool)

## Decisions

### Downscale before drawing
View JPGs can be large (5000+ px). Downscale to max 1500px on the longest edge before annotating — keeps the base64 output under ~300 KB and avoids truncation in MCP clients. The `_debug/` saved file uses this same downscaled version.

### Colour scheme
Cycle through a fixed palette of distinct colours indexed by region number. Box outline width scales with image size.

### Return format
Return a `str` with `data:image/jpeg;base64,<data>` followed by a brief JSON summary (region count, saved path). MCP clients that understand data URIs may render inline; others show the summary.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| No regions detected yet | Return `{"status": "not_detected"}` with no image |
| Base64 string too large for some clients | 1500px downscale cap keeps output manageable; document the cap |
