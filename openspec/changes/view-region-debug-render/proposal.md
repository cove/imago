## Why

After running `photoalbums_detect_view_regions`, the only way to validate detected boundaries is to read raw normalised coordinates from `photoalbums_review_view_regions` and mentally map them to the image. A debug-render endpoint closes this loop by producing a JPEG with coloured bounding boxes drawn directly on the view image, so a human or AI reviewer can instantly see whether the region boundaries are correct before committing to a photo split.

## What Changes

- New MCP tool `photoalbums_render_view_regions(album_id, page)` that reads the current `mwg-rs:RegionList` from the view image's XMP sidecar, draws labelled bounding boxes over the view JPG using Pillow, and returns the annotated image as a base64-encoded JPEG string in the tool response
- The rendered image is also saved to a `_debug/` subdirectory inside the `_View/` folder for persistent inspection
- Region boxes are colour-coded by index and labelled with the region number and caption (if present)

## Capabilities

### New Capabilities

- `mcp-region-debug-render`: MCP tool that renders detected photo regions as annotated bounding boxes on a copy of the view JPG, returning the result as a base64 JPEG string and saving it to `<Album>_View/_debug/<stem>_regions_debug.jpg`

### Modified Capabilities

## Impact

- New dependency: Pillow (already present in the project)
- Writes to `<Album>_View/_debug/` — not part of the archive; debug output only
- One new tool added to `mcp_server.py`; no XMP or schema changes
