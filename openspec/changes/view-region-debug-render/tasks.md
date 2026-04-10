## 1. Render Helper

- [x] 1.1 Create `photoalbums/lib/ai_view_region_render.py` with `render_regions_debug(image_path, regions, output_path) -> bytes` — opens view JPG, downscales to max 1500px, draws coloured labelled boxes using Pillow `ImageDraw`, saves to `output_path`, returns JPEG bytes
- [x] 1.2 Implement colour palette (8 distinct colours, cycled by region index) and outline width scaled to image size

## 2. MCP Tool

- [x] 2.1 Add `photoalbums_render_view_regions(album_id, page, album_set=None)` tool to `mcp_server.py` — resolves view path, reads regions from XMP via `read_region_list`, calls render helper, returns `data:image/jpeg;base64,...` string with a brief JSON summary appended

## 3. Tests

- [x] 3.1 Add unit test for `render_regions_debug`: verify output is valid JPEG bytes and that calling with empty regions list still returns a valid image
- [x] 3.2 Add unit test verifying downscale: image larger than 1500px on longest edge is scaled down in the output
