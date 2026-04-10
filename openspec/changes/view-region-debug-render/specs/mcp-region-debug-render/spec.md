## ADDED Requirements

### Requirement: MCP tool renders detected regions as an annotated image
The system SHALL expose an MCP tool `photoalbums_render_view_regions(album_id, page)` that reads the current `mwg-rs:RegionList` from the view image's XMP sidecar, draws coloured labelled bounding boxes on a downscaled copy of the view JPG, saves the result to disk, and returns it as a base64 JPEG string.

#### Scenario: Successful render with multiple regions
- **WHEN** `photoalbums_render_view_regions(album_id="Egypt_1975_B00", page="26")` is called and the XMP sidecar contains 3 detected regions
- **THEN** the tool returns a string beginning with `data:image/jpeg;base64,` followed by the encoded image, and saves the annotated file to `<Album>_View/_debug/Egypt_1975_B00_P26_V_regions_debug.jpg`

#### Scenario: Each region box is visually distinct
- **WHEN** the annotated image is rendered
- **THEN** each region's bounding box is drawn in a different colour from a fixed palette, and labelled with its index number and caption (if present)

#### Scenario: No regions detected
- **WHEN** `photoalbums_render_view_regions` is called for a page whose XMP sidecar contains no `mwg-rs:RegionList` block
- **THEN** the tool returns `{"status": "not_detected"}` without producing an image file

#### Scenario: Output image is downscaled for manageability
- **WHEN** the view JPG is larger than 1500px on its longest edge
- **THEN** the annotated output image is downscaled to fit within 1500px before encoding and saving; the original view JPG is not modified
