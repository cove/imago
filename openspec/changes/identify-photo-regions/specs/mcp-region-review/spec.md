## ADDED Requirements

### Requirement: MCP tool to trigger view region detection
The system SHALL expose an MCP tool `photoalbums_detect_view_regions(album_id, page=None, force=False)` that enqueues an async detection job and returns a `job_id` for monitoring.

#### Scenario: Trigger detection for a single page
- **WHEN** `photoalbums_detect_view_regions(album_id="bennett_01", page="001")` is called
- **THEN** the system enqueues a job to run detection on that page's view image and returns `{"job_id": "<id>", "status": "started", ...}`

#### Scenario: Trigger detection for all pages in an album
- **WHEN** `photoalbums_detect_view_regions(album_id="bennett_01")` is called with no `page` argument
- **THEN** the system enqueues jobs for all view images in the album whose XMP sidecar does not yet contain a `mwg-rs:RegionList` block (or all if `force=True`)

### Requirement: MCP tool to review current region boundaries
The system SHALL expose an MCP tool `photoalbums_review_view_regions(album_id, page)` that returns structured data about the current detected regions for a view image, intended for external AI validation.

#### Scenario: Review returns region data with image path
- **WHEN** `photoalbums_review_view_regions(album_id="bennett_01", page="001")` is called
- **THEN** the system returns a JSON object containing: `image_path` (absolute path to view JPG), `image_width`, `image_height`, `regions` (list of region objects with pixel coords, normalised coords, confidence, and caption), and `caption_ambiguous` flag

#### Scenario: Review when no detection has run yet
- **WHEN** `photoalbums_review_view_regions` is called for a page whose XMP sidecar contains no `mwg-rs:RegionList` block
- **THEN** the system returns `{"regions": [], "status": "not_detected"}` without calling the vision model

#### Scenario: Review returns data even when LM Studio is offline
- **WHEN** LM Studio is unreachable but the XMP sidecar already contains a `mwg-rs:RegionList` block
- **THEN** the review tool reads and returns the region data from the XMP successfully

### Requirement: MCP tool to update region boundaries
The system SHALL expose an MCP tool `photoalbums_update_view_region(album_id, page, region_index, x, y, width, height)` that allows an external agent to correct a region's pixel bounding box and re-triggers XMP write-back.

#### Scenario: Correct a misdetected boundary
- **WHEN** `photoalbums_update_view_region(album_id="bennett_01", page="001", region_index=1, x=50, y=50, width=450, height=600)` is called
- **THEN** the `mwg-rs:RegionList` entry for region 1 is updated in the XMP sidecar, and the response confirms the update with the new normalised coordinates
