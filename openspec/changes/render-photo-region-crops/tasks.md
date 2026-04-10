## 1. _Photos Directory Helper

- [ ] 1.1 Add `get_photos_dirname(archive_path)` to `stitch_oversized_pages.py` mirroring `get_view_dirname`: replace `_Archive` suffix with `_Photos`
- [ ] 1.2 Add unit test: `get_photos_dirname("Egypt_1975_Archive")` → `"Egypt_1975_Photos"`

## 2. caption_hint and person_names Storage in Region XMP (prerequisite)

- [ ] 2.1 Update `write_region_list` in `xmp_sidecar.py` to store each region's `caption_hint` as `imago:CaptionHint` and `person_names` as an `imago:PersonNames` bag in the region's XMP block alongside `dc:description`
- [ ] 2.2 Update `read_region_list` (or equivalent reader) in `xmp_sidecar.py` to return both `caption_hint` and `person_names` per region when reading back `mwg-rs:RegionList`
- [ ] 2.3 Add unit test: region written with `caption_hint="People at the beach"` and `person_names=["Alice", "Bob"]` reads back with the same values; region with empty `person_names` reads back as `[]`

## 3. Caption Resolution Helper

- [ ] 3.1 Add `resolve_region_caption(region_dc_description, region_caption_hint, page_dc_description) -> str` in `ai_photo_crops.py`: returns first non-empty value in priority order; returns `""` if all are empty
- [ ] 3.2 Add unit tests for `resolve_region_caption`: region description wins; hint used when description empty; page caption used when both empty; empty string when all empty

## 4. Region Crop Module

- [ ] 4.1 Create `photoalbums/lib/ai_photo_crops.py` with `mwgrs_normalised_to_pixel_rect(cx, cy, w, h, img_w, img_h) -> tuple[int, int, int, int]`: convert centre-point normalised to `(left, top, right, bottom)` pixel rect, clamped to image bounds; log warning if any dimension was clamped by >5%
- [ ] 4.2 Add `crop_output_path(view_path, region_index, photos_dir) -> Path`: builds `_D{index:02d}-00_V.jpg` path under `photos_dir`
- [ ] 4.3 Add `crop_page_regions(view_path, photos_dir, *, force=False) -> int`: reads `mwg-rs:RegionList` and page-level `dc:description` from view sidecar; for each region resolves caption via `resolve_region_caption`; converts coords, crops pixels from `_V.jpg` using Pillow, writes JPEG, writes sidecar; returns count of crops written; skips silently if no regions; skips existing crops without `force`
- [ ] 4.4 Add `_write_crop_sidecar(crop_path, view_path, caption, view_state, locations_shown, person_names)` in `ai_photo_crops.py`: calls `assign_document_id`, `write_derived_from`, `write_pantry_entry` from `xmpmm_provenance`; writes `dc:description` only if `caption` is non-empty; writes `dc:source`; propagates location and date from `view_state`: `exif:GPSLatitude/Longitude`, `photoshop:City/State/Country`, `Iptc4xmpExt:Sublocation`, `Iptc4xmpExt:LocationCreated`, `xmp:CreateDate`, `dc:date`; passes `locations_shown` as the `LocationShown` bag; propagates `dc:subject` keywords; writes `person_names` (from the region's `imago:PersonNames` bag, as read by `read_region_list`) to `Iptc4xmpExt:PersonInImage`; omit each field if absent/empty
- [ ] 4.5 Add unit tests for `mwgrs_normalised_to_pixel_rect`: centre region, edge-touching region, out-of-bounds region clamped correctly
- [ ] 4.6 Add unit tests for `crop_page_regions`: no regions → 0 crops; 2 regions with captions → correct captions on sidecars; region with empty caption + non-empty page caption → page caption on sidecar; existing file skipped without force; existing file overwritten with force
- [ ] 4.7 Add unit test for `_write_crop_sidecar`: sidecar contains DocumentID, DerivedFrom pointing to view JPEG, caption when provided, no `dc:description` field when caption is empty
- [ ] 4.8 Add unit test for `_write_crop_sidecar` location/subject propagation: GPS, city, country, LocationShown, CreateDate, and dc:subject present on view → all appear on crop sidecar; empty location/subject on view → those fields absent on crop

## 5. Pipeline State Integration

- [ ] 5.1 Call `write_pipeline_step(view_xmp, "crop_regions")` in `crop_page_regions` after all crops for a page succeed
- [ ] 5.2 Add pipeline state check at start of `crop_page_regions`: return early with skip message if `pipeline.crop_regions` present and `force=False`
- [ ] 5.3 Add `--force` handling: call `clear_pipeline_steps(view_xmp, ["crop_regions"])` before re-cropping; delete all existing `_D##-00_V.jpg` files and their `.xmp` sidecars in `_Photos/` for the page before writing new crops
- [ ] 5.4 Add unit test: second call without force prints skip and returns 0; force clears state and re-crops
- [ ] 5.5 Add unit test for orphan cleanup: previous run wrote 3 crops; force re-run with 2 regions → only 2 crops remain; `_D03-00_V.jpg` and its sidecar are deleted

## 6. CLI Integration

- [ ] 6.1 Add `run_crop_regions(*, album_id, photos_root, page, force, skip_crops)` to `photoalbums/commands.py`: iterates matching view JPEGs, calls `crop_page_regions` per page; handles per-page errors without aborting; returns non-zero exit if any page failed
- [ ] 6.2 Add `crop-regions` subcommand to `photoalbums.py` CLI
- [ ] 6.3 Add `--skip-crops` flag to `render-pipeline` subcommand; pass through to `run_render_pipeline`
- [ ] 6.4 Wire `crop-regions` step into `run_render_pipeline` between detect-regions and face-refresh; skip if `--skip-crops` set
- [ ] 6.5 Ensure face-refresh step in `run_render_pipeline` iterates both `_View/` and `_Photos/` JPEGs for the page
- [ ] 6.6 Add `photoalbums-crop-regions` recipe to `justfile`

## 7. Tests

- [ ] 7.1 Add integration test for full pipeline with crops: verify `_Photos/` created, correct number of crop files written, each crop sidecar has DocumentID and DerivedFrom pointing to `_V.jpg`, captions resolved correctly
- [ ] 7.2 Add test: `--skip-crops` produces no `_Photos/` files and does not write `pipeline.crop_regions` state
- [ ] 7.3 Add test: face-refresh runs on crop JPEGs and writes face regions to crop sidecars
- [ ] 7.4 Add test: page with only a page-level `dc:description` (no per-region captions) → all crop sidecars inherit the page caption
