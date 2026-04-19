## Why

Crop outputs in `_Photos/` currently number derived files from `D01-00` upward for each page without checking what derived numbers already exist in `_Archive/`. On `Family_1907-1946_B01_P40`, `_Archive/` already contains `D01` through `D03`, while `_Photos/` reuses `D01` through `D05`, so those files would collide if the album contents were flattened into one directory.

## What Changes

- Make crop output numbering allocate `D##-00` values after the highest `D##` already present for that page in `_Archive/`
- Keep crop numbering sequential within `_Photos/` for each page once the archive-derived offset is chosen
- Add a repair path that detects existing `_Archive/`/`_Photos/` derived-number collisions and renames crop JPEG/XMP pairs to the canonical non-colliding numbers
- Update crop-related provenance and pipeline behavior so repaired crop sidecars still point at the same parent page and keep their metadata intact

## Capabilities

### New Capabilities
- `photo-crop-derived-numbering`: Allocate non-colliding crop-derived numbers from archive ground truth and repair existing crop numbering collisions

### Modified Capabilities

## Impact

- Affected code: `photoalbums/lib/ai_photo_crops.py`, crop-related command wiring in `photoalbums/commands.py` and `photoalbums/cli.py`, naming helpers, and crop-focused tests
- Affected data: existing `_Photos/` crop JPEG/XMP pairs may be renamed during repair so the on-disk schema becomes canonical
- No new external dependency is expected
