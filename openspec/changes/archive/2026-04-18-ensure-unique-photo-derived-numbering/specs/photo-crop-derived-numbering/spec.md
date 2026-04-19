## ADDED Requirements

### Requirement: Crop derived numbers start after the highest archive-derived number for the page
The system SHALL allocate `_Photos/` crop filenames from the next available `D##` after the highest derived number already present for the same page in `_Archive/`. Crop outputs for that page SHALL remain in a contiguous sequence using the `-00` iteration slot.

#### Scenario: Archive-derived images shift crop numbering upward
- **WHEN** page `Family_1907-1946_B01_P40` has archive-derived files with `D01`, `D02`, and `D03`
- **THEN** the first crop written to `_Photos/` for that page is `Family_1907-1946_B01_P40_D04-00_V.jpg`
- **AND** later crops for the same page continue as `D05-00`, `D06-00`, and so on

#### Scenario: Crop numbering starts at D01 when the archive page has no derived files
- **WHEN** a page has no `_Archive/` files matching `_P##_D##-##`
- **THEN** the first crop written to `_Photos/` for that page is `_D01-00_V.jpg`

### Requirement: Crop numbering is deterministic from archive ground truth
The system SHALL derive crop output numbers from archive-derived filenames for the page rather than from existing `_Photos/` filenames. Re-running crop generation for an unchanged page SHALL target the same canonical crop filenames.

#### Scenario: Existing non-canonical crop names do not move the allocation window
- **WHEN** `_Archive/` ends at `D03` for a page and `_Photos/` already contains stale crops named `D01-00` through `D05-00`
- **THEN** a canonical crop output path calculation for that page still begins at `D04-00`

### Requirement: Repair renames existing crop outputs into the canonical non-colliding sequence
The system SHALL provide a repair operation that scans existing `_Photos/` crop JPEG/XMP pairs by page, detects overlap with `_Archive/` derived numbers, and renames the crop outputs into the canonical contiguous range after the archive-derived maximum.

#### Scenario: Repair fixes the P40 collision set
- **WHEN** `_Archive/` for `Family_1907-1946_B01_P40` contains `D01` through `D03` and `_Photos/` contains five crop pairs named `D01-00` through `D05-00`
- **THEN** repair renames the five crop pairs to `D04-00` through `D08-00`
- **AND** no `_Photos/` crop pair for that page remains in `D01-00`, `D02-00`, or `D03-00`

#### Scenario: Repair leaves already-canonical crop numbers unchanged
- **WHEN** `_Archive/` ends at `D03` for a page and `_Photos/` already contains crop pairs `D04-00` and `D05-00`
- **THEN** repair reports no rename for that page

### Requirement: Repair preserves crop pair metadata while renaming
The repair operation SHALL rename the crop JPEG and its `.xmp` sidecar together as one logical unit. It SHALL preserve the existing JPEG pixels, XMP contents, and pipeline state stored in the crop sidecar.

#### Scenario: Repair keeps crop sidecar provenance intact
- **WHEN** a crop pair is renamed from `_D01-00_V.jpg` to `_D04-00_V.jpg`
- **THEN** the renamed `.xmp` sidecar retains its existing `xmpMM:DocumentID`, `xmpMM:DerivedFrom`, and pipeline metadata

#### Scenario: Repair fails on incomplete crop pairs
- **WHEN** a selected crop JPEG exists without its matching `.xmp` sidecar, or vice versa
- **THEN** the repair operation fails for that page instead of silently dropping the unmatched file
