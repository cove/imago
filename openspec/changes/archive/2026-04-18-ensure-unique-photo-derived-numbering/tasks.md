## 1. Crop numbering allocation

- [x] 1.1 Add a helper in the crop/naming path that scans page-matching `_Archive/` derived filenames and returns the highest existing `D##` for that page
- [x] 1.2 Update crop output path generation so `_Photos/` crops use `archive_max + region_index` and still write the `-00` iteration slot
- [x] 1.3 Update crop force/rerun handling so expected crop outputs are computed from the canonical archive-based numbering rather than the legacy `_D01-00...` sequence

## 2. Existing crop repair

- [x] 2.1 Add a library repair function that groups `_Photos/` crop JPEG/XMP pairs by page, validates complete pairs, and computes the canonical target sequence after the page's archive-derived max
- [x] 2.2 Implement safe two-phase renaming through temporary filenames so pages with existing target names can be repaired without overwriting files
- [x] 2.3 Add a CLI/command entry point for crop-number repair with album/page targeting and machine-readable reporting of renamed paths

## 3. Tests and validation

- [x] 3.1 Add unit tests for archive-based crop numbering on pages with no archive-derived files and with pre-existing archive-derived files
- [x] 3.2 Add repair tests covering the `Family_1907-1946_B01_P40` collision shape, already-canonical pages, and incomplete crop-pair failures
- [x] 3.3 Run `uv run python -m py_compile` on changed Python modules and run `just test`, `just dupes`, `just deadcode`, and `just complexity`
