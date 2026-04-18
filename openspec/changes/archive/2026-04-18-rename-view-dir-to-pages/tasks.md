## 1. Runtime directory rename

- [x] 1.1 Add a lightweight shared naming helper for canonical album directory suffixes and sibling derivation across `_Archive`, `_Pages`, and `_Photos`
- [x] 1.2 Update shared path helpers in `photoalbums` so rendered page siblings derive as `_Pages` instead of `_View`
- [x] 1.3 Replace direct `_View` suffix checks, glob defaults, and string replacements in `photoalbums` runtime code with `_Pages`, using the shared helper where practical
- [x] 1.4 Update `cast` ingest/server/UI defaults so rendered page folder scans use `*_Pages`

## 2. XMP write-path updates

- [x] 2.1 Update XMP/provenance helpers that write page-relative references so new sidecars encode `*_Pages`
- [x] 2.2 Add or update tests that assert crop/page XMP references use `_Pages` and never `_View`

## 3. Existing XMP migration

- [x] 3.1 Implement a targeted migration for album `.xmp` sidecars that rewrites page references from `*_View` to `*_Pages` without changing unrelated fields
- [x] 3.2 Run the migration against `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums`
- [x] 3.3 Verify that no `.xmp` files under the album root still contain `_View`

## 4. Repo-tracked references

- [x] 4.1 Update docs, help text, and examples that currently mention `*_View`
- [x] 4.2 Update repo-tracked eval fixtures and test fixtures that encode `_View` paths
- [x] 4.3 Update or add tests for `_Archive`/`_Pages`/`_Photos` sibling resolution and Cast ingest defaults

## 5. Validation

- [x] 5.1 Run `uv run python -m py_compile` on changed Python modules
- [x] 5.2 Run `just test`
- [x] 5.3 Run `just dupes`
- [x] 5.4 Run `just deadcode`
- [x] 5.5 Run `just complexity`
