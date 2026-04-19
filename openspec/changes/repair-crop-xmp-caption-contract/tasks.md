## 1. XMP contract updates

- [x] 1.1 Split page-side and crop-side `dc:description` write behavior so crop captions own `x-default`
- [x] 1.2 Stop writing legacy custom `dc:description` alt-text entries such as `x-caption`, `x-author`, and `x-scene`
- [x] 1.3 Update page-side `dc:description` formatting to emit labeled `OCR:` and `Scene Text:` sections when both are present
- [x] 1.4 Update crop-side inheritance so page OCR is written as `imago:ParentOCRText` instead of `imago:OCRText`

## 2. Read paths and crop metadata flows

- [x] 2.1 Update crop metadata write paths to keep region captions ahead of inherited page OCR and page descriptions
- [x] 2.2 Update sidecar read and refresh paths to understand `imago:ParentOCRText` for crop-side parent context
- [x] 2.3 Preserve backward-compatible reads for legacy sidecars only long enough to support migration and verification

## 3. Caption-layout migration

- [x] 3.1 Add a targeted verification flow that reports sidecars still using the legacy caption-layout contract
- [x] 3.2 Implement an in-place migration for crop and page sidecars that rewrites legacy caption/OCR fields without regenerating image files
- [x] 3.3 Resolve migrated crop captions from parent `mwg-rs:Name` first, then legacy crop caption data, with page description as final fallback
- [x] 3.4 Run the migration against `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums` and verify no legacy caption-layout entries remain

## 4. Tests and validation

- [x] 4.1 Update XMP writer tests to cover the new page summary layout and the removal of custom `dc:description` alt-text entries
- [x] 4.2 Update crop-sidecar tests to assert caption-in-`x-default`, `imago:ParentOCRText`, and no page-OCR override of crop captions
- [x] 4.3 Add migration tests for verification mode, in-place rewrite behavior, and preservation of unrelated XMP fields
- [x] 4.4 Run `uv run python -m py_compile` on changed Python modules
- [x] 4.5 Run `just test`
- [x] 4.6 Run `just dupes`
- [x] 4.7 Run `just deadcode`
- [x] 4.8 Run `just complexity`
