## 1. Metadata Contract

- [x] 1.1 Define the copy-safe top-level XMP field set that rendered page and crop sidecars must inherit and preserve
- [x] 1.2 Define provenance ownership for rendered JPEG sidecars so `dc:source` keeps archive scan lineage while `xmpMM:DerivedFrom` and `xmpMM:Pantry` capture immediate rendered-parentage

## 2. View and Crop Inheritance

- [x] 2.1 Update page view sidecar flows to preserve inherited top-level metadata from the source page sidecar across initial render and later rewrites
- [x] 2.2 Update crop sidecar creation to inherit the full rendered-page metadata contract, including captions, dates, `LocationCreated`, `LocationShown`, and other copy-safe top-level fields
- [x] 2.3 Ensure crop-sidecar refresh paths can resolve parent page context from rendered-parent linkage rather than only from archive TIF names

## 3. Preservation on Rewrite

- [x] 3.1 Update metadata rewrite paths such as face refresh so they round-trip inherited top-level fields they do not own
- [x] 3.2 Prevent later rewrites from clearing `LocationShown`, `LocationCreated`, captions, dates, OCR text layers, or archive lineage when those values already exist and are not being recomputed

## 4. Verification

- [x] 4.1 Add tests for propagation of captions, subjects, dates, and structured location metadata from source page sidecars into rendered view sidecars
- [x] 4.2 Add tests for propagation and preservation of the same metadata on crop sidecars, including archive-lineage `dc:source` plus rendered-parent `DerivedFrom`
- [x] 4.3 Add tests proving later face-refresh-style rewrites preserve inherited metadata on both views and crops
- [x] 4.4 Run focused validation with `just test`, `just dupes`, `just deadcode`, and `just complexity`
