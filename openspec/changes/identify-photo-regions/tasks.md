## 9b. Switch Docling Config to the Local Library Path

- [ ] 9b.1 Update `photoalbums/ai_models.toml` and `photoalbums/lib/ai_model_settings.py` so region detection is Docling-only and the Docling config carries the preset/backend settings needed by the local library path.
- [ ] 9b.2 Document in config/tests that the selected Docling preset must resolve from local model assets on Windows without a runtime fetch, using automatic GPU/CPU selection.
- [ ] 9b.3 Remove tests and fixtures that still expect Gemma or LM Studio to be part of page photo region detection.

## 9c. Update the Docling Pipeline Wrapper

- [ ] 9c.1 Change `photoalbums/lib/_docling_pipeline.py` to call `VlmConvertOptions.from_preset(preset)` without `ApiVlmEngineOptions`.
- [ ] 9c.2 Build `VlmPipelineOptions` for the local Docling path, keep `DocumentConverter` configured for `InputFormat.IMAGE`, and use Windows auto GPU/CPU backend selection.
- [ ] 9c.3 Preserve picture-item iteration, bbox-to-pixel conversion, and caption extraction from the `DoclingDocument`.
- [ ] 9c.4 Keep the empty-document warning path and return `[]` when no picture items are found.
- [ ] 9c.5 Retry Docling runtime failures a bounded number of times, then preserve the underlying error and surface the terminal pipeline result as `failed`.
- [ ] 9c.6 Export the raw Docling result to a JSON-serializable debug payload for later artifact writing, without making that payload part of the XMP contract.

## 9d. Wire the New Path Through Callers and Tests

- [ ] 9d.1 Remove the Gemma/LM Studio region-detection path from `photoalbums/lib/ai_view_regions.py` so page photo region detection only uses the Docling library path.
- [ ] 9d.2 Keep `validate_regions()` as the acceptance gate, but distinguish runtime `failed` from `validation_failed` and do not retry validation failures.
- [ ] 9d.3 When `--debug` view-region output is enabled, write the raw Docling JSON to the per-image `_debug` area as a separate artifact instead of embedding it in XMP.
- [ ] 9d.4 Update `photoalbums/tests/test_docling_pipeline.py` to assert the local Docling configuration, Windows auto backend selection, runtime retry behavior, and exported debug payload shape.
- [ ] 9d.5 Update `photoalbums/tests/test_ai_view_regions.py` to cover the Docling-only call path, `failed` pipeline state, and debug-artifact write path.

## 9e. Keep Crop Boundaries Sourced from XMP

- [ ] 9e.1 Update the spec and/or crop tests so `crop_page_regions()` is explicitly treated as reading `mwg-rs:RegionList` from the page view XMP sidecar.
- [ ] 9e.2 Verify the crop step converts those stored regions into pixel rectangles and crops from the page `_V.jpg`.
- [ ] 9e.3 Store per-region captions in `mwg-rs:Name` and update crop-side caption resolution to read `mwg-rs:Name` as the primary caption field.
- [ ] 9e.4 Keep the crop step as the source of truth for crop boundaries, not the Docling pipeline output in memory.

## 9f. Keep Page dc:description Searchable

- [ ] 9f.1 Update the page view XMP writer so the top-level `dc:description` combines the page OCR text and Gemma scene text into a human-readable summary.
- [ ] 9f.2 Keep the raw `imago:OCRText` and `imago:SceneText` fields populated separately for indexing and debugging.
- [ ] 9f.3 Ensure the page-level `dc:description` remains page-oriented searchable text and is not reused as the per-region caption store.
- [ ] 9f.4 Add or update tests to verify that page-side XMP surfaces OCR/scene text in top-level `dc:description` while per-region captions live in `mwg-rs:Name`.

## 9g. Verification

- [ ] 9g.1 Run focused region-detection and Docling tests with `just test`.
- [ ] 9g.2 Run `just dupes`.
- [ ] 9g.3 Run `just deadcode`.
- [ ] 9g.4 Run `just complexity`.
