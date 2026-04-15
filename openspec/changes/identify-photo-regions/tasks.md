## 9b. Switch Docling Config to the Local Library Path

- [ ] 9b.1 Update `photoalbums/ai_models.toml` so the docling section only carries the preset needed by the library path; document that the preset must resolve from local model assets without a runtime fetch.
- [ ] 9b.2 Remove `default_docling_model_id()` and the `docling_model_id` setting from `photoalbums/lib/ai_model_settings.py` if nothing else uses it.
- [ ] 9b.3 Update any tests or config fixtures that still expect a Docling LM Studio model override or a Hugging Face-backed runtime fetch.

## 9c. Update the Docling Pipeline Wrapper

- [ ] 9c.1 Change `photoalbums/lib/_docling_pipeline.py` to call `VlmConvertOptions.from_preset(preset)` without `ApiVlmEngineOptions`.
- [ ] 9c.2 Build `VlmPipelineOptions` for the local Docling path and keep `DocumentConverter` configured for `InputFormat.IMAGE`.
- [ ] 9c.3 Preserve picture-item iteration, bbox-to-pixel conversion, and caption extraction from the `DoclingDocument`.
- [ ] 9c.4 Keep the empty-document warning path and return `[]` when no picture items are found.
- [ ] 9c.5 Make the Docling branch fail fast with the underlying model-resolution error if the local model assets are missing, rather than falling back to LM Studio or a remote fetch.

## 9d. Wire the New Path Through Callers and Tests

- [ ] 9d.1 Update `_detect_regions_docling()` in `photoalbums/lib/ai_view_regions.py` to stop passing `base_url` and `model_id` into the Docling pipeline.
- [ ] 9d.2 Keep the existing `validate_regions()` behavior and pipeline-step outcomes unchanged.
- [ ] 9d.3 Update `photoalbums/tests/test_docling_pipeline.py` to assert the local Docling configuration and remove LM Studio-specific expectations.
- [ ] 9d.4 Update `photoalbums/tests/test_ai_view_regions.py` to cover the new Docling call signature.
- [ ] 9d.5 Run the targeted Docling and region-detection tests, then the full test suite if the targeted run passes.

## 9e. Keep Crop Boundaries Sourced from XMP

- [ ] 9e.1 Update the spec and/or crop tests so `crop_page_regions()` is explicitly treated as reading `mwg-rs:RegionList` from the page view XMP sidecar.
- [ ] 9e.2 Verify the crop step converts those stored regions into pixel rectangles and crops from the page `_V.jpg`.
- [ ] 9e.3 Keep the crop step as the source of truth for crop boundaries, not the Docling pipeline output in memory.

## 9f. Publish Page OCR and Scene Text in Top-Level dc:description

- [ ] 9f.1 Update the page view XMP writer so the top-level `dc:description` combines the page OCR text and Gemma scene text into a human-readable summary.
- [ ] 9f.2 Keep the raw `imago:OCRText` and `imago:SceneText` fields populated separately for indexing and debugging.
- [ ] 9f.3 Ensure derived `_D##-##_V.jpg` outputs use the same top-level `dc:description` composition rule as the page view output.
- [ ] 9f.4 Add crop-side OCR so `_Photos/_D##-00_V.jpg` outputs write the crop's own OCR text into `imago:OCRText`.
- [ ] 9f.5 Ensure `_Photos/_D##-00_V.jpg` crop outputs set top-level `dc:description` to the region caption plus crop-local OCR text, caption first with a blank line between when both are present.
- [ ] 9f.6 Add or update tests to verify that page-side, derived-view, and crop-side XMP all surface the right text in `dc:description` while preserving the raw text fields.
