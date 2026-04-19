## Why

Crop photo sidecars currently put page OCR text into `dc:description` `x-default` and relegate the Docling region caption to a custom `x-caption` alt-text entry. That makes the wrong text show up by default, drifts from the repo's current `view-xmp-regions` contract, and uses non-standard `dc:description` language tags for OCR and scene layers.

## What Changes

- Make crop sidecars write the Docling region caption as the default visible caption in `dc:description`.
- Stop using custom `dc:description` alt-text entries such as `x-caption`, `x-author`, and `x-scene` for page and crop metadata.
- Keep OCR- and scene-derived text in separate XMP fields instead of letting it run together in the default caption field.
- Introduce `imago:ParentOCRText` for crop sidecars so inherited page OCR remains available without pretending it is crop-local OCR.
- Add a targeted migration and verification flow that rewrites existing sidecars to the new caption/OCR layout without regenerating images.

## Capabilities

### New Capabilities
- `xmp-caption-migration`: verify and migrate existing XMP sidecars from the legacy crop-caption and OCR field layout to the new contract in place.

### Modified Capabilities
- `view-xmp-regions`: change the page and crop XMP contract so crop captions come from `mwg-rs:Name` into `dc:description` `x-default`, page OCR and scene text remain separated, and inherited crop OCR uses `imago:ParentOCRText`.

## Impact

- Affected code: `photoalbums/lib/xmp_sidecar.py`, `photoalbums/lib/ai_photo_crops.py`, crop and XMP tests, plus CLI/command wiring for a migration command.
- Affected metadata: existing page and crop `.xmp` sidecars under the live photo album root.
- Runtime impact: no image regeneration is required for the metadata repair path; rerender remains a fallback only if migration cannot recover a caption source for a given sidecar.
