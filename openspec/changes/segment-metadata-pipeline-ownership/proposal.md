## Why

The metadata pipeline is currently split across several producers and several write paths, but the ownership boundaries between them are mostly implicit. That makes bugs feel like whack-a-mole: Docling, Gemma, `buffalo_l`, YOLO, Nominatim, and the XMP writers each contribute part of the final metadata, yet the code does not expose one clear contract for which system owns which decision and where final page/crop metadata is resolved.

## What Changes

- Define an explicit ownership model for the metadata pipeline, separating geometry, semantic interpretation, people identity, object detection, geocoding, deterministic resolution, and XMP serialization.
- Introduce a canonical metadata-resolution stage that merges outputs from Docling, Gemma, `buffalo_l`, YOLO, and Nominatim into effective page and crop metadata before XMP write.
- Introduce a numbered region-association overlay artifact so Gemma can associate caption/location/date semantics to the exact accepted Docling regions rather than inventing a separate reading-order numbering.
- Change Gemma caption matching to consume the numbered overlay contract directly instead of using coordinate-based scanline ordering.
- Keep `buffalo_l` as the owner of people identity, YOLO as the owner of object detections, Nominatim as the owner of resolved location payloads, and XMP writers as the owner of provenance and serialization.

## Capabilities

### New Capabilities
- `metadata-pipeline-ownership`: Defines which system owns each stage of metadata production and the deterministic resolver contract for effective page/crop metadata.
- `region-association-overlay`: Defines the numbered overlay artifact derived from accepted Docling regions and its use as the authoritative prompt input for region semantic association.

### Modified Capabilities
- `gemma4-caption-matching`: Change caption/location/date association from inferred reading-order numbering to overlay-number-driven region identity.

## Impact

- `photoalbums/lib/ai_view_regions.py` — produce the numbered region-association overlay and pass it into Gemma association.
- `photoalbums/lib/ai_view_region_render.py` — add a prompt-safe overlay mode with only outlines and visible numbers.
- `photoalbums/lib/_caption_matching.py` — remove scanline-sort numbering from the association contract and map results directly from overlay numbers.
- Metadata assembly paths in `ai_index_runner.py`, `ai_photo_crops.py`, `xmp_sidecar.py`, and related helpers — align field ownership with the new resolver boundaries.
- OpenSpec documentation — add an explicit repository contract for which subsystem owns geometry, semantics, people identity, objects, geocoding, effective metadata, and XMP provenance.
