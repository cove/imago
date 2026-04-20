## 1. Region Association Overlay

- [ ] 1.1 Add a prompt-safe overlay rendering mode in `photoalbums/lib/ai_view_region_render.py` that draws only accepted region outlines and visible numbers over the page image
- [ ] 1.2 Update `photoalbums/lib/ai_view_regions.py` to generate and persist the numbered region-association overlay artifact from the accepted Docling regions
- [ ] 1.3 Ensure the overlay numbering maps directly to the accepted region identity used later in memory/XMP, without introducing a second numbering scheme

## 2. Gemma Association Contract

- [ ] 2.1 Update `photoalbums/lib/_caption_matching.py` prompt text and response parsing to use `region-N` keys keyed to the visible overlay numbers
- [ ] 2.2 Remove scanline-sort-based numbering from the Gemma caption-association merge path and replace it with direct overlay-number mapping
- [ ] 2.3 Update the LM Studio call site to feed the numbered region-association overlay artifact into semantic association

## 3. Metadata Ownership And Resolver Boundaries

- [ ] 3.1 Introduce resolver helpers that separate producer-owned facts from effective page/crop metadata assembly
- [ ] 3.2 Encode explicit precedence rules for effective crop location, including region override, region assignment, caption-matched `LocationShown`, and page fallback
- [ ] 3.3 Move final `PersonInImage` assembly behind resolver-owned filtering so location strings are removed before write while `buffalo_l` remains the identity owner
- [ ] 3.4 Audit page and crop write paths in `ai_index_runner.py`, `ai_photo_crops.py`, and related helpers so they consume resolved metadata instead of re-deriving producer precedence locally

## 4. Regression Coverage

- [ ] 4.1 Add tests for prompt-safe overlay generation to prove only outlines and visible numbers are rendered
- [ ] 4.2 Add caption-matching tests proving overlay-number-keyed responses map directly to accepted regions and unknown keys are ignored
- [ ] 4.3 Add resolver tests for multi-location crop precedence and `PersonInImage` filtering against known location strings
- [ ] 4.4 Add end-to-end regression coverage for a multi-photo page showing that Docling region identity, Gemma caption association, `buffalo_l` people identity, and final crop XMP fields remain correctly segmented
