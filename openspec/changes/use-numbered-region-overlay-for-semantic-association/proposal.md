## Why

Gemma caption and location association currently depends on an implicit reading-order contract: the model is asked to number photos left-to-right/top-to-bottom, while the code separately sorts Docling regions into that same order. That is brittle, hard to debug, and fails exactly where this pipeline most needs precision: telling Gemma which detected photo region Docling actually meant.

## What Changes

- Introduce a dedicated numbered region overlay artifact for semantic association, generated from the accepted Docling regions and rendered over the page image with only precise outlines and visible region numbers.
- Change Gemma caption/location association to use that numbered overlay as the authoritative region-identity map instead of asking the model to invent its own reading-order numbering.
- Replace coordinate-based region reordering in caption matching with direct mapping from Gemma response keys to the visible overlay numbers.
- Extend the semantic-association contract so captions, location choices, and related page-to-region associations are keyed to the accepted Docling regions directly.
- Keep geometry ownership in Docling and keep deterministic write-time concerns such as provenance, inheritance, geocoding, and final XMP field assembly in code.

## Capabilities

### New Capabilities
- `region-association-overlay`: Defines the numbered overlay artifact derived from accepted Docling regions and the contract for using it as prompt input during semantic association.

### Modified Capabilities
- `gemma4-caption-matching`: Change caption/location association from inferred reading-order numbering to overlay-number-driven region identity.

## Impact

- `photoalbums/lib/ai_view_regions.py` — generate the numbered overlay artifact from accepted regions and feed it into the Gemma association step.
- `photoalbums/lib/ai_view_region_render.py` — support a prompt-safe overlay mode with only outlines and visible numbers.
- `photoalbums/lib/_caption_matching.py` — replace scanline sort numbering with overlay-number-based response parsing and prompt contract.
- Prompt/debug artifacts — the numbered overlay becomes a first-class pipeline input rather than a debug-only byproduct.
