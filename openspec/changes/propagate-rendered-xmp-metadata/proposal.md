## Why

Rendered view and photo crop sidecars are currently losing top-level XMP that already exists on the page source sidecars. In practice this means captions, subjects, dates, provenance context, and structured location metadata such as `Iptc4xmpExt:LocationShown` and the details within can disappear somewhere between the archive page sidecar, the rendered page `_V.jpg` sidecar, the crop `_D##-00_V.jpg` sidecar, and later metadata refresh steps.

The archived `photo-region-crops` change already expected page-level location/date/subject inheritance, but the current behavior is narrower and later rewrites can clear inherited metadata that was not being actively recomputed. We need one explicit rule set for what top-level metadata is inherited, what provenance fields belong to each rendered output, and what later rewrite steps must preserve.

## What Changes

- Define a canonical rendered-sidecar metadata inheritance contract for page `_V.jpg` outputs and crop `_D##-00_V.jpg` outputs
- Require copy-safe top-level descriptive, temporal, OCR/text-layer, and location metadata from the source page sidecar to propagate to rendered outputs when present
- Clarify provenance ownership so `dc:source` keeps archive scan lineage while `xmpMM:DerivedFrom` and `xmpMM:Pantry` capture immediate rendered-parent relationships
- Require later sidecar rewrite steps such as face refresh and similar metadata updates to preserve inherited top-level metadata unless that step intentionally replaces a specific field
- Require crop-sidecar refresh paths to resolve parent page context from rendered-parent linkage rather than assuming all upstream context can be recovered only from archive TIF names

## Impact

- New change clarifies behavior across `page-stitch-render`, `photo-region-crops`, and `render-face-region-refresh`
- `xmp_sidecar` read/write behavior will need to round-trip more inherited top-level fields instead of clearing them opportunistically
- Crop-sidecar creation and later refresh flows will need more explicit parent-view metadata resolution
- Tests will need coverage for `LocationShown`, `LocationCreated`, captions, subjects, dates, OCR text layers, and rendered provenance on both views and crops
