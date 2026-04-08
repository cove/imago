## Why

Rendered photoalbum outputs can inherit stale person-identifying XMP regions from an archive sidecar copy, even though the rendered image is the file users actually review and export. We need render-time outputs to refresh those person regions against the rendered pixels so the names and boxes come from a fresh `buffalo_l` Cast-backed face-identification pass.

## What Changes

- Refresh person-identifying metadata for rendered view outputs during the render process instead of leaving inherited face regions in place.
- Re-run `buffalo_l` face identification against the rendered image and use the Cast database to supply matched names.
- Replace any existing person-identifying XMP image regions on the rendered output with the refreshed face matches from that render-time pass.
- Preserve non-person render metadata already copied or produced during render, including page OCR, source references, and other non-face regions.

## Capabilities

### New Capabilities
- `render-face-region-refresh`: Refresh person-identifying metadata on rendered outputs by rerunning Cast-backed `buffalo_l` face matching on the rendered image and replacing face regions with the fresh results.

### Modified Capabilities

## Impact

- Affected render pipeline code in `photoalbums/stitch_oversized_pages.py`.
- Likely reuse or extension of the people-refresh logic in `photoalbums/lib/ai_index_runner.py`.
- XMP face-region output behavior in `photoalbums/lib/xmp_sidecar.py`.
- Test coverage for rendered outputs and people-region replacement behavior.
