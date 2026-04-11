## 1. Render-Time People Refresh Entry Point

- [x] 1.1 Add a focused render-time helper that reuses the existing Cast-backed `buffalo_l` people-refresh logic against a rendered output image and sidecar.
- [x] 1.2 Update the refresh flow so rendered-output `PersonInImage` names come only from the fresh rendered-image Cast result instead of unioning inherited copied names with images that have a diffrent geometry.

## 2. Face-Region Replacement Behavior

- [x] 2.1 Add targeted XMP update logic that removes only existing face-identifying `ImageRegion` entries and writes refreshed face regions without deleting non-face regions.
- [x] 2.2 Wire the render pipeline to run the people refresh after single-scan page renders, stitched page renders, and derived JPEG renders once the rendered sidecar exists.

## 3. Verification

- [x] 3.1 Add or update render tests to cover render-time people refresh for copied page sidecars and derived rendered outputs.
- [x] 3.2 Add or update XMP/index tests to verify stale inherited person regions and names are removed while non-face metadata and non-face image regions are preserved.
