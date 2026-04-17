## Why

Photos from the 1960s–1970s have degraded color (lost cyan channel), and earlier photos are black-and-white — both are candidates for AI-based restoration and colorization. Applying restoration to individual cropped photos rather than full album pages increases quality since restoration models are trained on single-image datasets.

## What Changes

- Add a new `photo-restoration` library module that wraps RealRestorer for local inference
- Integrate restoration into the render pipeline so it runs automatically after cropping writes `_Photo/` images
- Restored images overwrite the cropped photo in-place (no backup kept)
- Hardware detection runs automatically (CUDA if available, CPU fallback) — no manual configuration needed

## Capabilities

### New Capabilities
- `photo-restoration`: Wraps the RealRestorer model to restore/colorize individual cropped photos, invoked as a pipeline step after crop output is written

### Modified Capabilities
- `photoalbum-rendering`: Add restoration as a post-crop pipeline step applied to `_Photo/` images by default

## Impact

- New dependency: RealRestorer (Python, local inference via PyTorch with auto hardware detection)
- Affected code: render pipeline (wherever `_Photo/` crop images are written), new `photoalbums/lib/photo_restoration.py` module
- No new API surface beyond the internal pipeline hook
- Cropped images are modified in-place; downstream steps see the restored version
