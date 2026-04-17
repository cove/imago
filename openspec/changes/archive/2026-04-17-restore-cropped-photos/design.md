## Context

Cropped photo JPEGs are written by `crop_page_regions()` in `photoalbums/lib/ai_photo_crops.py`. After `crop_img.save(output_path, ...)` at line ~371, the crop is committed to disk and the sidecar is written. Photos from the 1960s–1970s commonly have cyan channel loss; earlier photos are black-and-white and are candidates for colorization. Applying restoration at the individual-crop level (rather than full-page scans) maximizes model quality since restoration models are trained on single-photo datasets.

RealRestorer (https://huggingface.co/RealRestorer/RealRestorer) is a diffusion-based image restoration pipeline built on a custom `diffusers` fork. It exposes a `RealRestorerPipeline` class that accepts an image and a text prompt describing the restoration goal. CPU offloading is handled by `pipe.enable_model_cpu_offload()`, which moves model layers to CPU between inference steps — this works without CUDA.

### Installation (for implementer — not project docs)
```bash
git clone https://github.com/yfyang007/RealRestorer.git
cd RealRestorer
cd diffusers && python -m pip install -e . && cd ..
python -m pip install -r requirements.txt
python -m pip install -e .
```
Model weights download automatically via `from_pretrained` on first use — no manual step needed.

### Python API
```python
import torch
from PIL import Image
from diffusers import RealRestorerPipeline

pipe = RealRestorerPipeline.from_pretrained("RealRestorer/RealRestorer", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # works without CUDA

# Image is passed and returned as a PIL Image — no disk round-trip needed
image = Image.open("crop.jpg").convert("RGB")
result = pipe(image=image, prompt="Restore the colors and make it look natural.",
              num_inference_steps=28, guidance_scale=3.0, seed=42, size_level=1024)
restored: Image.Image = result.images[0]
restored.save("crop.jpg")  # single write to disk
```

Relevant prompts:
- Faded/cyan-loss color photos: `"Restore the colors and make it look natural."`
- Black-and-white photos: `"Please colorize this black and white photograph."`
- General restoration fallback: `"Please restore this low-quality image, recovering its normal brightness and clarity."`

## Goals / Non-Goals

**Goals:**
- Add a `photoalbums/lib/photo_restoration.py` module that wraps RealRestorer inference
- Call it with the in-memory PIL Image before the crop is saved, so the restored image is written once
- The restored PIL Image is saved as the final JPEG — no double encode/decode
- Auto-detect hardware (CUDA → MPS → CPU) with no manual configuration
- Degrade gracefully: if RealRestorer is not installed or inference fails, log a warning and leave the original crop unchanged

**Non-Goals:**
- Restoring full page scans or stitched images
- Keeping a backup of the pre-restoration crop
- Exposing restoration as a standalone CLI command (this change hooks it into the crop pipeline only)
- Training or fine-tuning the model

## Decisions

### D1: Hook point — before `crop_img.save()`, replacing the image written to disk
`restore_photo(image: Image.Image) -> Image.Image` accepts the in-memory PIL crop and returns the restored PIL Image. `crop_page_regions()` calls it before saving, then saves the returned image. This avoids a JPEG encode → decode → re-encode cycle: the crop is encoded to JPEG exactly once, as the final output.

On failure (RealRestorer absent or inference error), `restore_photo` returns the original image unchanged, so the save path is identical either way.

### D2: In-place overwrite, no backup
The original crop is a derived product (cropped from the page scan). If restoration produces a bad result the operator can delete the crop and re-run `crop-regions --force` to regenerate from the archive scan. Keeping a backup adds storage overhead with little value.

### D3: Graceful degradation — restoration is best-effort
If the RealRestorer package is absent, the import fails silently and a one-time warning is logged. If inference throws, the warning is logged and the original crop is preserved. This lets the pipeline run without RealRestorer installed.

**Alternative considered**: hard dependency via `uv add`. Rejected because RealRestorer requires downloading model weights and is not yet on PyPI as a clean package — installation is manual.

### D4: Hardware via `enable_model_cpu_offload()` — no manual device selection needed
`RealRestorerPipeline.enable_model_cpu_offload()` handles the CUDA/CPU split automatically. On machines without CUDA it runs fully on CPU. No `REAL_RESTORER_PATH` or manual device flag is needed.

### D5: RealRestorer installed as a custom diffusers fork, imported directly
The repo provides a custom `diffusers` fork that exposes `RealRestorerPipeline`. Installation is manual (clone + pip install). The module path is added to `sys.path` if needed, but since `pip install -e .` is run in the RealRestorer dir, `from diffusers import RealRestorerPipeline` should work globally after installation. Absence is detected by catching `ImportError` at lazy import time.

## Risks / Trade-offs

- **Model weight download**: RealRestorer requires manually downloading weights. If weights are missing, inference will fail. Mitigation: log a clear error message pointing to the RealRestorer README.
- **CPU inference is slow**: On CPU, restoring a single crop may take 30–120 seconds depending on resolution. Mitigation: document this; users on CPU may want to disable restoration for bulk runs. Add a `--skip-restoration` flag to `crop-regions`.
- **Research-repo API instability**: RealRestorer's inference interface may change. Mitigation: pin to a specific commit hash in documentation; isolate all RealRestorer calls inside `photo_restoration.py`.
- **Color artifacts**: Restoration may introduce incorrect colors on some images. Mitigation: in-place overwrite is recoverable by re-running `crop-regions --force --skip-restoration`.

## Migration Plan

1. Clone RealRestorer and run its install steps (custom diffusers fork + requirements)
2. Model weights download automatically from HuggingFace Hub on first inference
3. Run `uv sync` (no new Python package dependencies beyond what RealRestorer installs)
4. Existing crops are not retroactively restored; re-run `crop-regions --force` on albums that need restoration

## Open Questions

- Should restoration be skippable per-album via a config flag (e.g., in `ai_models.toml`) rather than only via CLI flag? Deferred — CLI flag is sufficient for now.
- Does RealRestorer handle already-color images gracefully (i.e., 1980s+ photos that don't need restoration)? Needs testing — if it degrades good-quality color photos, we may need a year-range gate.
