## 1. Setup

- [x] 1.1 Follow the RealRestorer README to clone the repo, install the custom diffusers fork, and verify `from diffusers import RealRestorerPipeline` works
- [ ] 1.2 Confirm model weights download automatically on first run via `from_pretrained`

## 2. photo_restoration Module

- [x] 2.1 Create `photoalbums/lib/photo_restoration.py` with a `restore_photo(image: Image.Image) -> Image.Image` function that returns the restored PIL Image (or the original on failure)
- [x] 2.2 Lazy-import `RealRestorerPipeline` at first call time; catch `ImportError` and log a one-time warning, returning the original image for all subsequent calls
- [x] 2.3 Load `RealRestorerPipeline.from_pretrained("RealRestorer/RealRestorer", torch_dtype=torch.bfloat16)` and call `pipe.enable_model_cpu_offload()` - cache the pipeline object as a module-level singleton so it is only loaded once per process
- [x] 2.4 Choose restoration prompt: use `"Please restore this low-quality image, recovering its normal brightness and clarity."` as the default; leave a clear comment identifying the prompt as tunable
- [x] 2.5 Run inference: `pipe(image=image, prompt=..., num_inference_steps=28, guidance_scale=3.0, seed=42, size_level=1024)` and return `result.images[0]`
- [x] 2.6 Wrap inference in a try/except; log warning with error on failure and return the original image unchanged

## 3. Hook into crop pipeline

- [x] 3.1 Add `skip_restoration: bool = False` parameter to `crop_page_regions()` in `ai_photo_crops.py`
- [x] 3.2 Before `crop_img.save(output_path, ...)`, call `crop_img = restore_photo(crop_img)` unless `skip_restoration` is True, then save the (possibly restored) image - a single JPEG write with no reload
- [x] 3.3 Verify that a restoration failure for one crop does not abort the loop (failure returns original image from inside `restore_photo`)

## 4. CLI flag

- [x] 4.1 Add `--skip-restoration` flag to the `crop-regions` CLI command in `commands.py`; wire it to `skip_restoration=True` in the `crop_page_regions()` call

## 5. Tests

- [x] 5.1 Write a unit test for `restore_photo` that mocks `RealRestorerPipeline` and asserts it returns the restored PIL Image from `result.images[0]`
- [x] 5.2 Write a test for the `ImportError` path: patch the import to raise, assert `restore_photo` returns the original image unchanged and does not raise
- [x] 5.3 Write a test for inference failure: patch `pipe(...)` to raise, assert the original image is returned unchanged
- [x] 5.4 Write a test verifying `crop_page_regions()` passes the in-memory crop PIL Image to `restore_photo` before saving when `skip_restoration=False`
- [x] 5.5 Write a test verifying `crop_page_regions()` does not call `restore_photo` when `skip_restoration=True`
