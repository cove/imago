## Requirements

### Requirement: Restore a cropped photo image using RealRestorer
Imago MUST provide a `photo_restoration` module with a `restore_photo(image: Image.Image) -> Image.Image` function that accepts an in-memory PIL Image, runs RealRestorer inference, and returns the restored PIL Image. The caller is responsible for saving the result; no disk I/O is performed inside `restore_photo`. Hardware selection is handled automatically by `enable_model_cpu_offload()`.

#### Scenario: Restore a color-degraded crop
- **WHEN** `restore_photo(image)` is called with a PIL Image
- **THEN** the module runs RealRestorer inference on the image
- **AND** returns the restored PIL Image
- **AND** does not write any files to disk

#### Scenario: RealRestorer package is not installed
- **WHEN** `restore_photo(image)` is called and the RealRestorer package cannot be imported
- **THEN** the module logs a one-time warning identifying that RealRestorer is unavailable
- **AND** returns the original image unchanged
- **AND** does not raise an exception

#### Scenario: RealRestorer inference raises an exception
- **WHEN** `restore_photo(image)` is called and RealRestorer inference throws an error
- **THEN** the module logs a warning with the error details
- **AND** returns the original image unchanged

#### Scenario: Hardware offload handles CPU-only machines
- **WHEN** `restore_photo(image)` is called on a machine without CUDA or MPS
- **THEN** inference runs successfully via `enable_model_cpu_offload()` on CPU

### Requirement: Detect RealRestorer availability via import
The module SHALL detect RealRestorer availability by attempting to import `RealRestorerPipeline` from `diffusers`. If the import fails, `restore_photo` SHALL behave as a pass-through (returning the original image) for the remainder of the process, logging a one-time warning on the first call.

#### Scenario: RealRestorer diffusers fork is installed
- **WHEN** `from diffusers import RealRestorerPipeline` succeeds
- **THEN** the pipeline is loaded lazily on the first `restore_photo` call and cached for subsequent calls

#### Scenario: RealRestorer diffusers fork is not installed
- **WHEN** `from diffusers import RealRestorerPipeline` raises `ImportError`
- **THEN** the module logs a one-time warning explaining that RealRestorer is not installed
- **AND** all subsequent `restore_photo` calls return the original image unchanged
