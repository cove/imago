## 1. Model & Config

- [ ] 1.1 Add a Gemma 4 CTM model entry to photoalbum AI model config (LM Studio / `ai_models.toml`)
- [ ] 1.2 Expose CTM model settings through the photoalbum model-settings layer
- [ ] 1.3 Add configurable validation thresholds for CTM sanity checks (confidence, clipping, coefficient bounds)

## 2. CTM Estimation Pipeline

- [ ] 2.1 Create a new photoalbum CTM library module (for example `photoalbums/lib/ai_ctm_restoration.py`)
- [ ] 2.2 Implement a Gemma 4 vision/text prompt that requests structured JSON CTM output from a stitched image
- [ ] 2.3 Add JSON parsing + retry logic for malformed model output
- [ ] 2.4 Add a `CTMResult` data structure with matrix coefficients, confidence, warnings, and provenance metadata
- [ ] 2.5 Add sanity validation for returned CTMs (9 finite values, coefficient bounds, clipping heuristics)

## 3. XMP Manifest Support

- [ ] 3.1 Register `crs`, `archive`, and any required XMP manifest namespaces in XMP helper code
- [ ] 3.2 Add XMP helpers to read/write `crs:ColorMatrix1`
- [ ] 3.3 Add XMP helpers to write/read the requested manifest structure (`xmpMM:DocumentID`, `crs:HasSettings`, optional stitch ingredients and homography metadata)
- [ ] 3.4 Ensure CTM metadata coexists with existing photoalbum XMP fields without destructive replacement

## 4. Render-Time Application

- [ ] 4.1 Implement deterministic 3×3 CTM application in the photoalbum render pipeline
- [ ] 4.2 Apply CTM correction before final stitched/rendered image output is written
- [ ] 4.3 Add handling for missing, invalid, or low-confidence CTMs (skip with warning or require explicit force)

## 5. CLI & MCP Jobs

- [ ] 5.1 Add a CLI command to generate CTMs for a page, album, or album set
- [ ] 5.2 Add an MCP tool to enqueue CTM-generation jobs via `JobRunner`
- [ ] 5.3 Add an MCP/CLI inspection path to review stored CTM values, confidence, and warnings
- [ ] 5.4 Support rerun / overwrite semantics with `force=True`

## 6. Tests

- [ ] 6.1 Add unit tests for CTM JSON parsing and retry behavior
- [ ] 6.2 Add unit tests for matrix validation and clipping heuristics
- [ ] 6.3 Add integration tests for XMP manifest write/read of `crs:ColorMatrix1`
- [ ] 6.4 Add render-path tests verifying CTM application changes output pixels deterministically
- [ ] 6.5 Add MCP/CLI tests for CTM job creation and review behavior

## 7. Documentation / Spec Hygiene

- [ ] 7.1 Add capability spec text for `photoalbum-ctm-restoration`
- [ ] 7.2 Validate the OpenSpec change with `openspec validate add-photoalbum-ctm-restoration`
