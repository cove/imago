## 1. Model & Configuration

- [ ] 1.1 Add a Gemma 4 CTM model entry to photoalbum AI model configuration
- [ ] 1.2 Expose CTM model settings through the photoalbum model-settings layer
- [ ] 1.3 Add configurable validation thresholds for CTM sanity checks and warnings

## 2. CTM Estimation Pipeline

- [ ] 2.1 Create a photoalbum CTM restoration module for model prompting, parsing, and validation
- [ ] 2.2 Implement the Gemma 4 prompt and LM Studio request flow for stitched-image CTM generation
- [ ] 2.3 Add structured JSON parsing and retry handling for malformed model responses
- [ ] 2.4 Add a CTM result structure including matrix values, confidence, warnings, and provenance
- [ ] 2.5 Add numerical validation and preview sanity checks for returned CTMs

## 3. `_Archive/` XMP Support

- [ ] 3.1 Register the Camera Raw namespaces required for CTM metadata in `_Archive/` XMP handling
- [ ] 3.2 Add XMP read/write helpers for `crs:ColorMatrix1` and required compatibility fields in `_Archive/` XMP
- [ ] 3.3 Ensure CTM metadata is written only to `_Archive/` XMP files
- [ ] 3.4 Ensure `_Archive/` CTM metadata remains cleanly separated from existing view-image region metadata

## 4. Post-Stitch Integration

- [ ] 4.1 Implement deterministic 3×3 CTM application in the photoalbum pipeline after stitching
- [ ] 4.2 Apply stored CTMs after stitching completes and before downstream rendered/exported outputs are written
- [ ] 4.3 Handle missing, invalid, or low-confidence CTMs safely in the post-stitch path

## 5. CLI & MCP Workflows

- [ ] 5.1 Add a CLI workflow to generate CTMs for a page, album, or album set
- [ ] 5.2 Add an MCP job workflow to launch CTM generation and return job status information
- [ ] 5.3 Add a CLI or MCP review path to inspect stored CTM values, confidence, and warnings
- [ ] 5.4 Support rerun semantics for overwriting or regenerating CTMs with explicit force behavior

## 6. Tests & Validation

- [ ] 6.1 Add unit tests for CTM response parsing and retry logic
- [ ] 6.2 Add unit tests for matrix validation and clipping or coefficient sanity checks
- [ ] 6.3 Add integration tests for `_Archive/` XMP write/read of CTM metadata
- [ ] 6.4 Add render-path tests verifying CTM application changes output deterministically
- [ ] 6.5 Add CLI and MCP tests for CTM job creation and review behavior
