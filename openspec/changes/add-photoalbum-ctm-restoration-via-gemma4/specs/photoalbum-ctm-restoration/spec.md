## ADDED Requirements

### Requirement: Generate CTM restoration metadata for stitched photoalbum images
Imago MUST support generating a 3×3 chromatic-restoration Color Transformation Matrix (CTM) for a stitched photoalbum image using a configured Gemma 4 model served through LM Studio.

#### Scenario: Generate CTM for a stitched image
- **WHEN** an operator runs CTM generation for a stitched photoalbum page or stitched image
- **THEN** Imago sends the stitched image to the configured Gemma 4 workflow
- **AND** requests structured output containing a 3×3 CTM
- **AND** validates that the response contains exactly 9 finite numeric coefficients before accepting it

### Requirement: Persist CTM metadata in the `_Archive/` XMP
Imago MUST persist accepted CTM metadata in the `_Archive/` version of the XMP using Adobe Camera Raw-compatible fields so the restoration recipe remains portable and non-destructive.

#### Scenario: Write CTM metadata to `_Archive/` XMP
- **WHEN** a valid CTM is produced for a stitched image
- **THEN** Imago writes the matrix to `crs:ColorMatrix1` in the `_Archive/` XMP
- **AND** sets any required Camera Raw compatibility fields there
- **AND** preserves existing non-CTM metadata in that XMP document

### Requirement: Apply stored CTM after stitching
Imago MUST be able to apply a stored CTM as a deterministic linear transform after stitching and before downstream rendered/exported outputs are produced.

#### Scenario: Apply CTM after stitching completes
- **WHEN** a stitched image has valid CTM metadata available in the `_Archive/` XMP
- **THEN** Imago applies the 3×3 transform after stitching completes
- **AND** does so before downstream rendered or exported outputs are written
- **AND** leaves source archival scans unchanged

### Requirement: Expose CTM generation as a job workflow
Imago MUST expose CTM generation and review through job-capable CLI and MCP workflows.

#### Scenario: Launch CTM generation as a tracked job
- **WHEN** an operator invokes the CTM-generation workflow for a page, album, or album set
- **THEN** Imago enqueues or tracks the work as a job
- **AND** returns a job identifier or equivalent status handle
- **AND** allows operators to inspect the stored CTM, confidence, and warnings after completion
