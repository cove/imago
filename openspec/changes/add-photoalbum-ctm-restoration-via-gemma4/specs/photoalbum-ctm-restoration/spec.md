## ADDED Requirements

### Requirement: Generate CTM restoration metadata for stitched photoalbum images
Imago MUST support generating a 3×3 chromatic-restoration Color Transformation Matrix (CTM) for a stitched photoalbum image using a configured Gemma 4 model served through LM Studio.

#### Scenario: Generate CTM for a stitched image
- **WHEN** an operator runs CTM generation for a stitched photoalbum page or stitched image
- **THEN** Imago sends the stitched image to the configured Gemma 4 workflow
- **AND** requests structured output containing a 3×3 CTM
- **AND** validates that the response contains exactly 9 finite numeric coefficients before accepting it

### Requirement: Persist CTM metadata in Adobe Camera Raw compatible XMP
Imago MUST persist accepted CTM metadata using Adobe Camera Raw-compatible XMP fields so the restoration recipe remains portable and non-destructive.

#### Scenario: Write CTM metadata to manifest XMP
- **WHEN** a valid CTM is produced for a stitched image
- **THEN** Imago writes the matrix to `crs:ColorMatrix1`
- **AND** sets `crs:HasSettings` to `True`
- **AND** preserves existing non-CTM photoalbum metadata in the same XMP document

### Requirement: Support stitch provenance alongside CTM values
Imago MUST support storing CTM metadata in an XMP manifest that can also include stitch provenance such as source ingredients and homography matrices.

#### Scenario: Persist CTM with stitch provenance metadata
- **WHEN** stitch provenance metadata is available for the stitched image
- **THEN** Imago stores ingredient file references and homography metadata alongside `crs:ColorMatrix1`
- **AND** does so without destructively modifying archival master scans

### Requirement: Apply stored CTM during render
Imago MUST be able to apply a stored CTM as a deterministic linear transform during render/export of a stitched photoalbum image.

#### Scenario: Render with stored CTM
- **WHEN** a stitched image has valid CTM metadata available
- **THEN** Imago applies the 3×3 transform before writing the final rendered output
- **AND** leaves source archival scans unchanged

### Requirement: Expose CTM generation as a job workflow
Imago MUST expose CTM generation and review through job-capable CLI and MCP workflows.

#### Scenario: Launch CTM generation as a tracked job
- **WHEN** an operator invokes the CTM-generation workflow for a page, album, or album set
- **THEN** Imago enqueues or tracks the work as a job
- **AND** returns a job identifier or equivalent status handle
- **AND** allows operators to inspect the stored CTM, confidence, and warnings after completion
