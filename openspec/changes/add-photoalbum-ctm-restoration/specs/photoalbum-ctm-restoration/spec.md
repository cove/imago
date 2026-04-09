## ADDED Requirements

### Requirement: Generate CTM restoration metadata for stitched photoalbum images
Imago MUST support generating a 3×3 chromatic-restoration Color Transformation Matrix (CTM) for a stitched photoalbum image using a configured Gemma 4 model served through LM Studio.

#### Scenario: Generate CTM for a stitched page image
- **WHEN** an operator runs CTM generation for a stitched photoalbum page or image
- **THEN** Imago sends the stitched image to the configured Gemma 4 model
- **AND** requests structured output containing a 3×3 CTM
- **AND** validates that the response contains exactly 9 finite numeric coefficients before accepting it

### Requirement: Persist CTM using Adobe Camera Raw compatible XMP
Imago MUST persist accepted CTM metadata in XMP using the Adobe Camera Raw `crs:ColorMatrix1` field so the restoration recipe remains portable and non-destructive.

#### Scenario: Write CTM metadata to XMP manifest
- **WHEN** a valid CTM is produced for a stitched image
- **THEN** Imago writes the matrix to `crs:ColorMatrix1`
- **AND** marks `crs:HasSettings=True`
- **AND** preserves existing non-CTM XMP metadata in the same document

### Requirement: Support archive-manifest metadata alongside CTM values
Imago MUST support writing CTM metadata in an XMP manifest that can also carry stitch provenance metadata such as ingredient file references and homography matrices.

#### Scenario: Persist CTM with stitch provenance
- **WHEN** stitch provenance is available for a stitched image
- **THEN** Imago can store ingredient references and homography metadata alongside `crs:ColorMatrix1`
- **AND** does so without requiring destructive changes to archival master scans

### Requirement: Apply stored CTM during render
Imago MUST be able to apply a stored CTM as a deterministic linear transform during render/export of a stitched photoalbum image.

#### Scenario: Render using stored CTM
- **WHEN** a stitched image has valid CTM metadata available
- **THEN** Imago applies the 3×3 transform before writing the final rendered output
- **AND** leaves source archival scans unchanged

### Requirement: Expose CTM generation as a job workflow
Imago MUST expose CTM generation through job-capable CLI and MCP entrypoints so operators can run the workflow over a page, album, or album set.

#### Scenario: Launch CTM generation job from MCP
- **WHEN** an operator invokes the MCP CTM-generation tool
- **THEN** Imago enqueues a job and returns a job identifier
- **AND** allows operators to inspect the resulting CTM, confidence, and warnings after completion
