# Photo Albums Processing Pipeline Specification

**Purpose:** Technical specification for reimplementing the photo album digitization and AI indexing pipeline.

**Scope:** Complete workflow from raw TIFF scans through AI-enriched metadata to final indexed photo archives.

---

## 1. Input & Prerequisites

### 1.1 Input Format
- **Source:** Raw TIFF scans from physical album pages
- **Location:** `{PHOTOS_ROOT}/_Archive/` directories
- **Filename Pattern:** `{Collection}_{Year}_B{Book:02d}_P{Page:02d}_S{Scan:02d}.tif`
  - Collection: alphanumeric (e.g., "Egypt", "Cordell")
  - Year: 4-digit YYYY or YYYY-YYYY range
  - Book: 2-digit number (00-99) or special ellipsis character
  - Page: 2-digit page number within album
  - Scan: 2-digit scan number per page (multiple scans per page due to oversized originals)
- **Scanner Configuration:** All preprocessing disabled—scans are raw, unrotated TIFF files

### 1.1a Scanner Hardware & Specifications
- **Device:** Epson Perfection V19 II flatbed scanner
- **Max Scan Width:** 8.5 inches (at full width for document scanning)
- **Physical Constraint:** Photo album pages are typically 11" × 14" or larger, exceeding scanner width
- **Scanning Method:** Multiple overlapping passes per album page (e.g., left portion, center, right portion)
- **Overlap Requirements:** 8-42% horizontal overlap between adjacent scans to enable stitching
- **Resolution:** Scans at native optical resolution (~600-1200 DPI depending on album condition)
- **Preprocessing:** Disabled in scanner settings; all color correction and auto-rotation handled in post-processing

### 1.2 Directory Structure
```
{PHOTOS_ROOT}/
├── {Collection}_{Year}_B{Book}_Album_Archive/
│   ├── (raw TIFF scans)
│   └── (subdirectories for multi-page/multi-scan albums)
├── {Collection}_{Year}_B{Book}_Album_Pages/
│   └── (stitched page view JPEGs and XMP sidecars)
└── {Collection}_{Year}_B{Book}_Album_Photos/
    └── (extracted crop photos and metadata)
```

### 1.3 Required Configuration Files
- `album_sets.toml`: Mapping of Photos root paths to archive sets (for people roster lookup)
- `ai_models.toml`: Model endpoint and parameter overrides per archive set
- Per-archive `render_settings.json`: Render overrides (if present)

---

## 2. Page Rendering (Stitch Oversized Pages)

### 2.1 Purpose & Hardware Context
Photo album pages (typically 11" × 14" or larger) exceed the maximum scan width of the Epson Perfection V19 II scanner (which has a physical scan width of ~8.5" for standard documents). Therefore:

- **Physical Constraint:** The Epson Perfection V19 II can only scan up to 8.5 inches wide at native resolution
- **Solution:** Scan each oversized album page in multiple overlapping passes (left column, center, right column, etc.)
- **Goal:** Combine these overlapping scans into a single unified page view JPEG with correct geometry

The stitching process reconstructs the full album page from 2-4 overlapping TIFF scans per page, accounting for varying overlap amounts and slight perspective differences in the scanned regions.

### 2.2 Purpose
Combine multiple TIFF scans of a single album page into a single stitched view JPEG.

### 2.2 Entry Point
- **Input:** Raw TIFF scan groups under `{PHOTOS_ROOT}/_Archive/`
- **Output:** Page view JPEG under `{PHOTOS_ROOT}/_Pages/` as `{Album}_{Page}_V.jpg`

### 2.3 Scan Grouping
Scans are grouped by (Collection, Year, Book, Page). All scans with the same page identifier belong to one stitched output.

### 2.4 Stitching Algorithm

**Affine Stitching (Primary Method)**
Try stitching attempts in order; first successful attempt is used:
```
AFFINE_STITCH_ATTEMPTS = [
    {"detector": "sift", "confidence_threshold": 0.3},
    {"detector": "sift", "confidence_threshold": 0.1},
    {"detector": "akaze", "confidence_threshold": 0.3},
    {"detector": "akaze", "confidence_threshold": 0.1},
    {"detector": "brisk", "confidence_threshold": 0.1},
]
```
- Uses `stitching.AffineStitcher` library
- Attempt parameters control feature detection and match confidence requirements
- Falls back to next method if current fails or produces invalid output

**Linear Fallback (Secondary Method)**
When affine stitching fails, use linear stitching with optimized overlap parameters:
```
LINEAR_FALLBACK_TARGET_WIDTH = 640
LINEAR_FALLBACK_MIN_OVERLAP_FRAC = 0.08
LINEAR_FALLBACK_MAX_OVERLAP_FRAC = 0.42
LINEAR_FALLBACK_MAX_VERTICAL_SHIFT_FRAC = 0.08
LINEAR_FALLBACK_MIN_SHARED_HEIGHT_FRAC = 0.6
LINEAR_FALLBACK_MIN_DETAIL_FRAC = 0.05
LINEAR_FALLBACK_OVERLAP_STEP = 12
LINEAR_FALLBACK_VERTICAL_STEP = 4
LINEAR_FALLBACK_REFINE_OVERLAP_RADIUS = 12
LINEAR_FALLBACK_REFINE_VERTICAL_RADIUS = 4
LINEAR_FALLBACK_EXPANSION_RATIO = 1.02
```
- Scales images to target width for faster alignment
- Searches optimal overlap and vertical shift within tolerance ranges
- Refines results in tighter radius around best candidate
- Expands final canvas slightly to prevent clipping

### 2.5 Image Format Handling
- **Input Formats:** TIFF (with auto-orient), JPEG, PNG
- **Read Methods:** 
  - Primary: Pillow (with EXIF transpose)
  - Fallback: ImageMagick (`magick` command) for problematic formats
- **Output Format:** JPEG (quality=95, RGB color space)

### 2.6 Output Validation
- Check file size > 0
- Validate JPEG/TIFF integrity with Pillow
- Skip existing valid outputs (unless `--force`)
- Re-render if existing output fails validation

### 2.7 Large Image Handling
- Lift PIL decompression bomb limits: `allow_large_pillow_images()`
- Set `OPENCV_IO_MAX_IMAGE_PIXELS = 2^40` for OpenCV/stitching library

---

## 3. Photo Region Detection

### 3.1 Purpose
Identify individual photographs within a stitched page view JPEG using Docling's standard image pipeline.

### 3.2 Entry Point
- **Input:** Page view JPEG (`{Album}_P{Page:02d}_V.jpg`)
- **Output:** Bounding boxes as MWG-RS region metadata in XMP sidecar

### 3.3 Docling Configuration

**Pipeline Type:** Standard image pipeline (not OCR, not document)
```
PdfPipelineOptions(
    do_ocr=False,
    accelerator_options=AcceleratorOptions(device={device}),
)
ImageDocumentBackend
```

**Settings:**
- Preset: from `default_docling_preset()` (e.g., "fast", "standard", "high_quality")
- Backend: from `default_docling_backend()` (e.g., "auto_inline", "transformers", "mlx")
- Device: from `default_docling_device()` (e.g., "auto", "cpu", "cuda", "mps")
- Retries: from `default_docling_retries()` (attempts if no regions found)

**Region Extraction from Docling Output:**
1. Iterate `document.pages[page_idx].iterate_items()`
2. Filter for items with `label == DocItemLabel.PICTURE`
3. Extract bounding boxes from `item.prov[0].bbox` (top-left origin)
4. Convert pixel coordinates to top-left origin (if needed): `bbox.to_top_left_origin(page_height)`
5. Extract caption hints from `item.captions[0].cref` → `document.texts[idx]`

**Coordinate Output:**
- Format: pixel coordinates (x, y, width, height) in image space
- Convert to MWG-RS normalized center-point coords: 
  ```
  cx = (x + w/2) / img_w
  cy = (y + h/2) / img_h
  nw = w / img_w
  nh = h / img_h
  ```

### 3.4 Region Validation
Validate detected regions before persisting:
- **Zero Area:** Region has width ≤ 0 or height ≤ 0 → reject
- **Full Page:** Region occupies ≥90% of page area → reject
- **Clamping:** Clamp regions to image bounds; log warnings if clamped by >5% of image dimension

---

## 4. Photo Cropping from Regions

### 4.1 Purpose
Extract individual cropped photos from page view JPEG using detected region bounding boxes.

### 4.2 Entry Point
- **Input:** Page view JPEG + XMP region metadata
- **Output:** Individual crop JPEGs in `{PHOTOS_ROOT}/_Photos/`
- **Naming:** `{Album}_P{Page:02d}_D{CropIndex:02d}-00_V.jpg`

### 4.3 Region Extraction
For each region in XMP sidecar:
1. Read MWG-RS normalized center-point coords (cx, cy, width, height)
2. Convert to pixel rectangle:
   ```
   left_f = (cx - width/2) * img_w
   top_f = (cy - height/2) * img_h
   right_f = (cx + width/2) * img_w
   bottom_f = (cy + height/2) * img_h
   left = max(0, round(left_f))
   top = max(0, round(top_f))
   right = min(img_w, round(right_f))
   bottom = min(img_h, round(bottom_f))
   ```
3. Clamp to image bounds; log warning if clamped by >5% of image dimension
4. Extract crop: `crop = image[top:bottom, left:right]`

### 4.4 Photo Restoration

**Model:** RealRestorer diffusion pipeline
- **Repo:** `https://github.com/yfyang007/RealRestorer.git`
- **Model Name:** `RealRestorer/RealRestorer`
- **Availability Check:** Try importing from `diffusers` library; skip if unavailable
- **RAM Requirement:** Model repo size ≈41.8 GB; skip if installed RAM < repo size

**Restoration Inference:**
```python
result = pipe(
    image=crop_image,
    prompt="Please restore this low-quality image, recovering its normal brightness and clarity.",
    num_inference_steps=28,
    guidance_scale=3.0,
    seed=42,
    size_level=1024,
)
output_image = result.images[0]
```

**Runtime:** Automatically selects dtype and device:
- CUDA available → `torch.bfloat16` + "cuda" with `pipe.enable_model_cpu_offload()`
- MPS available → `torch.float32` + "mps"
- CPU fallback → `torch.float32` + "cpu"

**Fallback Behavior:**
- If RealRestorer unavailable (not installed): return original crop unchanged, log "restoration_unavailable"
- If inference fails: return original crop unchanged, log error with exception details

### 4.5 Output Handling
- **Format:** JPEG (quality=95, RGB)
- **Skipping Existing:** Skip crop output if already exists (unless `--force`)
- **Force Restoration Flag:** Re-run restoration on existing crops without forcing full crop step

---

## 5. AI Processing Pipeline

### 5.1 Overview
For each page view, run a series of AI analyses on the original page and extracted crops, writing results to XMP sidecars.

### 5.2 Processing Modes
- **default:** Process all pages/crops without XMP
- **gps:** Reprocess only GPS/location estimation (re-runnable)
- **force:** Force full reprocessing even if XMP exists
- **skip-existing:** Skip pages/crops with valid existing XMP

### 5.3 Page-Level Processing

#### 5.3.1 OCR & Caption Extraction
**Model:** Claude API with vision
**Prompt:** Structured metadata extraction from page images

**System Prompt:**
```
- You extract metadata from scanned photo album pages.
- Return only valid JSON matching the response_format schema.
- Return one entry in `photos` for each photograph visible on the page.
- `photo_number`: The number shown in the bounding box label on the image for this photo. Use this to identify which photo you are describing.
- `location`: Nominatim-queryable location string for the photo. Always include a country name. If the country is not visible, infer it from the album title.
- `location_name`: Famous or well-known location name only if it can be confidently identified from visible evidence. Always include a country name, and city if possible. Use the country name from the album title if unsure.
- `est_date`: Estimated date or date range the photo was taken. First use explicit date evidence visible on the album page near the photo, in page titles, captions, handwritten notes, or scene text, preserving the best supported precision (`1988 Aug.` -> `1988-08`). If the page gives month/day but no year, use the album title as the year. If no date is visible on the page, use the album title year only. Do not infer a different year from clothing, vehicles, or visual style when an album title year is available.
- `scene_ocr`: Any text visible within the photo itself (signs, labels, banners). Empty string if none.
- `caption`: The handwritten or typed caption text written by the album owner near this photo on the album page (beneath, beside, or above it). Copy it verbatim — keep the original wording, casing, and punctuation. **Always populate this field whenever any caption text exists on the page near a photo, even if that text also describes a location and was used to derive `location` or `location_name`.** A single caption may apply to several adjacent photos; in that case repeat the same caption text in each of those photos' entries. Empty string only when there is no caption text written near this photo at all.
- `corrected_caption`: Correct obvious spelling errors in `caption` only when the caption appears to be a place/location name and the corrected spelling would make the place queryable. Keep this empty when there is no confident correction. Do not rewrite normal prose captions.
- `people_count`: Number of clearly visible people in the photo.
```

**User Prompt:**
```
Analyze this album page.
Album title: {album_title}
```

**Parameters:**
- `max_tokens = 2048`
- `temperature = 0.1`
- `timeout = 300.0 seconds`

**Output Schema:** JSON with fields: `photos[]` (array of photo metadata objects)

#### 5.3.2 People Count (Per-Crop Refinement)
For each detected crop, optionally run separate people-counting to refine estimates.

**System Prompt:**
```
- You count visible people in photographs.
- Return only valid JSON matching the response_format schema.
- Count clearly visible real people only.
- Do not include reasoning or extra fields.
```

**User Prompt:**
```
- Count the number of clearly visible real people.
```

**Parameters:**
- `max_tokens = 48`
- `temperature = 0.0`
- `timeout = 300.0 seconds`

**Output Schema:** JSON with field: `count` (integer)

#### 5.3.3 People Matching (Optional)
Use Cast service (external face matching system) to match detected faces against roster.

**Inputs:**
- Detected faces from YOLO (optional)
- People roster from album set configuration
- Cast service embeddings

**Output:** Person name + confidence score per detection

#### 5.3.4 Object Detection (Optional)
Use YOLO to detect visual objects in page/crop images.

**Output:** Object class labels + bounding boxes + confidence scores

#### 5.3.5 GPS Location Estimation
Estimate GPS coordinates from image content and page context.

**Inputs:**
- Page/crop image content
- Page caption/description from OCR
- Album context (collection, year, page number)
- People names (matched or manually entered)

**Output:** Latitude/longitude coordinates with confidence metadata

#### 5.3.6 Geocoding (Nominatim Reverse Lookup)
Convert GPS coordinates to human-readable location names.

**Service:** Nominatim OpenStreetMap API
**Base URL:** `https://nominatim.openstreetmap.org`
**User Agent:** `imago-photoalbums-ai-index/1.0`
**Timeout:** 20.0 seconds per request
**Min Interval:** 1.0 second between requests (rate limiting)
**Cache:** `{PHOTOALBUMS_DIR}/data/geocode_cache.json`

**Query Structure:**
- Input: lat, lon, zoom (optional)
- Output: JSON with `address` dict containing: `country`, `state`, `city`, `town`, `village`, `road`, `postcode`, etc.

**Fallback Behavior:**
- Network timeout → retry with exponential backoff (up to 3 attempts)
- No result found → store empty location, continue processing
- Rate limited (HTTP 429) → pause and retry
- Connection refused → skip geocoding, store estimated coords only

### 5.4 Crop-Level Processing

Individual cropped photos inherit page-level metadata (caption, date, location) and may be refined with:
- Crop-specific people counting
- Crop-specific location grounding
- Crop XMP verification (see Section 7)

---

## 6. XMP Sidecar Metadata Schema

### 6.1 File Structure
**Naming:** `{SourceImageName}.xmp` (parallel to JPEG/TIFF)
**Format:** XML with RDF namespaces and custom schema

### 6.2 Namespace Declarations
```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
      xmlns:imago="https://imago.local/ns/1.0/"
      xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
      xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/">
```

### 6.3 Dublin Core (dc:) Metadata
```xml
<dc:title>
  <rdf:Alt>
    <rdf:li xml:lang="x-default">{title}</rdf:li>
  </rdf:Alt>
</dc:title>

<dc:description>
  <rdf:Alt>
    <rdf:li xml:lang="x-default">{description}</rdf:li>
  </rdf:Alt>
</dc:description>

<dc:date>
  <rdf:Seq>
    <rdf:li>{YYYY-MM-DDTHH:MM:SS format or YYYY-MM or YYYY}</rdf:li>
  </rdf:Seq>
</dc:date>

<dc:creator>
  <rdf:Seq>
    <rdf:li>{person_name}</rdf:li>
  </rdf:Seq>
</dc:creator>

<dc:subject>
  <rdf:Bag>
    <rdf:li>{keyword}</rdf:li>
  </rdf:Bag>
</dc:subject>

<dc:source>{source_page_filename}</dc:source>
```

### 6.4 EXIF Metadata
```xml
<exif:DateTimeOriginal>{YYYY:MM:DD HH:MM:SS}</exif:DateTimeOriginal>
<exif:ImageWidth>{width}</exif:ImageWidth>
<exif:ImageLength>{height}</exif:ImageLength>
<exif:GPSLatitude>{lat_degrees},{lat_minutes},{lat_seconds}</exif:GPSLatitude>
<exif:GPSLongitude>{lon_degrees},{lon_minutes},{lon_seconds}</exif:GPSLongitude>
<exif:GPSAltitude>{altitude_in_meters}</exif:GPSAltitude>
```

### 6.5 IPTC-Ext (iptc:) Metadata
```xml
<Iptc4xmpExt:LocationShown>
  <rdf:Bag>
    <rdf:li>
      <Iptc4xmpExt:LocationName>{location}</Iptc4xmpExt:LocationName>
      <Iptc4xmpExt:LocationCreated>
        <!-- nested location hierarchy -->
      </Iptc4xmpExt:LocationCreated>
    </rdf:li>
  </rdf:Bag>
</Iptc4xmpExt:LocationShown>
```

### 6.6 Imago Custom Schema (imago:)
**Namespace:** `https://imago.local/ns/1.0/`

```xml
<imago:ProcessingStatus>{state}</imago:ProcessingStatus>
<imago:PipelineSteps>
  <rdf:Seq>
    <rdf:li>
      <rdf:Description>
        <imago:StepName>{step_name}</imago:StepName>
        <imago:StepModel>{model_name}</imago:StepModel>
        <imago:StepResult>{result_status}</imago:StepResult>
        <imago:StepTimestamp>{ISO8601_datetime}</imago:StepTimestamp>
        <imago:StepExtra>{JSON_object}</imago:StepExtra>
      </rdf:Description>
    </rdf:li>
  </rdf:Seq>
</imago:PipelineSteps>

<imago:PersonCount>{count}</imago:PersonCount>
<imago:PersonInImage>
  <rdf:Bag>
    <rdf:li>
      <rdf:Description>
        <imago:PersonName>{name}</imago:PersonName>
        <imago:MatchConfidence>{0.0-1.0}</imago:MatchConfidence>
        <imago:MatchSource>{source}</imago:MatchSource>
      </rdf:Description>
    </rdf:li>
  </rdf:Bag>
</imago:PersonInImage>

<imago:ObjectDetections>
  <rdf:Bag>
    <rdf:li>
      <rdf:Description>
        <imago:ObjectClass>{class_label}</imago:ObjectClass>
        <imago:Confidence>{0.0-1.0}</imago:Confidence>
      </rdf:Description>
    </rdf:li>
  </rdf:Bag>
</imago:ObjectDetections>

<imago:LocationPayload>{JSON_object}</imago:LocationPayload>
<imago:LocationReverse>{JSON_from_Nominatim}</imago:LocationReverse>
```

### 6.7 MWG-RS Region Metadata (mwg-rs:)
**Use Case:** Photo region detection results and crop metadata
**Schema:** MWG (Metadata Working Group) regions specification

```xml
<mwg-rs:RegionList>
  <rdf:Bag>
    <rdf:li>
      <rdf:Description>
        <!-- Bounding box in normalized center-point coords (0-1) -->
        <mwg-rs:Area>
          <rdf:Description>
            <mwg-rs:x>{cx}</mwg-rs:x>
            <mwg-rs:y>{cy}</mwg-rs:y>
            <mwg-rs:w>{width}</mwg-rs:w>
            <mwg-rs:h>{height}</mwg-rs:h>
            <mwg-rs:unit>normalized</mwg-rs:unit>
          </rdf:Description>
        </mwg-rs:Area>
        
        <!-- Region type and name -->
        <mwg-rs:Type>Photo</mwg-rs:Type>
        <mwg-rs:Name>{photo_number or auto-index}</mwg-rs:Name>
        
        <!-- Region description and captions -->
        <mwg-rs:Description>{caption_hint}</mwg-rs:Description>
      </rdf:Description>
    </rdf:li>
  </rdf:Bag>
</mwg-rs:RegionList>
```

### 6.8 XMP Standard Fields (xmp:)
```xml
<xmp:CreatorTool>imago-photoalbums</xmp:CreatorTool>
<xmp:CreateDate>{ISO8601_datetime}</xmp:CreateDate>
<xmp:MetadataDate>{ISO8601_datetime}</xmp:MetadataDate>
<xmp:ModifyDate>{ISO8601_datetime}</xmp:ModifyDate>
```

### 6.9 XMP Media Management (xmpMM:)
```xml
<xmpMM:DocumentID>urn:uuid:{unique-id}</xmpMM:DocumentID>
<xmpMM:InstanceID>urn:uuid:{instance-id}</xmpMM:InstanceID>
<xmpMM:History>
  <rdf:Seq>
    <rdf:li>
      <rdf:Description>
        <xmpMM:action>created</xmpMM:action>
        <xmpMM:when>{timestamp}</xmpMM:when>
        <xmpMM:software>imago-photoalbums</xmpMM:software>
      </rdf:Description>
    </rdf:li>
  </rdf:Seq>
</xmpMM:History>
```

### 6.10 Date Format Normalization
- **Dates on file:** ISO 8601 preferred (e.g., `2024-05-08T14:30:00Z`)
- **Partial dates:** `YYYY`, `YYYY-MM`, `YYYY-MM-DD` accepted
- **EXIF format:** `YYYY:MM:DD HH:MM:SS` (colon separator)
- **Parsing:** Normalize all formats to canonical ISO 8601 + remove microseconds
- **Timezone:** All times in UTC (Z suffix)

### 6.11 Example Page View XMP
```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:imago="https://imago.local/ns/1.0/"
      xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/">
      
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Egypt 1975, Book B, Page 05</rdf:li>
        </rdf:Alt>
      </dc:title>
      
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Vacation photos from Cairo. Two photos: 1. Temple entrance with tourists. 2. Local market scene.</rdf:li>
        </rdf:Alt>
      </dc:description>
      
      <dc:date>
        <rdf:Seq>
          <rdf:li>1975-08</rdf:li>
        </rdf:Seq>
      </dc:date>
      
      <exif:ImageWidth>3200</exif:ImageWidth>
      <exif:ImageLength>2400</exif:ImageLength>
      
      <mwg-rs:RegionList>
        <rdf:Bag>
          <rdf:li>
            <rdf:Description>
              <mwg-rs:Area>
                <rdf:Description>
                  <mwg-rs:x>0.25</mwg-rs:x>
                  <mwg-rs:y>0.30</mwg-rs:y>
                  <mwg-rs:w>0.35</mwg-rs:w>
                  <mwg-rs:h>0.40</mwg-rs:h>
                  <mwg-rs:unit>normalized</mwg-rs:unit>
                </rdf:Description>
              </mwg-rs:Area>
              <mwg-rs:Type>Photo</mwg-rs:Type>
              <mwg-rs:Name>1</mwg-rs:Name>
            </rdf:Description>
          </rdf:li>
        </rdf:Bag>
      </mwg-rs:RegionList>
      
      <imago:ProcessingStatus>complete</imago:ProcessingStatus>
      <imago:PipelineSteps>
        <rdf:Seq>
          <rdf:li>
            <rdf:Description>
              <imago:StepName>view_regions</imago:StepName>
              <imago:StepModel>docling-standard-image</imago:StepModel>
              <imago:StepResult>regions_found</imago:StepResult>
              <imago:StepTimestamp>2024-05-08T14:30:00Z</imago:StepTimestamp>
            </rdf:Description>
          </rdf:li>
        </rdf:Seq>
      </imago:PipelineSteps>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

### 6.12 Example Crop XMP
```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:exif="http://ns.adobe.com/exif/1.0/"
      xmlns:imago="https://imago.local/ns/1.0/">
      
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Temple entrance with tourists</rdf:li>
        </rdf:Alt>
      </dc:title>
      
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Temple entrance. Handwritten caption: "Cairo temple 1975"</rdf:li>
        </rdf:Alt>
      </dc:description>
      
      <dc:date>
        <rdf:Seq>
          <rdf:li>1975-08</rdf:li>
        </rdf:Seq>
      </dc:date>
      
      <dc:source>Egypt_1975_B01_P05_S01_V.jpg</dc:source>
      
      <exif:ImageWidth>1024</exif:ImageWidth>
      <exif:ImageLength>768</exif:ImageLength>
      <exif:GPSLatitude>30.0288,3.0,0.0</exif:GPSLatitude>
      <exif:GPSLongitude>31.2495,0.0,0.0</exif:GPSLongitude>
      
      <imago:PersonCount>4</imago:PersonCount>
      <imago:LocationPayload>{
        "inferred_location": "Cairo, Egypt",
        "confidence": 0.85,
        "source": "image_content"
      }</imago:LocationPayload>
      
      <imago:ProcessingStatus>complete</imago:ProcessingStatus>
      <imago:PipelineSteps>
        <rdf:Seq>
          <rdf:li>
            <rdf:Description>
              <imago:StepName>crop</imago:StepName>
              <imago:StepResult>success</imago:StepResult>
              <imago:StepTimestamp>2024-05-08T14:35:00Z</imago:StepTimestamp>
            </rdf:Description>
          </rdf:li>
          <rdf:li>
            <rdf:Description>
              <imago:StepName>restoration</imago:StepName>
              <imago:StepModel>RealRestorer</imago:StepModel>
              <imago:StepResult>restored</imago:StepResult>
              <imago:StepTimestamp>2024-05-08T14:36:00Z</imago:StepTimestamp>
            </rdf:Description>
          </rdf:li>
        </rdf:Seq>
      </imago:PipelineSteps>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

---

## 7. Crop Verification & Validation

### 7.1 Purpose
Validate crop metadata (caption, date, location) against page context before finalizing.

### 7.2 Entry Point
Verify crops after region detection and AI processing.

### 7.3 Verification Prompt

**System Prompt:**
```
- You verify already-generated crop metadata against family photo album page context.
- Return only valid JSON matching the response_format schema.
- Treat the crop image as the primary review target and the page image as supporting context.
- Use the supplied page XMP text and crop XMP text as evidence, not as unquestioned truth.
- Mark a concern `good` only when the metadata is supported by what a careful human would infer from the page.
- Mark a concern `bad` when the metadata conflicts with the page context.
- Mark a concern `uncertain` when the page context does not support a confident human judgment.
- Keep each concern `reasoning` to one sentence.
- Keep each concern `failure_reason` empty when verdict is `good`.
- When any concern is `bad` or `uncertain`, include `human_inference` describing what a person would actually infer from the page.
- `needs_another_pass` and `needs_human_review` must use readable concern names from: `caption`, `gps`, `shown_location`, `date`.
- `overall` is summary only and must not appear in retry-routing arrays.
```

**User Prompt:**
```
- Review target: one crop image plus its crop XMP metadata, using the full album page image and page XMP metadata as supporting context.
- Judge whether the crop caption belongs to that crop on the page, including nearby-caption carry-over when a human would read neighboring captions together.
- Judge whether shown location matches the crop image, `dc:description`, or visible landmarks.
- Judge whether the GPS coordinates are generally consistent with the crop image and `dc:description`.
- For GPS and shown-location concerns, use the supplied location verification evidence: GPS coordinates plus Nominatim reverse-lookup name. OCR text is intentionally not supplied for location verification.
- If the reverse-lookup name and GPS coordinates align with what is visible in the image or stated in `dc:description`, consider the location grounded and mark it as good.
- Judge whether date preserves the best supported precision from the page, including month-plus-year evidence such as `AUG. 1988` -> `1988-08` that is non-standard but human-readable. If no date is visible in the page context, accept the album title year as the correct fallback year. Do not mark an album-title-year-only date as uncertain merely because the page itself lacks a visible date.
- If page image context is missing, do not pretend a full review happened.
- If page XMP text context is missing, do not pretend a full metadata-context review happened.
- Base your verdicts on what a human flipping through the album would accept as belonging together.
```

### 7.4 Verification Concerns
Evaluated fields: `caption`, `gps`, `shown_location`, `date`

**Verdict Values:**
- `good`: Metadata supported by page context
- `bad`: Metadata conflicts with page context
- `uncertain`: Insufficient page context for confident judgment

**Output:** JSON array of concern objects with verdicts + reasoning

### 7.5 Retry & Human Review Routing
If verification finds concerns:
- **Needs Another Pass:** Re-run AI processing with adjusted prompts or models (specific failing fields)
- **Needs Human Review:** Metadata cannot be auto-corrected; flag for manual inspection

---

## 8. Location Resolution & Fallbacks

### 8.1 Location Payload Structure
Internal representation of location metadata during processing:

```json
{
  "inferred_location": "Cairo, Egypt",
  "confidence": 0.75,
  "source": "image_content",
  "explicit_gps": null,
  "location_shown": {
    "city": "Cairo",
    "country": "Egypt",
    "additional": "Temple district"
  },
  "nominatim_result": {
    "lat": "30.0288",
    "lon": "31.2495",
    "display_name": "...",
    "address": { ... }
  }
}
```

### 8.2 Location Extraction
**Input:** AI-generated location string from page OCR
**Query:** Pass string to Nominatim for coordinate resolution

**Nominatim Request Parameters:**
- `q`: location query string (e.g., "Cairo, Egypt")
- `format`: "json"
- `limit`: 1 (top result only)
- `timeout`: 20.0 seconds
- `User-Agent`: "imago-photoalbums-ai-index/1.0"

### 8.3 Reverse Geocoding
**Input:** GPS coordinates from AI estimation
**Query:** Nominatim reverse lookup for human-readable name

**Request Parameters:**
- `lat`, `lon`: coordinates
- `format`: "json"
- `zoom`: 10 (city/county level detail)
- `timeout`: 20.0 seconds

### 8.4 Cache Management
**File:** `{PHOTOALBUMS_DIR}/data/geocode_cache.json`
**Format:** JSON dict mapping query string → result
**Behavior:**
- Check cache before network request
- Store successful results
- Skip failed requests (timeout, 404, rate limit)

### 8.5 Rate Limiting
- **Min Interval:** 1.0 second between Nominatim requests
- **Backoff:** Exponential (2s, 4s, 8s, 16s) on timeout/error
- **Max Retries:** 3 attempts before giving up

### 8.6 Fallback Behavior by Scenario

**Scenario: Nominatim unavailable (network error)**
- Log warning
- Retry with exponential backoff (up to 3 attempts)
- If still fails: store original location string only, no coordinates

**Scenario: Location string not found**
- Return empty/null coordinates
- Preserve original location string
- Continue processing

**Scenario: Rate limited (HTTP 429)**
- Sleep 60+ seconds
- Retry request
- If persists: skip geocoding for remaining batch

**Scenario: Invalid coordinates**
- Skip reverse lookup
- Preserve original coordinates in XMP

---

## 9. Pipeline State Tracking

### 9.1 Processing Status in XMP
Field: `imago:ProcessingStatus`
Values:
- `pending`: Not yet processed
- `processing`: Currently running
- `complete`: Successfully finished
- `failed`: Error occurred

### 9.2 Pipeline Steps
Field: `imago:PipelineSteps` (rdf:Seq of step descriptions)

**Per-Step Metadata:**
- `imago:StepName`: identifier (e.g., "view_regions", "crop", "restoration", "ai_caption")
- `imago:StepModel`: model/service used (e.g., "docling-standard-image", "RealRestorer", "Claude-opus")
- `imago:StepResult`: outcome (e.g., "regions_found", "success", "no_regions", "failed")
- `imago:StepTimestamp`: ISO 8601 timestamp when step completed
- `imago:StepExtra`: arbitrary JSON object with step-specific details (e.g., `{"result": "no_regions"}`)

### 9.3 Skip Existing Logic
When determining whether to reprocess:
1. Check if XMP exists and is readable
2. Look for pipeline step with target name
3. If step result is in terminal states (`no_regions`, `validation_failed`, `failed`), consider complete
4. If step result is in success states (`regions_found`, `success`), verify metadata presence
5. If metadata missing despite success state, force reprocess

---

## 10. Configuration & Settings

### 10.1 Album Sets (`album_sets.toml`)
```toml
[archive_set_name]
photos_root = "/path/to/Photo Albums"
people_roster_path = "people.csv"
```

Maps archive sets to directory roots and people roster files.

### 10.2 AI Models (`ai_models.toml`)
```toml
[archive_set_name.docling]
preset = "standard"
backend = "auto_inline"
device = "auto"
retries = 3

[archive_set_name.restoration]
enabled = true
model_name = "RealRestorer/RealRestorer"
```

Per-archive model configuration overrides.

### 10.3 Render Settings (`render_settings.json`)
```json
{
  "archive_settings": {
    "render_scale": 1.0,
    "jpeg_quality": 95
  },
  "chapter_settings": {
    "Egypt_1975_B01": {
      "stitch_detector": "sift"
    }
  }
}
```

Per-album rendering control parameters.

---

## 11. Naming & Identification

### 11.1 File Naming Patterns

**Album Directory Base Name:**
```
{Collection}_{Year}_B{Book:02d}_{Album}
```

**TIFF Scan:**
```
{Collection}_{Year}_B{Book:02d}_P{Page:02d}_S{Scan:02d}.tif
```

**Page View:**
```
{Collection}_{Year}_B{Book:02d}_P{Page:02d}_V.jpg
```

**Cropped Photo:**
```
{Collection}_{Year}_B{Book:02d}_P{Page:02d}_D{CropIndex:02d}-{IterationIndex:02d}_V.jpg
```

Notation:
- Collection: alphanumeric string
- Year: YYYY or YYYY-YYYY range
- Book: 2-digit number, ellipsis char, or unknown marker
- Page: 2-digit decimal
- Scan: 2-digit decimal
- CropIndex: 2-digit index within page (00, 01, 02, ...)
- IterationIndex: 2-digit iteration number (00=first crop, useful for versioning)

### 11.2 Regex Patterns
See `/photoalbums/naming.py` for authoritative patterns:
- `SCAN_TIFF_RE`: Match raw TIFF scan filenames
- `SCAN_NAME_RE`: Match page/scan identifiers
- `DERIVED_NAME_RE`: Match cropped photo filenames
- `BASE_PAGE_NAME_RE`: Match base album identifier
- `VIEW_PAGE_RE`: Match page view files
- `DERIVED_VIEW_RE`: Match derived crop files

---

## 12. Error Handling & Recovery

### 12.1 Validation Failures
**Region validation fails:**
- Log region as invalid (zero area, full page, etc.)
- Continue with remaining regions
- Mark page as "validation_failed" in pipeline state

**XMP parsing fails:**
- Log error with path
- Retry parse (up to 2 times)
- If persists: skip image, log to error list

**Image reading fails (Pillow):**
- Try ImageMagick fallback
- If both fail: skip image

### 12.2 Network Failures
**Nominatim timeout:**
- Retry with exponential backoff (1s, 2s, 4s base)
- Max 3 attempts
- If all fail: skip geocoding for image

**Claude API failure:**
- Log error
- Skip AI extraction step
- Continue with remaining images

### 12.3 Processing Locks
To prevent concurrent processing of same image:
- Acquire lock file before processing: `{image_path}.processing`
- Release lock on completion (success or failure)
- If lock exists and stale (>1 hour old), allow reacquisition

---

## 13. Output Artifacts

### 13.1 Directory Structure After Full Processing

```
{PHOTOS_ROOT}/
├── {Album}_Archive/
│   ├── (original raw TIFFs)
├── {Album}_Pages/
│   ├── {Album}_P01_V.jpg
│   ├── {Album}_P01_V.xmp
│   ├── {Album}_P02_V.jpg
│   ├── {Album}_P02_V.xmp
│   └── ...
└── {Album}_Photos/
    ├── {Album}_P01_D00-00_V.jpg
    ├── {Album}_P01_D00-00_V.xmp
    ├── {Album}_P01_D01-00_V.jpg
    ├── {Album}_P01_D01-00_V.xmp
    ├── {Album}_P02_D00-00_V.jpg
    ├── {Album}_P02_D00-00_V.xmp
    └── ...
```

### 13.2 XMP Sidecar Metadata
Every JPEG (page view or crop) has companion `.xmp` file with:
- OCR text & captions
- Detected faces & objects
- GPS coordinates
- Estimated dates
- People names & confidence
- Processing pipeline history
- Region detection metadata (page views only)

### 13.3 Optional: Debug Artifacts
When `--debug` flag enabled:
- Docling pipeline debug JSON: `{image_path}.view-regions.debug.json`
- AI processing logs: stdout/stderr redirection
- Verification results: `{image_path}.verify-crops.debug.json`

---

## 14. Maintenance & Updates

### 14.1 Keeping SPEC.md Current
This specification references specific:
- **Prompts:** In `photoalbums/prompts/` directory
- **Models:** Specified by name in model configuration
- **Parameters:** In `params.toml` files and code constants
- **Services:** Nominatim URL, Claude API, Cast service

When any of these change:
1. Update relevant source files (prompts, code constants)
2. Update corresponding sections in SPEC.md
3. Document the rationale for change in commit message
4. Consider backward-compatibility impact

### 14.2 Version Tracking
Include version/timestamp in SPEC.md header when document is updated.

### 14.3 Algorithm Changes
If you modify stitching fallback parameters, Docling presets, restoration prompts, or verification logic:
1. Update exact values in this spec
2. Consider whether change affects reprocessing requirements
3. Document in commit message why change improves output quality

---

## 15. Key Dependencies & External Services

### 15.1 Python Libraries & Versions
- `opencv-contrib-python` **4.10.0.84**: Image stitching (AffineStitcher), feature detection (SIFT, AKAZE, BRISK), image manipulation
- `pillow` **12.1.1**: Image I/O (JPEG, TIFF, PNG), EXIF handling, image format conversion
- `stitching` **0.6.1**: High-level affine stitcher wrapper (uses OpenCV backend)
- `docling` **2.88.0**: Document layout analysis and image region detection (photo detection from page layouts)
- `diffusers` (RealRestorer fork): RealRestorer pipeline for photo restoration (from https://github.com/yfyang007/RealRestorer.git)
- `torch` **2.10.0**: Neural network inference for RealRestorer and other ML models
- `anthropic`: Claude API client for vision-based OCR, caption extraction, and location estimation
- `xml.etree.ElementTree`: Standard library for XMP sidecar generation/parsing

### 15.2 External Services
- **Nominatim (OpenStreetMap):** Geocoding service (https://nominatim.openstreetmap.org)
- **Claude API:** Vision & text processing for OCR, captions, people counting, location estimation
- **Cast Service:** Face recognition and people matching (internal/custom)
- **YOLO:** Object detection (optional, via external model)

### 15.3 System Tools
- **ImageMagick (`magick` command):** Fallback image reading for problematic formats
- **Git:** Not required at runtime, used for version control only

---

## 16. Performance Characteristics

### 16.1 Processing Time Estimates (per page)
- **Stitching:** 2-10 seconds (varies by detector, image size)
- **Docling region detection:** 1-5 seconds
- **Cropping + restoration:** 0.5-2 seconds per crop
- **AI caption extraction:** 5-20 seconds (network latency dominated)
- **Geocoding:** 1-5 seconds per location query (network latency + cache hits)

### 16.2 Memory Usage
- Page view JPEG in memory: ~50-500 MB (depending on resolution)
- RealRestorer model: ~42 GB loaded (but uses CPU offload if needed)
- Docling converter: ~200 MB (cached per session)
- Claude API: Minimal (image sent to remote service)

### 16.3 Disk Space
- Each album page JPEG: ~2-20 MB
- Each crop JPEG: ~0.1-2 MB
- XMP sidecar: ~5-50 KB per image
- Geocoding cache: ~100-500 KB

---

## 17. Testing & Validation

### 17.1 Unit Test Entry Points
- `photoalbums/tests/` directory contains test suites
- Test data in `photoalbums/data/` and `photoalbums/evals/`

### 17.2 Integration Validation
- **Stitch rendering:** Validate output JPEG exists and is readable
- **Docling regions:** Check region count > 0 and bounding boxes within image bounds
- **XMP sidecars:** Validate XML structure and required namespace declarations
- **Crop images:** Check dimensions and format
- **Metadata round-trip:** Read XMP, parse all fields, ensure no data loss

### 17.3 Known Limitations
- **Stitching:** May fail on pages with non-overlapping scans or extreme perspective distortion
- **Docling:** Struggles with ornate page borders or text-heavy layouts
- **Location inference:** Relies on visible landmarks or context; isolated photos may have low confidence
- **Restoration:** Slow on CPU; benefits from CUDA acceleration
- **Nominatim:** Occasional rate limiting; geographical biases in coverage

---

**End of Specification**
