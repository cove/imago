## Context

Rendered sidecars currently move through three distinct phases:

```text
archive/page sidecar
  -> rendered page view sidecar
  -> crop sidecar
  -> later rewrite steps (face refresh, pipeline-state writes, CTM-related updates)
```

The first transition starts from real page metadata, but later transitions rewrite sidecars through narrower helper APIs. That creates two failure modes:

1. A field is never copied to the rendered output even though it exists upstream.
2. A field is copied once, then silently cleared by a later rewrite that only round-trips the subset it knows about.

The missing user-visible cases are exactly in that class: captions, provenance context, and structured location metadata such as `Iptc4xmpExt:LocationShown`.

## Goals

- Preserve copy-safe top-level metadata from source page sidecars into rendered page and crop sidecars
- Keep rendered provenance explicit without violating the repository rule that `dc:source` on view outputs names archive scan lineage
- Prevent later rewrite paths from deleting inherited top-level metadata unless they are intentionally replacing it
- Make crop refresh logic able to recover parent page context without depending on archive-scan-only parsing

## Non-Goals

- Defining a generic arbitrary-XML copier for every XMP namespace
- Changing region geometry, face-region semantics, or CTM behavior
- Introducing new metadata fields beyond what already exists upstream

## Decisions

### Copy-safe top-level metadata is inherited by contract

The rendered-output contract should explicitly preserve the top-level fields that are already meaningful on rendered JPEGs and do not depend on source-only image geometry. At minimum this includes:

- `dc:title`
- `dc:description`
- `dc:subject`
- `dc:date`
- `xmp:CreateDate`
- `exif:DateTimeOriginal`
- `imago:AlbumTitle`
- `imago:OCRText`
- `imago:AuthorText`
- `imago:SceneText`
- `photoshop:City`
- `photoshop:State`
- `photoshop:Country`
- `Iptc4xmpExt:Sublocation`
- `Iptc4xmpExt:LocationCreated`
- `Iptc4xmpExt:LocationShown`

These are the fields the user expects to survive from page source XMP into rendered views and photo crops.

### Parent view metadata is authoritative for crop inheritance

Crop sidecars should inherit from the page view sidecar rather than directly from the archive page sidecar whenever both are available. The page view sidecar is the accepted rendered-page state after any page-level enrichment, repair, or location reconciliation. Crops should reflect that accepted page-level metadata, not bypass it.

### Provenance has two layers and both matter

Rendered JPEGs need both archive lineage and immediate derivation:

- `dc:source` keeps archive scan lineage for all view JPEG outputs, including crop JPEGs
- `xmpMM:DerivedFrom` and `xmpMM:Pantry` capture the immediate rendered-parent relationship

That keeps repository semantics consistent while still making the crop-to-page relationship explicit.

### Later rewrite steps preserve fields they do not own

Face refresh and similar rewrite flows own people-related metadata and pipeline state. They do not own page captions, `LocationShown`, `LocationCreated`, or archive lineage. Those fields should therefore round-trip unchanged unless that exact step is intentionally recomputing them.

### Crop refresh needs rendered-parent context resolution

Crop sidecars cannot rely on archive-scan-only `dc:source` parsing to recover page context, because crop outputs also need access to their parent rendered page metadata. The refresh path should therefore resolve parent context from rendered-parent provenance or an equivalent explicit parent-view linkage.

## Resulting Data Flow

```text
archive/page sidecar
  copy-safe top-level metadata
        |
        v
page view sidecar
  + render provenance
        |
        +--> later page rewrite steps preserve inherited metadata
        |
        v
crop sidecar
  inherits accepted page-view metadata
  + crop provenance to parent view
  + archive lineage in dc:source
        |
        v
later crop rewrite steps preserve inherited metadata
```
