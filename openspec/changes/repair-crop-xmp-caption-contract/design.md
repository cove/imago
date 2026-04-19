## Context

The current XMP writer uses one generic `dc:description` alt-text builder for both page sidecars and crop sidecars. That builder makes OCR and scene text win `x-default` and stores the explicit caption in custom entries such as `x-caption`, `x-author`, and `x-scene`.

That behavior is tolerable for page sidecars, where `dc:description` is being used as a searchable page-text summary, but it is wrong for crop sidecars. Crop metadata already has a canonical caption source in the parent page's `mwg-rs:Name`, and that caption is what users expect to see as the default description of the cropped photo.

The change also affects existing live sidecars. The repo already has examples of targeted, in-place XMP migrations that preserve unrelated metadata, so this should follow the same model instead of requiring image regeneration.

## Goals / Non-Goals

**Goals:**
- Make crop sidecar `dc:description` default to the region caption rather than inherited page OCR
- Keep page OCR and scene text available as separate fields without custom `dc:description` language tags
- Rename inherited crop page OCR to `imago:ParentOCRText` so the field meaning is explicit
- Provide an in-place migration and verification flow for existing sidecars
- Keep migration idempotent and preserve unrelated XMP content

**Non-Goals:**
- Re-running Docling, OCR, or crop image generation as the primary repair path
- Introducing a generic metadata migration framework beyond this targeted rewrite
- Changing how region geometry or `mwg-rs:Name` captions are detected
- Defining crop-local OCR extraction for the cropped photo itself

## Decisions

### 1. Split page and crop description semantics instead of sharing one alt-text contract

Page sidecars and crop sidecars do not have the same `dc:description` job. Page sidecars use it as a readable page-text summary. Crop sidecars use it as the visible caption of a single derived photo. The writer contract should therefore distinguish those cases instead of routing both through the same custom alt-text layout.

Alternative considered:
- Keep the shared writer behavior and adjust only the crop caller. Rejected because it leaves the invalid custom `dc:description` entries in place and keeps the contract ambiguous.

### 2. Crop captions own `dc:description` `x-default`

For crop sidecars, the resolved region caption should be written as the `x-default` value of `dc:description`. If no crop caption exists, the field may fall back to the page description exactly as today, but page OCR must not override a non-empty crop caption.

Alternative considered:
- Keep storing crop captions in a secondary field such as `x-caption`. Rejected because the default-visible caption is the thing users care about, and `x-caption` is not part of the standard Lang Alt model.

### 3. Remove custom `dc:description` language tags from the contract

The system should stop writing `x-caption`, `x-author`, and `x-scene` entries under `dc:description`. OCR and scene text already have dedicated `imago:*` fields, so the custom alt entries are redundant and make the metadata contract harder to reason about.

Alternative considered:
- Continue writing the custom entries as compatibility data. Rejected because it prolongs the legacy format and makes migration non-final.

### 4. Keep page OCR on crops under `imago:ParentOCRText`

The inherited OCR currently copied from the page view sidecar is still useful for search, troubleshooting, and later context recovery. It should remain on crop sidecars, but under a field whose name makes the provenance honest: `imago:ParentOCRText`.

Alternative considered:
- Drop inherited page OCR from crops entirely. Rejected because later crop refresh paths currently rely on parent page text context, and removing it would make migration more destructive.

Alternative considered:
- Keep using `imago:OCRText` on crop sidecars. Rejected because it falsely implies the OCR was extracted from the crop itself.

### 5. Page summaries should use labeled sections, not punctuation separators

When page OCR and scene text are both present, the page-side `dc:description` summary should use explicit section labels with blank lines, for example `OCR:` followed by `Scene Text:`. This reads cleanly in metadata viewers and avoids turning punctuation such as `|` into presentation syntax that can collide with real OCR content.

Alternative considered:
- Join sections with a pipe delimiter. Rejected because OCR text is multiline and may already contain punctuation, making pipe separation harder to parse and less readable.

### 6. Migration should reconstruct captions from authoritative sources before rewriting

The migration should prefer the strongest available caption source in this order:

1. Parent page region `mwg-rs:Name`
2. Existing crop `dc:description` `x-caption` value, if present
3. Existing crop logical description if it is already caption-like
4. Existing page description fallback when no per-crop caption can be recovered

This keeps the migrated output aligned with the canonical region contract and avoids depending entirely on legacy crop-side custom entries.

Alternative considered:
- Rewrite only based on crop-local fields. Rejected because parent page `mwg-rs:Name` is already the intended source of truth.

## Risks / Trade-offs

- [Legacy sidecars may have incomplete caption data] -> Mitigation: use parent page `mwg-rs:Name` first and allow page-description fallback only when no crop caption can be recovered
- [Changing field names may break code that expects `imago:OCRText` on crops] -> Mitigation: update refresh/read paths in the same change and add tests for crop-specific versus parent-inherited text
- [Migration may touch manual metadata unintentionally] -> Mitigation: rewrite only the owned description and inherited-parent OCR fields, preserving unrelated XMP content byte-for-byte where possible
- [Page summary labels slightly change existing page `dc:description` text] -> Mitigation: keep raw OCR and scene text unchanged in their dedicated fields and scope the visible text change to the summary field only

## Migration Plan

1. Update the XMP contract and writer logic for page sidecars and crop sidecars
2. Add read-path support for `imago:ParentOCRText` on crop sidecars
3. Add a targeted verification mode that reports sidecars still using the legacy caption layout
4. Add an in-place migration that rewrites existing sidecars to the new layout without touching image files
5. Run the migration against the live photo album root
6. Verify that no sidecars still contain legacy custom description entries such as `x-caption`, `x-author`, or `x-scene`

Rollback is a code rollback plus restoration of the pre-migration sidecars if needed. Because the migration mutates persisted XMP, rollback is not just a code revert.

## Open Questions

- None. The user explicitly chose `imago:ParentOCRText` for inherited crop OCR, and the preferred separator strategy for page summaries is to use labeled sections instead of punctuation-only delimiters.
