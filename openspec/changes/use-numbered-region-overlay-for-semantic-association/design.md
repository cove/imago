## Context

Gemma caption matching currently relies on a two-step coincidence: the prompt asks the model to number photos left-to-right/top-to-bottom, and the code separately sorts Docling regions into that same inferred order before merging the model response. That works only as long as the model and the scanline sort make the same judgment about row grouping, ambiguous layouts, and caption adjacency.

The real missing contract is not caption quality but region identity. Docling already determines which photo regions exist on the page, and the repository already has an overlay renderer that can draw numbered boxes over those regions for debugging. This change turns that visual artifact into a first-class prompt input so Gemma can associate captions and location/date semantics to the exact accepted regions Docling identified, without inventing a separate numbering scheme.

This change does not move geometry ownership or provenance ownership into Gemma. Geometry remains owned by Docling. Deterministic metadata assembly remains owned by code. Geocoding remains owned by Nominatim. Face identity remains owned by `buffalo_l`. Object detection remains owned by YOLO.

## Goals / Non-Goals

**Goals:**
- Make the Docling -> Gemma handoff explicit by rendering a numbered region-association overlay from the accepted regions.
- Make the visible overlay numbers the authoritative region identifiers for Gemma caption/location/date association.
- Remove coordinate-based reading-order inference from the caption-matching merge path.
- Keep the semantic-association prompt artifact minimal so Gemma sees only the page plus precise outlines and visible region numbers.

**Non-Goals:**
- Replacing Docling region geometry or changing the accepted-region validation rules.
- Moving people identity into Gemma; `buffalo_l` remains the authority for person identity.
- Moving geocoding into Gemma; Nominatim remains the authority for GPS/location resolution.
- Defining a general unified metadata resolver in this change. This proposal only fixes the region-identity contract used during semantic association.

## Decisions

### D1: The numbered overlay becomes a pipeline input, not a debug-only byproduct
**Decision:** Generate a dedicated region-association overlay image from the accepted Docling regions and feed that image to Gemma during caption/location association.

**Rationale:** The core problem is that Gemma has no direct way to know which regions Docling accepted. A visual overlay solves that problem using the exact geometry Docling already produced, instead of asking the model to recreate region numbering from the raw page.

**Alternative considered:** Keep the current left-to-right/top-to-bottom numbering prompt. Rejected because it preserves the fragile dual-numbering contract between model and code.

### D2: Use a single overlay image, not a pair of original-plus-overlay prompt images
**Decision:** The semantic-association prompt uses the overlay image alone.

**Rationale:** The overlay preserves the original page image while adding only precise outlines and visible numbers, so Gemma still sees the page content and also gets the region identity contract in one image. Using a single image avoids asking the model to reconcile two views of the same page.

**Alternative considered:** Send both the original page image and the overlay image. Rejected for now because the overlay already contains the underlying page content and the user expects the precise overlay boxes not to interfere materially with Gemma's understanding.

### D3: The overlay must be prompt-safe and minimal
**Decision:** The region-association overlay contains only the original page image plus outlines rendered from the accepted regions' normalized geometry and visible region numbers. It does not include captions, person names, validation notes, or other debug annotations.

**Rationale:** The existing accepted debug image is helpful for humans, but it can include text that is irrelevant or stale for prompt use. The prompt artifact should communicate only region identity.

**Alternative considered:** Reuse the generic accepted debug image as-is. Rejected because prompt inputs must not leak incidental debug labels into model reasoning.

### D4: Visible overlay numbers replace inferred reading-order numbering
**Decision:** Gemma returns caption/location/date associations keyed directly to the visible overlay numbers, and the merge path maps those keys straight back to the corresponding accepted regions.

**Rationale:** Once the overlay numbers are authoritative, scanline sorting is no longer needed for semantic association. This removes a class of mismatches where Docling geometry is correct but the inferred numbering diverges from the model's interpretation.

**Alternative considered:** Keep scanline sort as a fallback beneath the overlay contract. Rejected because a hidden second numbering system would weaken the new contract and make failures harder to reason about.

### D5: Scope Gemma association to semantics, not identity or geocoding
**Decision:** Gemma uses the overlay to associate captions and location/date semantics to regions. Person identity continues to come from `buffalo_l`, and location resolution continues to come from Nominatim.

**Rationale:** This change is about clarifying region identity during semantic association, not broadening Gemma's ownership. Keeping model boundaries intact avoids turning one prompt improvement into a larger metadata-ownership refactor.

**Alternative considered:** Expand the overlay-driven prompt so Gemma also returns people identity and fully resolved locations. Rejected because those concerns already have stronger deterministic or specialized owners in the existing pipeline.

## Risks / Trade-offs

- [Overlay labels obscure meaningful page text] → Mitigation: keep labels minimal, use small visible numbers, and place labels at region edges where possible.
- [Existing prompt/debug renderer is reused accidentally with extra annotation text] → Mitigation: define a separate prompt-safe overlay mode and spec it explicitly.
- [Response keys drift from overlay numbering] → Mitigation: require the prompt to use the visible overlay numbers as authoritative identifiers and ignore keys that do not match an accepted region.
- [This improves caption/location association but leaves broader metadata-assembly complexity in place] → Mitigation: keep the scope narrow and document that unified metadata resolution is a separate architectural concern.

## Migration Plan

1. Add the prompt-safe numbered region-association overlay artifact generation.
2. Update Gemma caption matching to use the overlay-based numbering contract.
3. Remove scanline-sort-based numbering from the merge path once overlay-keyed responses are in place.
4. Re-run affected pages through the region-association path to verify captions and per-photo location choices still map correctly.

Rollback is straightforward: revert the prompt contract and merge logic to the existing reading-order-based path, and stop generating the prompt overlay artifact.

## Open Questions

- Should the response keys stay `photo-N` for backward familiarity, or change to `region-N` to match the new contract more honestly?
- Should the overlay numbering always mirror stored `region.index + 1`, or should a separate prompt-specific identifier be introduced later if region persistence semantics change?
