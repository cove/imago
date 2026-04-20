## Context

The current metadata pipeline already has multiple specialized producers:

- Docling detects photo regions and page layout.
- Gemma associates captions and location/date semantics.
- `buffalo_l` identifies people.
- YOLO detects objects.
- Nominatim resolves location queries into normalized GPS/location payloads.
- XMP writers serialize the final results and maintain provenance.

What is missing is a clear contract for how those producers fit together. Some fields are written as page facts, some are stored as region-scoped hints, some are inherited into crops, and some are re-decided late in write paths. That makes it easy for one stage to accidentally overwrite or misinterpret data produced by another stage, which is exactly the pattern behind the recent caption/location/`PersonInImage` contamination issues.

The numbered region overlay is part of the solution, but not the whole solution. It fixes the Docling -> Gemma identity handoff. The larger architectural need is to make ownership explicit across the entire pipeline so every stage has a clear responsibility and the deterministic resolver has one place to compute effective page and crop metadata.

## Goals / Non-Goals

**Goals:**
- Define explicit ownership boundaries for geometry, semantics, people identity, objects, geocoding, effective metadata resolution, and XMP serialization.
- Introduce a deterministic metadata resolver stage that combines producer outputs into effective page and crop metadata before write.
- Make the numbered region overlay the authoritative contract for Docling -> Gemma region identity.
- Clarify that final XMP fields such as top-level location values and `PersonInImage` are resolved outputs, not raw model-owned facts.

**Non-Goals:**
- Replacing Docling with a different layout system.
- Replacing `buffalo_l`, YOLO, or Nominatim with Gemma outputs.
- Collapsing all metadata generation into a single Gemma prompt or making Gemma emit provenance/XMP-only fields.
- Implementing a complete end-state refactor of every metadata write path in this change. This design defines the contract first.

## Decisions

### D1: Treat the pipeline as multiple fact producers feeding one deterministic resolver
**Decision:** Model and service outputs are treated as producer-owned facts that flow into a deterministic metadata resolver before XMP write.

```text
Docling    -> geometry facts
Gemma      -> caption/location/date semantics
buffalo_l  -> people identity
YOLO       -> object facts
Nominatim  -> resolved location payloads
Resolver   -> effective page/crop metadata
Writers    -> XMP serialization + provenance
```

**Rationale:** This keeps each subsystem specialized and prevents later write paths from re-deriving upstream intent ad hoc.

**Alternative considered:** Let each write path continue to assemble the subset of fields it needs from raw sidecar state. Rejected because that is the current failure pattern.

### D2: Make ownership explicit by metadata layer, not just by file or step
**Decision:** The design distinguishes:
- source facts
- region assignments
- manual overrides
- effective metadata
- serialized XMP fields

**Rationale:** Many bugs come from conflating those layers. For example, `imago:LocationAssigned` on a region is not the same thing as the final top-level crop `photoshop:City`, and region `person_names` candidates are not the same thing as final `Iptc4xmpExt:PersonInImage`.

**Alternative considered:** Define ownership only in terms of CLI steps such as `detect-regions` or `ai-index`. Rejected because the bug surface crosses step boundaries and file scopes.

### D3: The numbered region overlay is the Docling -> Gemma identity contract
**Decision:** The accepted Docling regions are rendered into a prompt-safe numbered overlay image, and the visible overlay numbers become the authoritative identifiers for Gemma semantic association.

**Rationale:** This removes the implicit dual-numbering contract where Gemma invents `photo-N` ordering and the code separately tries to reproduce the same order with scanline sorting.

**Alternative considered:** Continue using left-to-right/top-to-bottom numbering in the prompt. Rejected because it keeps region identity implicit and fragile.

### D4: Gemma owns only semantic association, not identity, objects, or geocoding
**Decision:** Gemma uses the numbered overlay to associate captions and location/date semantics to accepted regions. It does not own people identity, object identity, resolved GPS coordinates, provenance, or final XMP field layout.

**Rationale:** The repository already has stronger owners for those concerns. Broadening Gemma ownership would blur boundaries instead of clarifying them.

**Alternative considered:** Ask Gemma to emit a full XMP-shaped JSON payload for all fields. Rejected because many fields are deterministic, inherited, or owned by other systems.

### D5: Final page and crop fields are resolved outputs
**Decision:** Fields such as top-level crop location, top-level page location, and `PersonInImage` are computed by resolver precedence rules rather than owned directly by any one producer.

Representative examples:

```text
effective crop location =
  manual region override
  else region-assigned location
  else caption-matched LocationShown entry
  else page effective location

effective PersonInImage =
  buffalo_l identities / region candidates
  filtered against known location strings
```

**Rationale:** These fields are where cross-producer conflicts must be reconciled. Making them explicit resolver outputs gives the pipeline one place to explain why the final value is what it is.

**Alternative considered:** Continue to compute these values opportunistically in crop/page write paths. Rejected because that distributes precedence logic across multiple callers.

## Risks / Trade-offs

- [The ownership contract is broader than one bug fix] → Mitigation: keep this as an architectural change proposal and implement it incrementally through follow-on tasks.
- [Resolver scope grows too large and becomes a “god object”] → Mitigation: keep producers specialized and constrain the resolver to precedence, inheritance, and final field assembly.
- [Overlay-based Gemma association improves region identity but not every metadata issue] → Mitigation: treat it as one component of the larger ownership redesign, not the entire solution.
- [Existing write paths may partially conform and partially bypass the new contract during rollout] → Mitigation: document which paths are canonical first and migrate callers in explicit phases.

## Migration Plan

1. Define the ownership and resolver contract in specs.
2. Introduce the numbered region-association overlay and update Gemma caption matching to use it.
3. Move page/crop effective metadata assembly behind explicit resolver helpers.
4. Update write paths to consume resolved metadata rather than re-deriving precedence rules locally.
5. Add regression coverage around multi-location pages, crop inheritance, and `PersonInImage` contamination.

Rollback is a code rollback plus reversion to the current write-path-owned assembly behavior. Because this change is primarily contractual and architectural, rollout can happen incrementally behind the existing external interfaces.

## Open Questions

- Should the canonical intermediate representation be persisted in `imago:Detections`, or remain an in-memory contract first?
- Should overlay response keys be named `photo-N` for continuity or `region-N` for accuracy?
- How much of the existing page-side `ai-index` output should be normalized into the new resolver in the first implementation phase versus later follow-on changes?
