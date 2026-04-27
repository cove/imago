## Context

The photo album pipeline already has explicit producer ownership in `metadata-pipeline-ownership`, but effective metadata is still assembled in several places. `read_ai_sidecar_state()` coalesces values for convenience, `write_xmp_sidecar()` merges new values with existing XMP, crop creation resolves captions and inherited page fields, and `propagate-to-crops` resolves another subset later.

That behavior makes local fixes easy but global behavior hard to reason about. A missing crop caption can be filled from region `mwg-rs:Name`, `CaptionHint`, page `dc:description`, existing crop `dc:description`, or migration/repair logic depending on which code path ran. The desired direction is an ETL contract where normal pipeline writes are explainable from exact source fields.

## Goals / Non-Goals

**Goals:**

- Separate raw extraction from metadata resolution.
- Make normal pipeline fallback rules explicit and centralized.
- Make crop/page XMP writes consume canonical resolved records.
- Preserve existing manual metadata only through declared preservation rules.
- Record enough field provenance to debug why a value was written.
- Keep the change incremental so existing sidecars continue to read and migrate safely.

**Non-Goals:**

- Replacing XMP as the persisted metadata format.
- Reprocessing all albums as part of the code change.
- Changing AI prompts or model behavior.
- Removing repair and migration tools that intentionally recover data from legacy layouts.
- Adding a database for intermediate metadata state.

## Decisions

### 1. Add raw extraction APIs alongside existing effective readers

Introduce raw extraction functions that return exact XMP fields without coalescing or fallback:

```text
read_raw_sidecar(path) -> RawSidecar
read_raw_region_list(path, dimensions) -> list[RawRegion]
```

Existing effective readers can remain for UI/display and compatibility, but normal pipeline transforms should use raw extraction. This avoids breaking unrelated callers while giving the ETL path a clean input contract.

**Alternative considered:** Rewrite `read_ai_sidecar_state()` to be raw-only. Rejected because many existing callers rely on its effective/coalesced behavior and the blast radius is too high for the first step.

### 2. Introduce canonical metadata records

Define small dataclasses or typed dictionaries for:

```text
RawSidecar
RawRegion
ResolvedPageMetadata
ResolvedRegionMetadata
ResolvedCropMetadata
ResolvedField(value, source, policy)
```

The resolved records are the only inputs normal pipeline writers should consume. Field provenance can be serialized into `imago:Detections["resolved_sources"]` or a similarly named sub-object.

**Alternative considered:** Continue passing dictionaries between steps. Rejected because loosely shaped dictionaries are part of why source ownership is unclear.

### 3. Centralize all precedence and fallback in resolver transforms

Resolver functions own the business rules:

```text
resolve_page_metadata(raw_page, producer_outputs) -> ResolvedPageMetadata
resolve_region_metadata(raw_region, metadata_photo) -> ResolvedRegionMetadata
resolve_crop_metadata(raw_crop, resolved_page, resolved_region, policy) -> ResolvedCropMetadata
```

Normal pipeline rules should prefer absence over hidden substitution. If a field is inherited or preserved, the resolver must name that policy in the resolved field source.

**Alternative considered:** Add more targeted fallback to propagation and crop creation. Rejected because it fixes symptoms while preserving multiple competing data-flow rules.

### 4. Make writers serialization-only for canonical writes

Keep low-level XMP serialization helpers, but route normal pipeline writes through explicit APIs:

```text
write_page_metadata(path, ResolvedPageMetadata)
write_crop_metadata(path, ResolvedCropMetadata)
```

These APIs may preserve unrelated XMP fields such as document IDs and manual tags, but they must not infer missing canonical values from unrelated fields. Preservation rules should be declared per field.

**Alternative considered:** Make `write_xmp_sidecar()` accept more flags to control fallback. Rejected because flags would spread policy decisions across call sites.

### 5. Treat repair, migration, and display fallback as separate modes

Fallback is allowed when a command is explicitly repairing legacy data, migrating old sidecars, or displaying an effective value in a UI. Those modes must not silently write fallback-derived values through the normal pipeline unless they record the source and policy.

**Alternative considered:** Ban fallback globally. Rejected because legacy sidecars need repair and operators need effective display values.

### 6. Migrate one path at a time

Start with the path that caused the recent confusion:

```text
page mwg-rs:RegionList[n].mwg-rs:Name
  -> ResolvedRegionMetadata.caption
  -> ResolvedCropMetadata.description
  -> crop dc:description
```

Then migrate location, date, people, OCR, and album title fields in separate focused tasks.

## Risks / Trade-offs

- Existing code may depend on implicit coalescing -> keep compatibility readers and migrate normal pipeline callers incrementally.
- Provenance may add noise to XMP payloads -> store compact source identifiers and only for fields written by resolver-owned paths.
- Writers may still need to preserve manual edits -> define preservation rules per field before changing behavior.
- The transition can temporarily have old and new flows -> add tests that fail when normal pipeline code uses effective readers for canonical writes.

## Migration Plan

1. Add raw extraction APIs and tests without changing existing behavior.
2. Add resolved metadata record types and resolver functions for crop descriptions.
3. Route crop creation and `propagate-to-crops` description writes through the resolver.
4. Add provenance serialization for resolved crop description.
5. Move location/date/person propagation into resolved crop records.
6. Restrict canonical writer entry points to resolved records.
7. Keep legacy repair/migration code on explicit fallback paths and label their output sources.
8. Run existing focused tests and then a sample page/crop validation pass before broad reprocessing.
