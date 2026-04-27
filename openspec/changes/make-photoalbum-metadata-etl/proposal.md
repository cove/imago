## Why

Photo album metadata currently flows through a mix of readers, writers, propagation steps, repair paths, and implicit fallback rules. This makes it hard to understand why a crop received a caption, date, location, or person value, and it creates regressions when one path silently substitutes another field.

This change makes the photo album metadata pipeline ETL-like: extraction reads exact source facts, transformation applies named ownership and precedence rules, and loading serializes the resolved records without hidden fallback.

## What Changes

- Split metadata handling into explicit extract, transform, and load contracts.
- Add raw sidecar and region extraction APIs that return exact fields without coalescing.
- Add canonical page, region, and crop metadata records with field-level source/provenance.
- Move fallback, inheritance, and precedence rules into named resolver transforms.
- Restrict XMP writers to serialization and declared preservation policy.
- Make hidden fallback behavior invalid for normal pipeline writes; allow it only in named migration, repair, or display-only code paths.
- Add diagnostics so resolved fields can report their source, such as `page.region[1].mwg-rs:Name`.

## Capabilities

### New Capabilities

- `photoalbum-metadata-etl`: Defines extract/transform/load boundaries, canonical metadata records, fallback policy, and field provenance for photo album metadata flow.

### Modified Capabilities

- `metadata-pipeline-ownership`: Tightens the existing ownership contract so the resolver is the only place that may apply precedence, inheritance, or fallback, and XMP writers may not infer missing values from unrelated fields.

## Impact

- `photoalbums/lib/xmp_sidecar.py`: raw extraction APIs, reduced implicit coalescing on canonical write paths, provenance serialization helpers.
- `photoalbums/lib/metadata_resolver.py`: canonical transform rules for page, region, and crop metadata.
- `photoalbums/lib/ai_index_propagate.py`: consume resolved crop records instead of independently applying fallback.
- `photoalbums/lib/ai_photo_crops.py`: create crop sidecars from resolved crop records.
- Repair and migration scripts: keep fallback behavior, but label it as repair/migration logic and report source fields.
- Tests: add coverage for no hidden fallback, source provenance, field ownership, and existing manual-field preservation policy.
