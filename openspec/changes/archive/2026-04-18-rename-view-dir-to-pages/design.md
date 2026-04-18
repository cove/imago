## Context

The current system uses `_Archive` for archival scans, `_Photos` for derived crop outputs, and `_View` for rendered page JPEGs. `_View` is now the inconsistent term in that set, but the suffix is baked into runtime path helpers, Cast ingest defaults, repo-tracked fixtures, and live XMP sidecars that store relative page references such as `../Egypt_1975_B00_View/Egypt_1975_B00_P26_V.jpg`.

This is a cross-cutting rename rather than a single-module edit. The change must update both code and on-disk metadata so runtime behavior, persisted provenance, and existing sidecars agree on the same directory contract.

## Goals / Non-Goals

**Goals:**
- Make `_Pages` the only canonical rendered page directory suffix
- Consolidate album directory suffix and sibling-derivation logic behind one small shared helper surface
- Update all runtime path derivation and lookup logic that currently depends on `_View`
- Ensure new XMP writes store `*_Pages` page references
- Migrate existing XMP sidecars that still point at `_View`
- Update tests, evals, and docs so the repo has no mixed `_View`/`_Pages` contract

**Non-Goals:**
- Renaming `_Archive`, `_Photos`, or the filename-level `_V` marker
- Supporting `_View` and `_Pages` as parallel long-term aliases
- Building a configurable directory-layout framework
- Designing a generic migration framework beyond the specific XMP rewrite needed here

## Decisions

### 1. Treat `_Pages` as a strict cutover rather than a compatibility alias

The code will stop treating `_View` as a supported runtime directory suffix. This keeps the contract single-valued and avoids indefinite branching in sibling resolution, ingest defaults, and provenance writes.

Alternative considered:
- Keep `_View` as a read fallback. Rejected because it broadens the scope into a compatibility feature and leaves the code and data model in a mixed state.

### 2. Split the change into runtime rename plus explicit XMP migration

The implementation will not assume that renaming directories is enough. It will also update persisted XMP references that encode `_View` in `stRef:filePath` or equivalent page-reference fields.

Alternative considered:
- Change only runtime code and let stale XMP references linger. Rejected because existing crop/provenance sidecars would continue pointing at non-canonical paths after the directory rename.

### 3. Consolidate naming logic behind a small shared helper

The implementation should add a lightweight shared naming/layout surface for album directory suffixes and sibling derivation instead of repeating `endswith`, `replace`, and hand-built suffix logic across modules. The helper should stay narrow: canonical suffix constants plus a few sibling/identity helpers for `_Archive`, `_Pages`, and `_Photos`.

Alternative considered:
- Leave the rename as scattered local string edits. Rejected because the current spread of duplicated suffix logic is what made this change cross-cutting in the first place.

Alternative considered:
- Build a generic configurable layout abstraction. Rejected because the repo only needs one canonical naming contract, not a framework for arbitrary layouts.

### 4. Update write paths at the same time as migrating existing sidecars

Any helper that writes new page-relative XMP references must switch to `_Pages` in the same change that migrates existing sidecars. This prevents new stale `_View` references from being written immediately after migration.

Alternative considered:
- Migrate existing sidecars first and defer writer changes. Rejected because it would allow regressions and create non-idempotent migration behavior.

### 5. Limit data migration to page-reference path rewrites

The one-off migration should update encoded page paths from `_View` to `_Pages` without rewriting unrelated XMP fields or changing the image filename tokens.

Alternative considered:
- Rebuild XMP sidecars wholesale. Rejected because it risks touching unrelated manual metadata and violates the project's preference to preserve existing sidecar content outside the targeted change.

## Risks / Trade-offs

- [Cross-cutting path assumptions] -> Mitigation: add the shared naming helper first, then convert direct suffix checks and string replacements to use the new canonical suffix
- [Persisted XMP references missed by the migration] -> Mitigation: search the live album root for `_View` in `.xmp` files before and after the implementation, and add tests for the rewritten reference shapes
- [Mixed repo terminology after the code change] -> Mitigation: migrate docs, eval fixtures, and help text in the same change rather than as follow-up cleanup
- [Breaking external workflows that still expect `_View`] -> Mitigation: document the rename in the proposal/tasks and keep the scope explicit that this is a breaking cutover

## Migration Plan

1. Add the shared album-directory naming helper for `_Archive`, `_Pages`, and `_Photos`
2. Update code-level page directory helpers and writer logic to use `_Pages`
3. Update tests, docs, eval fixtures, and defaults that embed `_View`
4. Run a targeted migration over album `.xmp` sidecars under the configured photo album root to replace page references from `_View` to `_Pages`
5. Validate that no `.xmp` files under the album root still contain `_View`
6. Run repo validation commands and fix any remaining path assumptions

Rollback is a code rollback plus restoration of the pre-migration XMP files if needed. Because this change mutates existing sidecars, rollback is not just a code revert.

## Open Questions

- None. The user chose a strict cutover and explicitly confirmed that the encoded XMP paths also need to be updated.
