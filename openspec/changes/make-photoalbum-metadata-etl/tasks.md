## 1. Raw Extraction Contracts

- [ ] 1.1 Add raw sidecar extraction types for exact XMP fields used by page and crop metadata.
- [ ] 1.2 Add raw region extraction types for exact MWG-RS fields, including `mwg-rs:Name`, `imago:CaptionHint`, `imago:PhotoNumber`, location payloads, and person names.
- [ ] 1.3 Implement raw extraction helpers alongside existing effective readers without changing existing callers.
- [ ] 1.4 Add tests proving raw extraction does not fill missing fields from fallback sources.

## 2. Resolved Metadata Records

- [ ] 2.1 Add resolved field structure carrying value, source, and policy.
- [ ] 2.2 Add resolved page, region, and crop metadata structures.
- [ ] 2.3 Add compact provenance serialization under the detections payload for resolver-owned fields.
- [ ] 2.4 Add tests for provenance source strings such as `page.region[1].mwg-rs:Name`.

## 3. Resolver Transform Rules

- [ ] 3.1 Implement region caption resolution from raw `mwg-rs:Name` without implicit `CaptionHint` fallback.
- [ ] 3.2 Implement crop description resolution from resolved region caption, preserving an existing crop description only through an explicit preservation policy.
- [ ] 3.3 Move crop location resolution into resolved crop metadata while preserving existing override and assignment precedence.
- [ ] 3.4 Move crop people filtering into resolved crop metadata using existing location-name filtering rules.
- [ ] 3.5 Move crop date and album-title inheritance into resolved crop metadata.
- [ ] 3.6 Add tests for empty resolved values staying empty when no resolver policy applies.

## 4. Canonical Write Paths

- [ ] 4.1 Add page and crop metadata writer entry points that accept resolved records.
- [ ] 4.2 Ensure canonical writers serialize resolved values without calling effective/coalescing readers for missing fields.
- [ ] 4.3 Keep low-level XMP preservation for unrelated fields such as document IDs and unrelated manual subjects.
- [ ] 4.4 Add tests proving writers do not infer crop descriptions from page description, OCR text, caption hint, or location text.

## 5. Pipeline Integration

- [ ] 5.1 Route crop creation in `ai_photo_crops.py` through resolved crop metadata for description, location, people, date, and album title.
- [ ] 5.2 Route `propagate-to-crops` through resolved crop metadata instead of local fallback logic.
- [ ] 5.3 Update step payloads so propagation records field provenance for values it writes.
- [ ] 5.4 Add focused regression coverage for the P02 crop-caption case using page region `mwg-rs:Name` as the source.

## 6. Fallback Mode Isolation

- [ ] 6.1 Identify repair, migration, and display-only code paths that intentionally use fallback.
- [ ] 6.2 Rename or wrap fallback helpers so normal pipeline code cannot call them accidentally.
- [ ] 6.3 Ensure repair and migration fallback writes record their source policy.
- [ ] 6.4 Add tests that normal pipeline code paths do not use display/effective fallback helpers for canonical writes.

## 7. Validation

- [ ] 7.1 Run focused py_compile checks for changed modules.
- [ ] 7.2 Run focused photo album tests for sidecar, crop creation, propagation, and metadata resolver behavior.
- [ ] 7.3 Run `just dupes`, `just deadcode`, `just complexity`, and `just test`.
- [ ] 7.4 Run a sample page/crop pipeline pass and verify field provenance explains the written crop caption, location, people, date, and album title.
