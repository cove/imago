## Why

The rendered page directory is still named `_View`, but that name no longer matches its role beside `_Archive` and `_Photos`. The repo and the live album sidecars also encode `_View` in path references, so renaming the directory requires both a runtime contract change and an XMP path migration.

## What Changes

- **BREAKING** Rename the canonical rendered page directory suffix from `_View` to `_Pages`
- Add a lightweight shared album-directory naming helper so `_Archive` / `_Pages` / `_Photos` relationships are defined in one place
- Update page-directory derivation, sibling lookup, and folder scans in `photoalbums` and `cast` to use `_Pages`
- Update user-facing defaults and examples, including Cast ingest globs and CLI/help text, to use `*_Pages`
- Rewrite new XMP path/provenance references so any page-JPEG reference points at `*_Pages`
- Migrate existing XMP sidecars under the photo album root that currently encode `../*_View/...` paths so they reference `../*_Pages/...`
- Update repo-tracked docs, eval fixtures, and tests that currently embed `_View`

## Capabilities

### New Capabilities
- `page-directory-layout`: Canonical naming and lookup rules for rendered page directories, including runtime path derivation, ingest defaults, and the shared naming helper
- `page-reference-migration`: Migration and write rules for XMP references that point at rendered page JPEGs

### Modified Capabilities
- `view-xmp-regions`: Page and crop XMP metadata must treat `*_Pages` as the rendered page location when referencing page-side sidecars and JPEGs

## Impact

- Affected code in `photoalbums`, `cast`, and XMP/provenance helpers that derive or persist rendered page paths
- Existing album sidecars under `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums` need a one-time `_View` to `_Pages` path rewrite
- Repo-tracked eval fixtures and tests that hard-code `_View` paths need to be migrated
- Any external workflow expecting `*_View` paths will need to move to `*_Pages`
