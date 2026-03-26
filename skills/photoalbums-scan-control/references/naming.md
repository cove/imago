# Naming

This reference owns the current scan naming convention for the incoming scan watcher.

## Current Convention

- Incoming file name: `incoming_scan.tif`
- Stored scan files: `{prefix}_P##_S##.tif`
- Retry rule: keep the same page number and increase the scan number
- The skill, not Python code, decides the `prefix`

## Operational Rule

- If this album changes to a different naming convention, update this reference first.
- Do not edit the watcher code to encode album-specific naming rules.
