# AGENTS.md

Purpose: operating rules for the `photoalbums/` project.

## Project Skills

- Do not use brittle regex or string replacements to edit AI model responses. Improve the prompt instead so the model produces the right JSON.
- `photoalbums/prompts/` is the runtime prompt source of truth for the `photoalbums/` pipeline.

## Photo Album File Naming Convention

All photo album files use this naming scheme:

```text
{Collection}_{Year}_B{book}_P{page}_{type}.{ext}
```

| Type token | Role | Archive ext | View ext |
| ------------ | ------ | ------------- | ---------- |
| `_S##` | Raw scan | `.tif` | - |
| `_D##-##` | Derived image | `.tif` | - |
| `_V` | View page (any scan count) | - | `.jpg` |
| `_D##-##_V` | View derived image | - | `.jpg` |

Rules:

- `_V` always and only marks a view output. `_S##` always and only marks an archive scan.
- `_D##-##` identifies a derived image; append `_V` for the view JPEG.
- Archive files are `.tif` and `.png`; view files are `.jpg` with no exceptions.
- `dc:source` on any view file references the archive TIF scan(s) it was derived from.
- Pages are numbered starting at `P01`. `P00` is not valid.
- XMP sidecars share the same stem as their companion image file.

## Data and Schema

- Photo albums are located in `C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums`.
- Prefer standard XMP schema fields over custom namespaces for `imago`.

## UI and Terminology

- Prefer precise naming that matches behavior.
- The current preferred term is `Gamma Correction`, not `Brightness`.
- Keep user-facing labels aligned with metadata and renderer semantics.

## Preview Render Behavior

- Preview should apply only the current wizard stage effects by default:
  - Review step: bad-frame repair only.
  - Gamma Correction step: gamma correction only.
  - Summary step: combined output.
