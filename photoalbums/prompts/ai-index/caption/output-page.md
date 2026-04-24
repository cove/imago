`{"ocr_text": "", "author_text": "", "scene_text": "", "location_name": ""}`

- `ocr_text`: you're an OCR engine when processing this, look for all clearly legible visible text on the page, copied verbatim with original capitalization, punctuation, spacing, and real line breaks.
- `author_text`: you're an OCR engine when processing this, typed album-authored annotation text that's typed on a typewriter on strips of white paper, otherwise empty string.
- Recover the full `author_text` when the strip is visibly present but cropped in this scan and the supplied `ocr_text` contains the missing words.
- `scene_text`: you're an OCR engine when processing this, readable text visible inside photographs, otherwise empty string.
- `author_text` and `scene_text` are classified subsets of `ocr_text`, not replacements for it. Fill them whenever the classification is supported by the page.
- The example JSON uses empty strings as placeholders. Do not copy literal `...` from any example or schema text.
- `location_name`: concise geocoding query for GPS lookup when supported strongly enough by visible evidence; otherwise empty string.
- Include a country name in non-empty queries. Prefer `place, city/state, country` style queries over ambiguous short names when the broader geography is supported by the page or album context.
- `album_title`: for album title pages or cover pages - the full album title as a single-line storage string, with any printed line breaks replaced by spaces (e.g. `"Egypt 1975"`, `"Mainland China Book 11"`, `"Europe 1973 Egypt 1974"`). Empty string for all other pages.
- `ocr_lang`: BCP-47 language code of the primary non-English text in `author_text` or `scene_text` (e.g. `"zh"`, `"fr"`, `"ar"` for Chinese, French, Arabic). Use `"en"` for English-only text. Empty string when there is no visible text.
- Just return the JSON without any extra text or explanation.
