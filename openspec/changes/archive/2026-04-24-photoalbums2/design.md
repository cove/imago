## Context

`photoalbums/` conceptually implements a build pipeline with dependencies: scanned pages are rendered to view JPEGs, regions are detected, crops are extracted, face embeddings are matched against Cast, and AI steps (OCR, caption, location, date) annotate each image. Each stage writes to an XMP sidecar adjacent to the image file. The existing code has some Makefile-like behavior, but the abstraction is leaky, the retry logic is wrong for the observed failure modes, the prompt source of truth is fragmented, and the human review surface is a terminal plus a text editor.

At 7,000 photos per album and about 20s per vision call on an M1 Mac with LM Studio, a full clean rerun is on the order of 6 days of serial wall-clock. The main product constraint is not parallelism; it is avoiding unnecessary reruns. Recognizing one new face in Cast should trigger only the cropped photos that plausibly match that face, not a full rebuild of stitch, regions, crops, OCR, or unrelated photos.

This proposal does **not** refactor `photoalbums/`. It stands up a parallel project in the same monorepo so both pipelines can run side by side until a separate cutover proposal decides when and how to retire the legacy code. That bounds risk: a broken design in `photoalbums2/` cannot regress existing XMP.

## Goals / Non-Goals

**Goals:**
- Replace hand-rolled step-DAG and input-hash logic with Dagster assets over explicit page and photo work units
- Copy or import useful legacy library code while leaving legacy orchestration behind
- Replace custom LM Studio wrapper + inverted self-tuning retry with structured AI calls, static validation, and AI rewrite retries
- Consolidate every step's prompt, output schema, and retry ladder into one version-controlled prompt module
- Record rendered prompts, responses, sampler settings, retry history, output paths, and XMP diffs in Dagster metadata
- Make newly recognized Cast faces drive selective face refresh of only plausible matching photo work units
- Provide a per-photo human review UI showing image, exact prompt sent, exact response received, validation history, and run status
- Reuse existing XMP sidecars as starting materializations so a migrating user does not need to re-process work already done
- Keep `photoalbums/` running unchanged during the parallel period

**Non-Goals:**
- Full cutover from `photoalbums/` to `photoalbums2/` (separate later proposal)
- Removing, renaming, or editing any file under `photoalbums/` or its tests
- Rewriting stable helpers such as `xmp_sidecar.py` or `cast/storage.py` when they can be imported read-only
- Replacing LM Studio as the default inference provider
- Archiving or closing existing active OpenSpec changes under `photoalbums/`
- Full-album runs during this change; the acceptance gate is a vertical slice

## Decisions

### 1. Dagster as the pipeline spine, with page and photo work units

Each pipeline step is one Dagster asset. A partition is just Dagster's name for one work unit of an asset. In this pipeline there are two work-unit shapes:

- Page work unit: `album_id/book_id/page_id`, e.g. `Egypt_1975/B00/P26`
- Photo work unit: `album_id/book_id/page_id/photo_id`, e.g. `Egypt_1975/B00/P26/D03`

Page assets (`stitch`, `regions`, `crops`) run once per album page. Photo assets (`people`, `ocr`, `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, `semantic_review`) run once per cropped photo. This lets a user materialize one photo serially today, then increase Dagster concurrency later if more hardware is available.

The asset graph:

```text
render_settings
       |
   stitch[page]
       |
   regions[page]
       |
   crops[page]  ->  photo work units
                       |
new_face_recognized -> people[photo]
                       |
                   caption[photo]        ocr[photo]
                       |                  |
                       +--------+---------+
                                |
                     location_queries[photo]
                         |              |
                  gps_location[photo]  locations_shown[photo]
                         |              |
                         +------+-------+
                                |
                         semantic_review[photo]

                 ocr[photo] + caption[photo] -> date[photo]
                                |
                                v
                         semantic_review[photo]
```

Dagster's SQLite-backed event log stores run history, materializations, and logs. No Docker. `dagster dev` spawns a UI on port 3000 and should bind to `0.0.0.0` for remote access on the local network.

### 2. Cast recognition changes target plausible photo work units

`cast_store_version` as a whole is too coarse. The useful event is "a new face identity was recognized." When that happens, `photoalbums2` computes which existing crop embeddings are plausible matches for the newly named face and marks only those `people[photo]` work units stale.

If face matching for a photo changes the identified people, downstream `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, and `semantic_review` for that same photo may become stale. The page-level assets (`stitch`, `regions`, `crops`) and unrelated photos remain fresh.

### 3. Location assets preserve primary GPS and IPTC LocationShown

The legacy pipeline's `locations` step returns three outputs: `location`, `locations_shown`, and `location_shown_ran`. `photoalbums2` splits those concerns so they show up in the Dagster DAG:

- `location_queries[photo]`: LLM-backed step that returns a primary Nominatim query plus named shown-location queries from the crop/page context.
- `gps_location[photo]`: resolves the primary query into EXIF GPS fields, map datum, and scalar location fields such as city/state/country/sublocation.
- `locations_shown[photo]`: resolves named shown-location queries into IPTC `LocationShown` rows, including optional GPS coordinates and Nominatim source metadata.

The XMP candidate writer consumes both `gps_location` and `locations_shown`. It writes primary GPS to EXIF GPS fields and scalar location fields, and writes shown locations to the IPTC `LocationShown` bag.

### 4. Static validation and AI retry are separate from Dagster retry

There are two retry layers:

- Dagster retry handles technical failures: connection errors, timeouts, HTTP 5xx, file locks, and process crashes. These retries use the same logical input because the failure was infrastructure, not content.
- The AI retry ladder handles bad model output inside a single asset run. Each failed attempt records the prompt, response, sampler parameters, and failure reason in Dagster metadata, then retries with the next rung or a rewrite instruction.

Static validation stays conservative. It catches only mechanical failures such as empty responses, invalid or incomplete JSON, schema mismatch, impossible date format, unexpected double quotes inside caption text, prompt echo, obvious truncation, and known machine-garbage repetition. It rejects and asks the model to rewrite; it does not silently repair model text with brittle string replacement.

A typical caption ladder:

| Pass | Temperature | top_p | frequency_penalty | Notes |
| ---- | ----------- | ----- | ----------------- | ----- |
| 0 | 0.2 | 1.0 | 0.0 | default |
| 1 | 0.0 | 0.9 | 0.3 | dampen repetition |
| 2 | 0.0 | 0.85 | 0.6 | stronger anti-repetition |
| 3 | 0.0 | 0.85 | 0.6 | smaller image edge or simpler rewrite prompt |
| 4 | - | - | - | flag for human review; no further inference |

Semantic mistakes use an AI rewrite instruction, not brittle Python cleanup. Example: "The previous answer failed because the caption appears to describe the neighboring crop. Rewrite only the caption using the crop image as primary evidence and the page image as context."

### 5. Prompts are version-controlled files and visible in Dagster metadata

Every step owns one Python file in `photoalbums2/prompts/` that contains its system prompt, Jinja user-prompt template, output schema, and retry ladder. SKILL.md becomes documentation only; `photoalbums2` does not load SKILL.md at runtime.

If old prompt rules are worth keeping, they are copied into the step prompt file or generated into the prompt file before runtime. The final rendered prompt, prompt file path, prompt version/hash, response, retry history, and sampler settings are recorded in Dagster metadata so they can be inspected in the Dagster UI.

Prompt editing remains version-controlled. Dagster is the audit surface for "what prompt ran." A small Streamlit prompt/review UI may edit prompt source files or prompt draft files, but prompt changes still land as repository changes.

### 6. Static validators gate writes; semantic review handles visual mistakes

Static validators run at caption/location-query/date time, before any candidate XMP is written. They prevent malformed model output from reaching dry-run or staging output.

`semantic_review[photo]` uses the crop image, parent page image, canonical XMP, candidate XMP, `gps_location`, and `locations_shown` to identify mistakes that require visual judgment, such as a caption belonging to the wrong crop or a location that does not match the visible scene. When semantic review finds a mistake that the model can fix, it asks the responsible step to rewrite the relevant field. If the mistake remains unresolved after the step's rewrite ladder, the photo is marked `needs_review`.

### 7. Read legacy XMP as existing materializations; support dry-run, staging, and promotion

`photoalbums2` treats existing `.xmp` sidecars from `photoalbums/` as already-materialized starting points. A bootstrap/import or observe job emits per-work-unit materialization metadata keyed by sidecar content hash and relevant input hashes. This lets the new pipeline understand existing work without re-running 7,000 photos.

New writes have explicit output modes:

- Dry-run mode writes candidate XMP and diagnostics under `_debug/photoalbums2/`, leaving canonical sidecars and adjacent album sidecars untouched.
- Staging mode writes `.xmp.new` adjacent to the image.
- Promote mode copies reviewed staged output over canonical `.xmp` atomically, after showing or recording a diff.

Until promote runs, `photoalbums/` is unaffected.

### 8. Useful legacy code can be imported or copied selectively

`photoalbums2` can import stable, library-like helpers from `photoalbums/`, especially XMP I/O, XMP provenance, filename parsing, image geometry, crop helpers, geocoding/cache logic, and Cast storage access.

If useful code is small but trapped inside legacy orchestration, it may be copied into `photoalbums2` and reshaped around the new asset model. Large orchestration modules such as `commands.py`, `ai_index_runner.py`, and `ai_verify_crops.py` are not copied wholesale.

### 9. Review UI is a separate Streamlit app reading Dagster metadata

Dagster's own UI shows runs, lineage, retries, failures, paths, and metadata. It is the engineering/audit view. A small Streamlit app is still the human review surface: one photo at a time, with the page image, crop image, canonical XMP, candidate XMP, prompt, response, validation history, and rerun controls.

### 10. Scope is a vertical slice, not full cutover

The new project must run a small vertical slice through the full graph. The slice can run serially with concurrency set to 1. Later runs can increase concurrency if hardware resources allow it. A follow-on proposal handles promote-to-canonical policy, full cutover, and retirement of legacy `photoalbums/` orchestration.

## Risks / Trade-offs

- **Dagster learning curve**: Mitigation: use `dagster dev --host 0.0.0.0`, simple assets, explicit materialization, and no schedules/sensors in v1.
- **Pydantic AI version churn**: Mitigation: pin the version; keep the agent wrapper small enough that swapping to `instructor` + `litellm` is tractable.
- **Dry-run/staging drift**: Mitigation: dry-run output is clearly under `_debug/photoalbums2/`; staging uses `.xmp.new`; promotion is explicit.
- **Legacy sidecars as materializations**: Mitigation: use an explicit bootstrap/import job; mismatches mark work as not materialized, which is safe.
- **Static validator false positives**: Mitigation: validators start permissive and only catch known mechanical failures.
- **Semantic review cost**: Mitigation: run it as part of the vertical slice and later tune when to run it, but do not replace it with brittle string cleanup.
- **At 7,000 photos, a full run is expensive**: Mitigation: vertical slice first, serial by default, optional concurrency later.

## Migration Plan

1. Land the new project skeleton under `photoalbums2/` with its own `pyproject.toml`.
2. Copy or import only the stable helper code needed for stitch, regions, crops, XMP, and Cast access.
3. Implement prompt modules, static validators, and AI retry/rewrite ladders for one LLM-backed step first.
4. Wire page assets (`stitch`, `regions`, `crops`) and photo assets (`people`, `ocr`, `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, `semantic_review`) into Dagster.
5. Run one page and one photo in dry-run mode under `_debug/photoalbums2/`.
6. Run a 50-100 photo vertical slice. Compare candidate output to canonical `.xmp` on overlapping fields and document deltas.
7. Add the Streamlit review UI and use it to audit the slice.
8. A later proposal handles staging promotion, full-album policy, and legacy retirement.

Rollback: delete the `photoalbums2/` directory and any `_debug/photoalbums2/` or `.xmp.new` outputs. Legacy `.xmp` remains untouched unless an explicit promote command was run.

## Open Questions

- Which LLM client: Pydantic AI or `instructor` + `litellm`?
- Exact implementation for prompt editing: Streamlit source-file editor, draft prompt files, or manual edits plus Dagster metadata inspection.
- Exact implementation for plausible face-match targeting: cosine-distance threshold only, nearest-neighbor search over stored crop embeddings, or a hybrid with a review threshold.

## Resolved During Implementation

- **LLM client**: Used the `openai` SDK directly (OpenAI-compatible for LM Studio). Pydantic AI was listed in `pyproject.toml` as a dependency but the `ai_runner.py` layer wraps the OpenAI client directly, keeping the Pydantic AI dependency available for future structured-output use. This keeps the retry/validation logic fully visible and avoids Pydantic AI version-churn risk.

- **Prompt editing**: Implemented as Streamlit source-file viewer with `st.code` display and a note to edit the file in-editor and commit. No in-app editing — prompt changes remain version-controlled git commits. Dagster metadata records prompt previews per attempt.

- **Face-match targeting**: Used cosine-distance threshold only (`PA2_FACE_MATCH_THRESHOLD`, default 0.5). The `find_matching_photo_keys()` function in `lib/face_match.py` scans crop embeddings from adjacent `.embeddings.json` files and returns partition keys above the threshold. A nearest-neighbor search is left for a follow-on optimization if the album grows large enough to warrant it.

- **Dagster installation**: Dagster is not included in the shared monorepo root venv (it conflicts with the heavy ML deps). Users should either run `uv pip install dagster dagster-webserver` in the root venv, or create a separate venv for `photoalbums2/` as documented in README.md.

- **Partition strategy**: Used `DynamicPartitionsDefinition` for both pages and photos. Two sensors (`discover_pages_sensor`, `discover_photos_sensor`) populate partitions by scanning disk. This avoids hard-coding partition keys while still supporting incremental discovery.
