## Why

The current `photoalbums/` pipeline has grown to ~21,000 LOC across ~50 modules and keeps producing quality failures that are expensive to observe and slow to recover from. Concretely:

- **Dependency tracking is homegrown.** `ai_index_steps.py` and `pipeline.py` each define overlapping DAGs with hand-rolled input-hash staleness checks. "When do faces need reprocessing?" is answered by ad-hoc reprocess-audit code rather than asset lineage. At 7,000 photos per album, a full rerun is ~6 days of wall-clock, so getting invalidation right is the load-bearing product concern.
- **The retry ladder tunes the wrong knobs.** `ai_verify_crops.py` retries with `max_tokens`, `temperature`, and image edge. It never touches `frequency_penalty`, `repetition_penalty`, `top_p`, or `presence_penalty` — the sampler knobs that actually fix per-photo-random decoding failures (`\\\\` spam, stray-quote hallucination, repetition loops). The existing pass-3 logic asks the vision model itself to suggest parameters, which is inverted: the model knows the image, not the sampler.
- **Verification runs after the write.** Bad XMP is written, then re-inspected. Mechanical validation should gate the write in the first place, while visual/semantic mistakes should be handled by an AI review/rewrite loop rather than brittle Python string cleanup.
- **Prompts are scattered.** SKILL.md sections are loaded via `_prompt_skill.required_section_text()` and then re-assembled in `_caption_prompts.py`, `ai_ocr.py`, `ai_verify_crops.py`, `ai_date.py`, each with its own inline system prompts. To know the final prompt for a given step you have to mentally execute Python.
- **Human oversight ergonomics are poor.** To audit one photo you read terminal logs, open an XMP in a text editor, and grep job artifacts. There is no per-photo view of "image + prompt sent + model response + validation history".

Rather than refactor in place, this change stands up a new `photoalbums2/` project in the monorepo that addresses all five problems with off-the-shelf frameworks: **Dagster** for partitioned-asset DAGs and UI, **Pydantic AI** for structured LLM calls with validator-driven retries, **Streamlit** for a per-photo review UI. The existing `photoalbums/` pipeline is untouched; both run in parallel until a later cutover proposal.

## What Changes

- Create a new top-level `photoalbums2/` project with its own `pyproject.toml`, independent of `photoalbums/`.
- Define each pipeline step as a Dagster asset with an explicit work unit: `stitch`, `regions`, and `crops` run per album page; `people`, `ocr`, `caption`, `location_queries`, `gps_location`, `locations_shown`, `date`, and `semantic_review` run per cropped photo.
- Track Cast face-recognition changes so recognizing a new face triggers face matching only for cropped photos whose embeddings are plausible matches for that newly recognized face. It MUST NOT re-run stitch, regions, crops, OCR, or unrelated photos.
- Replace the custom LM Studio wrapper layer with Pydantic AI agents or an equivalent structured-output client, one per LLM-backed step. Structured outputs are Pydantic models; retry happens via a step-owned AI retry/rewrite ladder tuned to known failure modes.
- Consolidate every step's prompt into a single file under `photoalbums2/prompts/`, containing system prompt, user-template (Jinja), and output schema. SKILL.md stays as documentation only; no prompt assembly in multiple files.
- Move the old "verify" concept into two smaller paths: conservative static validation at write time, plus an AI-backed semantic review/rewrite step for mistakes that require visual judgment rather than brittle string fixes.
- Ship a minimal Streamlit review UI that reads from Dagster's event log and lets a human browse one photo at a time with its image, prompts sent, responses received, validation history, and last-run status.
- Read existing XMP sidecars written by `photoalbums/` as a starting materialization so the new pipeline does not need to re-process work already done.
- Support output modes for the parallel period: dry-run writes candidate XMP and diagnostics under `_debug/photoalbums2/`; staging writes adjacent `.xmp.new`; promotion to canonical `.xmp` is explicit and never automatic.
- Scope this proposal to one album end-to-end as a vertical slice. Full cutover across all albums, removal of `photoalbums/`, and archive of existing OpenSpec captions capabilities are out of scope here.

## Capabilities

### New Capabilities
- `photoalbums2-pipeline`: the Dagster-based asset graph, Pydantic-AI step agents, validator-driven retry ladder, prompt file layout, and Streamlit review UI for the new `photoalbums2/` project.

### Modified Capabilities
- None. `photoalbums/` and its capabilities (`view-xmp-regions`, `unified-process-pipeline`, etc.) are unchanged by this proposal.

## Impact

- **Affected code**: new `photoalbums2/` tree only. Read-only imports from `photoalbums.lib.xmp_sidecar` (mature XMP I/O) and `cast.storage` (face store). No changes to existing modules.
- **Affected metadata**: none on the canonical `.xmp` sidecars during the parallel period. Dry-run output lands under `_debug/photoalbums2/`; staging output lands in `.xmp.new` (name configurable) per photo.
- **Runtime impact**: none on the existing pipeline. The new pipeline runs on demand (`dagster dev` + manual materialize) and does not share a process with `photoalbums/`.
- **Inference impact**: LM Studio stays as the default provider. Provider abstraction allows later swap to MLX-VLM or Ollama without touching step agents.
- **Scope caveat**: at 7,000 photos per album, a full clean run is expensive. The vertical slice is expected to run over a small subset (e.g. 50–100 photos from one album, or one book) to validate the architecture. A full-album run happens only after the slice is green.
