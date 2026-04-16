## Context

The page-region stage is the point where the pipeline decides what becomes an individual photo crop and what capation goes with which photo. That stage should use Docling for page layout and caption extraction, but it must run from local model assets instead of reaching out to Hugging Face Hub or LM Studio during detection.

The follow-up change keeps the Docling-based detection branch, but makes the branch explicitly local/offline. Downstream text in photos OCR remains a separate concern and is not part of the region-finding and captioning contract.

The crop step must consume the region list stored in the page view XMP sidecar. That keeps region detection and crop generation decoupled while still making the stored `mwg-rs:RegionList` the source of truth for crop boundaries, and the capation is stored in the `mwg-rs:Name`, since `mwg-rs:Description` is technically constrained to focus-utility categorization.

## Goals / Non-Goals

**Goals:**
- Run Docling photo-region detection directly through the Python library.
- Require the Docling branch to run from local model assets without runtime network fetches other than inital download of the models. 
- Remove the existing non-Docling region-detection path and caption OCR and assocation with a photo.
- Keep the current XMP writeback, validation, and pipeline-step contract intact.
- Keep in photo-text OCR/text recognition separate from page-region detection.
- Keep the capations linked to the photos they belong to, and propigate them to the View and cropped photos.

**Non-Goals:**
- Changing the in photo-text extraction from using LM Studio and populating the scene text field in the XMP file.
- Creating a format to save the crop and capation in that's different than standard XMP.
- Changing logic outside of the photoalbums project.
- Changing the face and object identifcation, this will be handled the same way as before.

## Decisions

1. Use Docling's preset-driven library path instead of an LM Studio API wrapper.
   - Rationale: the preset already defines the prompt and response format; the extra LM Studio transport layer only adds failure modes and is the wrong place for page layout.
   - Alternatives considered: keep `ApiVlmEngineOptions`, or replace the pipeline with a custom parser. Both were rejected because they preserve the external dependency or reintroduce brittle parsing.

2. Keep `DocumentConverter` and `VlmPipeline` in the implementation, but require local model assets.
   - Rationale: those are the supported library entry points and already produce structured `DoclingDocument` output that maps cleanly to `RegionResult`.
   - Alternatives considered: reimplement picture extraction from raw doctags, or introduce a separate model server abstraction. Both add complexity without solving the stated problem.

3. Keep the `docling` model selector as a configuration gate, but make the selected preset resolve from local assets only.
   - Rationale: the runtime should stay configuration-driven, but region detection must not depend on network resolution of model weights.
   - Alternatives considered: add a second CLI flag or a separate model kind enum. Rejected to avoid UI churn for a narrow backend change.

4. Preserve validation and pipeline-step outcomes.
   - Rationale: the existing `no_regions` / `validation_failed` / `regions_found` contract prevents bad crops and infinite reruns.
   - Alternatives considered: accept raw library output without validation. Rejected because it would trade one brittle backend for another.

5. Leave photo-text OCR as a downstream concern, not a region-detection concern.
   - Rationale: LM Studio is a better fit for semantic text recognition on already-cropped photos than for finding page boundaries.
   - Alternatives considered: keep OCR mixed into page-region detection. Rejected because it couples unrelated behaviors.

6. Keep the crop step reading boundaries from XMP instead of from Docling directly.
   - Rationale: the page sidecar is the durable contract between detection and cropping, so the cropper should consume the stored region list rather than any in-memory detector output.
   - Alternatives considered: pass detector boxes directly from region detection into crop generation. Rejected because it would weaken the sidecar as the source of truth.

## Risks / Trade-offs

- [Risk] The local Docling engine may require more memory or a different hardware profile than the current path. -> Keep the preset-driven configuration narrow and document the expected local engine in the task rollout.
- [Risk] The local model assets may be missing on some machines. -> Fail fast with the underlying model-resolution error and document the requirement to preload or cache the Docling weights locally.
- [Risk] The local engine may still emit oversized regions on some pages. -> Keep the current validation step and add regression tests against representative pages; add geometry post-processing only if the defect remains after the backend swap.

## Migration Plan

1. Update the Docling-specific config so the branch no longer depends on a loaded LM Studio model or base URL.
2. Update `_docling_pipeline.py` to call Docling directly with the preset-driven local engine path and local model assets.
3. Update tests to assert the local Docling path and remove LM Studio-specific expectations.
4. Run the existing validation and docling tests, then fix any geometry regressions before considering the change complete.

Rollback:
- Restore the previous LM Studio-backed Docling branch if the local engine is not stable enough in practice.
- Because the non-Docling path is untouched, rollback can stay scoped to the Docling branch and its config.

## Open Questions

- Which local Docling engine should be considered canonical here: MLX, transformers, or another preset-backed engine? The implementation should use whichever preset can run from preloaded local assets without a runtime fetch.
