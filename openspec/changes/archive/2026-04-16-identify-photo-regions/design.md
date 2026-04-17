## Context

The page-region stage is the point where the pipeline decides what becomes an individual photo crop and what caption goes with which photo. That stage should use Docling for page layout and caption extraction, but it must run from local model assets instead of reaching out to Hugging Face Hub or LM Studio during detection.

The follow-up change keeps the Docling-based detection branch, but makes the branch explicitly local/offline. It also separates page captions from text that appears inside a photo: Docling is responsible for page layout and caption extraction, while downstream OCR handles scene text or handwriting that is part of the photo content.

The crop step must consume the region list stored in the page view XMP sidecar. That keeps region detection and crop generation decoupled while still making the stored `mwg-rs:RegionList` the source of truth for crop boundaries. Per-region photo captions are stored in `mwg-rs:Name`, while the page view's top-level `dc:description` remains the searchable OCR-and-scene-text summary for the page itself rather than a second per-region caption store.

## Goals / Non-Goals

**Goals:**
- Run Docling photo-region detection directly through the Python library.
- Remove the Gemma/LM Studio region-detection path for page photo boundaries.
- Require the Docling branch to run from local model assets on Windows without runtime network fetches other than initial model download.
- Use the backend's automatic GPU/CPU selection on Windows rather than introducing a manual hardware switch.
- Preserve raw Docling output in a separate debug JSON artifact when region-debug output is enabled.
- Retry Docling only for runtime failures, not for validation failures.
- Keep the current XMP writeback, validation, and pipeline-step contract intact.
- Keep in-photo OCR/text recognition, including handwriting inside photos, separate from page-region detection.
- Keep the captions linked to the photos they belong to through `mwg-rs:Name`, and keep the page-level top-level `dc:description` focused on searchable OCR and scene text.

**Non-Goals:**
- Changing in-photo text extraction beyond what is needed to keep page OCR and scene text searchable at the page level.
- Creating a format to save the crop and caption in that's different than standard XMP.
- Embedding raw Docling JSON in XMP as a second persistence format for the same regions and captions.
- Changing logic outside of the photoalbums project.
- Changing face and object identification; this is handled the same way as before.

## Decisions

1. Remove the Gemma/LM Studio region-detection backend and use Docling's preset-driven library path as the only region detector.
   - Rationale: the Gemma/LM Studio path did not produce reliable photo boundaries, while Docling already performs the page-layout job this stage needs.
   - Alternatives considered: keep Gemma as a fallback, or route both backends through a shared abstraction. Both were rejected because they preserve the unreliable path and complicate a workflow that now has one intended detector.

2. Keep `DocumentConverter` and `VlmPipeline` in the implementation, but require local model assets on Windows with automatic GPU/CPU selection.
   - Rationale: those are the supported library entry points and already produce structured `DoclingDocument` output that maps cleanly to `RegionResult`, and Windows with auto hardware selection is the deployment target we need now.
   - Alternatives considered: reimplement picture extraction from raw doctags, introduce a separate model server abstraction, or add an explicit backend selector now. All add complexity without solving the stated problem.

3. Keep the `docling` model selector as a configuration gate, but require the selected preset to run from local assets only.
   - Rationale: the runtime should stay configuration-driven, but region detection must run the selected Docling model on the local machine and must not call a remote or cloud-hosted model service.
   - Alternatives considered: add a second CLI flag or a separate model kind enum. Rejected to avoid UI churn for a narrow backend change.

4. Preserve validation as a separate local gate, but add a distinct runtime `failed` result and retry only runtime failures.
   - Rationale: validation failures and backend failures are different problems. Validation failures should remain visible as `validation_failed`, while transient or startup/runtime errors should get bounded retries before settling as `failed`.
   - Alternatives considered: retry validation failures too, or collapse all failures into `validation_failed`. Both were rejected because they blur two different failure modes and make diagnosis harder.

5. Leave photo-text OCR as a downstream concern, not a region-detection concern.
   - Rationale: page captions and text inside a photo are different extraction problems. Docling should be used for page layout and caption association, while downstream OCR remains responsible for scene text and handwriting inside the cropped photo.
   - Alternatives considered: keep OCR mixed into page-region detection. Rejected because it couples unrelated behaviors.

6. Keep the crop step reading boundaries from XMP instead of from Docling directly, and store each region caption in `mwg-rs:Name`.
   - Rationale: the page sidecar is the durable contract between detection and cropping, so the cropper should consume the stored region list rather than any in-memory detector output. `mwg-rs:Name` is the per-region caption/title slot that belongs with the photo, while top-level `dc:description` stays page-oriented and searchable.
   - Alternatives considered: pass detector boxes directly from region detection into crop generation, or continue storing per-region captions in region `dc:description`. Both were rejected because they weaken the sidecar contract or blur page-level and region-level text semantics.

7. Do not store raw Docling JSON in XMP; write it as a separate debug artifact instead.
   - Rationale: the standard XMP region and description fields are the interoperability contract for other tools. Embedding Docling-specific JSON would duplicate the same data, tie the sidecar to an implementation-specific format, and make the XMP larger without improving the canonical crop contract. A separate debug artifact preserves the raw Docling structure for troubleshooting without changing the XMP contract.
   - Alternatives considered: serialize the full Docling response into a custom XMP field for future discovery. Rejected because future programs are better served by the standard XMP fields already being written; raw Docling output belongs in a separate debug artifact rather than inside the XMP contract.

## Risks / Trade-offs

- [Risk] The local Docling engine may require more memory or a different hardware profile than the current path. -> Keep the preset-driven configuration narrow and document the expected local engine in the task rollout.
- [Risk] The local model assets may be missing on some machines. -> Fail fast with the underlying model-resolution error and document the requirement to preload or cache the Docling weights locally.
- [Risk] The local engine may still emit oversized regions on some pages. -> Keep the current validation step and add regression tests against representative pages; validation failures are recorded without retry for now.
- [Risk] Runtime failures may be transient or backend-specific. -> Retry only runtime failures a bounded number of times, then record `failed` with the underlying error preserved in logs/debug output.

## Migration Plan

1. Update the Docling-specific config so the branch no longer depends on a loaded LM Studio model or base URL.
2. Update `_docling_pipeline.py` to call Docling directly with the preset-driven local engine path, Windows auto hardware selection, and local model assets.
3. Update XMP region read/write and crop-caption handling so per-region captions live in `mwg-rs:Name` instead of region `dc:description`.
4. Add Docling runtime retry handling plus a terminal `failed` pipeline result for exhausted runtime failures.
5. Add a Docling debug artifact path that writes raw Docling JSON outside XMP when region-debug output is enabled.
6. Update tests to assert the Docling-only path, the new caption-field contract, the runtime retry/failure behavior, and the Docling debug artifact behavior.
7. Run the existing validation and Docling tests, then fix any geometry regressions before considering the change complete.

Rollback:
- Restore the previous region-detection implementation only if the Docling-only path proves unworkable in practice.
- Because this change removes the Gemma/LM Studio detector rather than keeping it in parallel, rollback is an explicit code rollback rather than a runtime fallback.
