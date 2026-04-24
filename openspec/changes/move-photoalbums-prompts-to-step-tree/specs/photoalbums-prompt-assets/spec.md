## ADDED Requirements

### Requirement: Runtime prompts live in a step-scoped prompt tree
The system SHALL store runtime prompt assets for the existing `photoalbums/` pipeline under `photoalbums/prompts/` in directories that match the owning pipeline step or prompt variant. Production prompt code SHALL NOT read runtime prompt sections from `skills/CORDELL_PHOTO_ALBUMS/SKILL.md`.

#### Scenario: Prompt tree contains ai-index step assets
- **WHEN** the repository is inspected after the migration
- **THEN** `photoalbums/prompts/ai-index/ocr/`, `photoalbums/prompts/ai-index/caption/`, `photoalbums/prompts/ai-index/people-count/`, `photoalbums/prompts/ai-index/locations/`, and `photoalbums/prompts/ai-index/date-estimate/` exist
- **AND** each directory contains the prompt files needed by that step

#### Scenario: Prompt tree contains verify-crops assets
- **WHEN** the repository is inspected after the migration
- **THEN** `photoalbums/prompts/verify-crops/verification/`, `photoalbums/prompts/verify-crops/retry/`, and `photoalbums/prompts/verify-crops/parameter-suggestion/` exist
- **AND** each directory contains the prompt files needed by that prompt variant

#### Scenario: SKILL.md is documentation only
- **WHEN** OCR, caption, people-count, location, date-estimate, or verify-crops prompt text is loaded at runtime
- **THEN** no production code path reads `skills/CORDELL_PHOTO_ALBUMS/SKILL.md`

### Requirement: Prompt loader renders files and reports underlying failures
The system SHALL provide a prompt asset loader that resolves files relative to `photoalbums/prompts/`, reads prompt text, renders runtime variables, loads parameter files, and computes deterministic prompt and parameter hashes. Loader failures SHALL include the underlying OS, TOML, JSON, or template rendering error in the raised error.

#### Scenario: Prompt renders with runtime variables
- **WHEN** a step loads a prompt containing runtime variables such as album title, OCR text, or page image name
- **THEN** the loader returns the rendered prompt text with supplied variables substituted
- **AND** the loader returns provenance containing source path and prompt hash

#### Scenario: Missing prompt file surfaces OS error
- **WHEN** a configured prompt file does not exist
- **THEN** loading fails
- **AND** the error message includes the missing path and the underlying file-not-found error

#### Scenario: Invalid params file surfaces parse error
- **WHEN** a step parameter file contains invalid TOML
- **THEN** loading fails
- **AND** the error message includes the parameter path and the underlying TOML parse error

### Requirement: Step parameters are loaded from adjacent params files
The system SHALL load step-specific tunable model-call parameters from parameter files adjacent to the owning prompt files. Model alias selection SHALL remain in `photoalbums/ai_models.toml`; prompt-tree parameter files SHALL own only step-call settings such as token limits, sampling parameters, image limits, timeouts, and retry ladders.

#### Scenario: Caption params provide default sampler values
- **WHEN** the caption step initializes without CLI or render-settings overrides
- **THEN** caption model-call settings are resolved from `photoalbums/prompts/ai-index/caption/params.toml`
- **AND** the selected caption model still resolves from `photoalbums/ai_models.toml`

#### Scenario: CLI overrides parameter defaults
- **WHEN** the user passes `--caption-max-tokens` or `--caption-temperature`
- **THEN** those explicit CLI values override the prompt-tree caption params for that run
- **AND** the resolved parameter metadata records that a CLI override was used

#### Scenario: Render settings override parameter defaults
- **WHEN** per-archive render settings define caption tuning values
- **THEN** those render settings override the prompt-tree caption params according to the existing settings precedence
- **AND** the resolved parameter metadata records that render settings supplied the override

### Requirement: Prompt debug records prompt and parameter provenance
The system SHALL record prompt asset provenance in prompt debug artifacts for all migrated model-backed steps. Each prompt debug record SHALL include source paths, hashes, rendered prompt text, rendered system prompt when applicable, resolved model-call parameters, and override source metadata.

#### Scenario: Debug artifact records prompt provenance
- **WHEN** a migrated step records prompt debug output
- **THEN** the debug entry includes the prompt file path and prompt hash
- **AND** the entry includes the final rendered prompt text sent to the model

#### Scenario: Debug artifact records parameter provenance
- **WHEN** a migrated step records prompt debug output
- **THEN** the debug entry includes the params file path and params hash when a params file was used
- **AND** the entry includes the resolved model-call parameters sent to the model

#### Scenario: Debug artifact records overrides
- **WHEN** CLI or render settings override prompt-tree defaults
- **THEN** the prompt debug entry identifies the override source and the resolved override values

### Requirement: Prompt and parameter hashes participate in step invalidation
The system SHALL include prompt and parameter hashes in the existing input hash for each migrated step. A changed prompt or parameter file SHALL invalidate only the owning step and downstream dependents declared by the existing step graph.

#### Scenario: Caption prompt edit invalidates caption dependents
- **WHEN** `photoalbums/prompts/ai-index/caption/` prompt text changes
- **THEN** the `caption` step input hash changes
- **AND** downstream `locations`, `date-estimate`, and `propagate-to-crops` rerun according to existing dependency rules
- **AND** independent `ocr`, `people`, and `objects` steps do not rerun solely because of that caption prompt edit

#### Scenario: OCR params edit invalidates OCR dependents
- **WHEN** `photoalbums/prompts/ai-index/ocr/params.toml` changes
- **THEN** the `ocr` step input hash changes
- **AND** downstream caption and date-estimate work reruns according to existing dependency rules

#### Scenario: Verify-crops params edit affects verify-crops only
- **WHEN** a parameter file under `photoalbums/prompts/verify-crops/` changes
- **THEN** verify-crops uses the new parameter hash for its review run
- **AND** ai-index step records are not invalidated solely by that verify-crops parameter edit

### Requirement: Documentation points prompt edits to the prompt tree
The system SHALL update repository guidance so future runtime prompt edits are made in `photoalbums/prompts/` rather than in `skills/CORDELL_PHOTO_ALBUMS/SKILL.md`. The skill file SHALL remain useful for operator workflow documentation.

#### Scenario: AGENTS guidance names prompt tree
- **WHEN** a developer reads repository guidance
- **THEN** the guidance identifies `photoalbums/prompts/` as the runtime prompt source of truth
- **AND** it no longer instructs developers to update `skills/CORDELL_PHOTO_ALBUMS/SKILL.md` for runtime prompt sections

#### Scenario: Skill file no longer claims runtime prompt ownership
- **WHEN** a developer reads `skills/CORDELL_PHOTO_ALBUMS/SKILL.md`
- **THEN** the file describes operator workflow and points runtime prompt editing to `photoalbums/prompts/`
