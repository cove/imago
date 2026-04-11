## MODIFIED Requirements

### Requirement: render-pipeline includes crop-regions step between detect-regions and face-refresh
The pipeline SHALL run `crop-regions` as an inline per-page step after `detect-regions` and before `face-refresh`. A `--skip-crops` flag SHALL suppress this step without affecting any other step.

#### Scenario: Pipeline runs crop-regions by default
- **WHEN** `photoalbums render-pipeline --album-id Egypt_1975 --photos-root <root>` is run
- **THEN** each page is processed in order: render -> detect-regions -> crop-regions -> face-refresh -> ctm-apply

#### Scenario: --skip-crops suppresses crop step
- **WHEN** `photoalbums render-pipeline --skip-crops --album-id Egypt_1975 --photos-root <root>` is run
- **THEN** the crop-regions step is skipped for all pages; all other steps run normally

#### Scenario: face-refresh runs on crop JPEGs as well as page view JPEGs
- **WHEN** face-refresh runs for a page that has crops in `_Photos/`
- **THEN** `buffalo_l` face identification is run against each crop JPEG and face regions are written to the crop sidecars before `ctm-apply` runs

### Requirement: Standalone crop-regions CLI command
The system SHALL provide a `crop-regions` subcommand that runs only the crop step for a matching album or page, using the same pipeline-state check and `--force` / `--skip-crops` semantics as when run inside `render-pipeline`.

#### Scenario: Standalone crop-regions run
- **WHEN** `photoalbums crop-regions --album-id Egypt_1975 --photos-root <root>` is run
- **THEN** only the crop step runs; pipeline state is checked and updated exactly as when run inside `render-pipeline`
