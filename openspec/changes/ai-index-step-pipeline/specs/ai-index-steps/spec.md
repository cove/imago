## ADDED Requirements

### Requirement: ai-index step graph is declared with explicit dependency edges
The system SHALL define the following named steps for the `ai-index` stage, each with declared upstream dependencies:

| Step | Depends on |
|---|---|
| `ocr` | *(none)* |
| `people` | *(none)* |
| `caption` | `ocr`, `people` |
| `locations` | `caption` — AI uses caption text + image to produce a primary GPS query and named location queries, all resolved via Nominatim |
| `objects` | *(none)* |
| `date-estimate` | `ocr`, `caption` |
| `propagate-to-crops` | `locations`, `people` — pushes GPS and person names into each crop XMP sidecar under `_Photos/` |

Steps with no declared dependencies MAY run in any order. Steps with dependencies SHALL NOT run until all declared upstream steps have completed for the current image.

#### Scenario: Step graph evaluated for a single image
- **WHEN** `ai-index` processes an image
- **THEN** steps are dispatched in an order consistent with the declared dependency edges, and no step begins before its declared upstream steps have produced output

#### Scenario: Caption step blocked until ocr and people complete
- **WHEN** `ai-index` begins processing and `ocr` and `people` have not yet run
- **THEN** `caption` is not dispatched until both `ocr` and `people` have produced output for the current image

---

### Requirement: Each step declares an input hash function covering only its relevant settings
The system SHALL compute a per-step input hash that covers only the settings and upstream outputs that affect that step's result. The hash function SHALL be deterministic and produce a fixed-length hex string.

Input hash coverage by step:

| Step | Hash inputs |
|---|---|
| `ocr` | ocr engine, ocr model, ocr language, scan group signature (for multi-scan pages) |
| `people` | cast store signature |
| `caption` | caption engine, caption model, people output hash |
| `locations` | caption engine, caption model, caption output hash, nominatim settings, locations prompt version |
| `objects` | object detection model |
| `date-estimate` | date-estimate model, ocr output hash, caption output hash |
| `propagate-to-crops` | locations output hash, people output hash, sorted list of crop paths for this page |

#### Scenario: Caption model changes, OCR step is not invalidated
- **WHEN** the caption model is updated in settings
- **THEN** the `ocr` step's input hash is unchanged and `ocr` is not rerun if its prior result is recorded in XMP

#### Scenario: Cast store changes, people and downstream steps invalidated
- **WHEN** the cast store signature changes
- **THEN** the `people` step's input hash changes, `people` reruns, and `caption`, `locations`, and `date-estimate` are subsequently marked stale because an upstream step reran

---

### Requirement: locations step writes to the existing "location" and "locations_shown" fields in imago:Detections
The `locations` step SHALL write its output into the same `imago:Detections` fields used by the previous `gps` and `locations-shown` steps, preserving backward compatibility for any reader of those fields:

- `imago:Detections["location"]` — dict containing the primary GPS result (keys: `gps_latitude`, `gps_longitude`, `map_datum`, `query`, `display_name`, `source`, and optional `city`, `state`, `country`)
- `imago:Detections["locations_shown"]` — list of dicts, one per named location (keys: `name`, `world_region`, `country_name`, `country_code`, `province_or_state`, `city`, `sublocation`, and optional `gps_latitude`, `gps_longitude`, `gps_source`)
- `imago:Detections["location_shown_ran"]` — boolean, `true` if the locations step ran regardless of whether any locations were found

#### Scenario: AI identifies a primary location and two named locations
- **WHEN** the `locations` step completes with one primary query and two named queries resolved by Nominatim
- **THEN** `imago:Detections["location"]` contains the primary GPS dict and `imago:Detections["locations_shown"]` contains two entries with resolved coordinates

#### Scenario: AI finds named locations but no primary GPS
- **WHEN** the AI returns no primary location query but does return named location queries
- **THEN** `imago:Detections["location"]` is absent or empty and `imago:Detections["locations_shown"]` contains the resolved named locations

---

### Requirement: locations step prompt informs the AI of Nominatim free-form query format
The prompt used by the `locations` step SHALL explain to the model that Nominatim accepts free-form natural-language place name queries in any language (e.g. `"Eiffel Tower, Paris, France"`, `"Cafe Paris, New York"`). The prompt SHALL make clear that Nominatim resolves place names to coordinates — the model MUST NOT return raw lat/lon values as the query.

#### Scenario: AI identifies a well-known landmark
- **WHEN** the caption and image clearly depict a specific named landmark
- **THEN** the `locations` step produces a free-form query such as `"Eiffel Tower, Paris, France"` and passes it to Nominatim

#### Scenario: AI identifies a city and country with high confidence
- **WHEN** the caption and image indicate a specific city with no ambiguity
- **THEN** the `locations` step MAY produce a structured query such as `city=London&country=United Kingdom` for higher Nominatim precision

#### Scenario: AI cannot determine a meaningful location
- **WHEN** the caption and image provide insufficient location context
- **THEN** the `locations` step returns no primary GPS and no named locations; no Nominatim call is made

### Requirement: A step is stale when its recorded input hash differs or an upstream step reran
The system SHALL evaluate each step's staleness before dispatching it:

1. If the step has no recorded XMP entry → stale (not yet run)
2. If `recorded_input_hash != current_input_hash` → stale (settings drifted); an empty `""` hash never matches any computed hash, so `"not-applicable"` records are always stale until the engine is configured
3. If any declared upstream step reran during this session → stale (upstream output changed)

A step that is not stale SHALL be skipped; its previous output SHALL be loaded from the `imago:Detections` XMP payload and used as-is.

#### Scenario: No prior XMP step record
- **WHEN** an image has no `ai-index/ocr` record in `imago:Detections["pipeline"]`
- **THEN** the `ocr` step is treated as stale and runs unconditionally

#### Scenario: Input hash matches, no upstream rerun
- **WHEN** the recorded `input_hash` for `caption` matches the current hash and neither `ocr` nor `people` reran this session
- **THEN** `caption` is skipped and the previous caption output is reused from `imago:Detections`

#### Scenario: Upstream step reran, downstream forced stale
- **WHEN** the `people` step reruns (cast store changed)
- **THEN** `caption`, `locations`, `date-estimate`, and `propagate-to-crops` are treated as stale regardless of their own recorded hashes

---

### Requirement: propagate-to-crops step writes locations and people metadata to each crop XMP sidecar
The `propagate-to-crops` step SHALL run automatically as the final step of every page's ai-index pass. It SHALL:

1. Discover the page's crop files by reading the MWG-RS region list from the page XMP
2. For each crop, write the page's resolved `location` GPS fields to the crop XMP sidecar
3. For each crop, write the person names associated with that region to the crop's `Iptc4xmpExt:PersonInImage` field
4. Record a step entry `ai-index/propagate-to-crops` in each crop's own `imago:Detections["pipeline"]`

The step SHALL be skipped (not stale) when neither `locations` nor `people` reran and the crop path set is unchanged. Pages with no crop files SHALL record the step with `result: "ok"` and zero crops updated.

#### Scenario: Cast store changes, crops get updated person names
- **WHEN** the `people` step reruns due to a cast store change and the page has two crops
- **THEN** `propagate-to-crops` reruns, writes the updated person names to both crop XMP sidecars, and records `ai-index/propagate-to-crops` in each crop's pipeline record

#### Scenario: Locations step reruns, crops get updated GPS
- **WHEN** the `locations` step reruns (Nominatim config changed) and the page has three crops
- **THEN** `propagate-to-crops` reruns and writes the updated GPS coordinates to all three crop XMP sidecars

#### Scenario: Page has no crops
- **WHEN** `propagate-to-crops` runs on a page with no MWG-RS regions or no corresponding crop files
- **THEN** the step completes with `result: "ok"` and no crop sidecars are written

#### Scenario: Neither upstream step reran
- **WHEN** `locations` and `people` both skipped and the crop path set is unchanged
- **THEN** `propagate-to-crops` is skipped and no crop XMP sidecars are touched

---

### Requirement: Step records are written to XMP only after the final payload write succeeds
The system SHALL write all per-step XMP records atomically with the final `imago:Detections` payload. Step records SHALL NOT be persisted if the final XMP write fails or the process exits abnormally.

#### Scenario: Process exits after step runs but before XMP write
- **WHEN** `ocr` runs successfully but the process crashes before the final XMP write
- **THEN** on the next run, `ocr` has no recorded step entry and reruns

#### Scenario: All steps complete, XMP write succeeds
- **WHEN** all steps complete and the XMP write succeeds
- **THEN** each step's `imago:Detections["pipeline"]` record is updated with the current `input_hash`, timestamp, and result

---

### Requirement: CLI supports targeting individual steps for forced re-run
The system SHALL accept a `--steps <name>[,<name>]` argument on the `ai-index` CLI command. When specified, the listed steps SHALL be treated as unconditionally stale regardless of their recorded hashes. Steps not listed are still subject to normal staleness evaluation (including being forced stale by upstream reruns).

#### Scenario: User forces caption re-run
- **WHEN** `ai-index --steps caption` is invoked
- **THEN** the `caption` step runs unconditionally; `ocr` and `people` are skipped if their hashes match; `locations`, `date-estimate`, and `propagate-to-crops` are forced stale because `caption` reran

#### Scenario: User forces ocr re-run
- **WHEN** `ai-index --steps ocr` is invoked
- **THEN** `ocr` runs unconditionally; `caption`, `locations`, `date-estimate`, and `propagate-to-crops` are forced stale as downstream dependents
