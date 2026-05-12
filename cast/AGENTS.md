# AGENTS.md

Purpose: operating rules for the `cast/` project.

## Storage layout

Three files live under `cast/data/` by default:

- `people.json` — canonical person identities (`person_id`, `display_name`, aliases, notes)
- `faces.jsonl` — face observations with embedding vectors and source metadata
- `review_queue.jsonl` — pending/decided review items for match approval

Face crops are saved under `cast/data/crops/`.

## Constraints

InsightFace `buffalo_l` is required for ingest and `label-photos`; the module refuses to ingest if it is unavailable rather than falling back to OpenCV Haar cascades.
Matching uses cosine similarity over stored face embeddings.
Auto-suggest an identity only when confidence gates pass (score margin, face quality, sample count).
This module is designed for small local archives (up to a few dozen people).
The web UI warns when stored face rows are legacy detections from an older model path.
