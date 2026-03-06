# cast

Shared face identity module for linking people across photo albums and VHS footage.

## Storage format

This module is text-file based and keeps three files in `cast/data/` by default:

- `people.json`: canonical person identities (`person_id`, `display_name`, aliases, notes)
- `faces.jsonl`: face observations with embedding vectors and source metadata
- `review_queue.jsonl`: pending/decided review items for match approval

## Run web UI

```bash
python cast.py init
python cast.py web
```

Open `http://127.0.0.1:8093`.

## What the UI can do now

- Ingest a photo path and auto-detect faces
- Ingest a VHS/video path and sample frames for face detection
- Bulk scan photo album view folders (for example `Photo Albums/*_View/*.jpg`)
- Save cropped face images under `cast/data/crops/`
- Auto-enqueue each detected face into a "who is this?" review queue
- Let you assign an existing person or type a new name to create+assign immediately
- Only auto-suggest an identity when confidence gates pass (score margin, face quality, sample count)
- Auto-prune obvious non-face detections from the pending queue
- Reset pending queue + unknown faces to start fresh
- Optional "Skip Statues/Paintings" ingest mode (enabled by default in UI)
- Click a face crop to open the full source image with face box highlight

## Notes

- Primary detection uses OpenCV YuNet (auto-downloads model on first run), with cascade fallback.
- Matching uses cosine similarity over lightweight grayscale embeddings.
- This scaffold is designed for small local archives (for example up to a few dozen people).
