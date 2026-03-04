# imago

Family media archive for VHS digitization, photo album preservation, and shared face IDs.

## Projects

- [`vhs/`](vhs/) - VHS tape digitization pipeline: bad-frame detection, gamma correction, subtitle generation, render wizard
- [`photoalbums/`](photoalbums/) - Scanned photo album pipeline: TIFF processing, metadata embedding, page stitching
- [`cast/`](cast/) - Shared face identity store + local web review UI (text files)

## Cast quick start

1. Initialize files:
   - `python cast.py init`
2. Run local web UI:
   - `python cast.py web`
3. Open:
   - `http://127.0.0.1:8093`

From the Cast web UI you can:

- ingest a photo path and detect/crop faces
- bulk scan `Photo Albums/*_View` for `.jpg` photos and detect/crop faces
- ingest a VHS/video path (sampled frames) and detect/crop faces
- review queued faces and label who each person is (one-by-one naming prompt included)
- auto-prune obvious non-face detections from the pending queue
- optional "Skip Statues/Paintings" mode for stricter filtering
- click a face crop to view the full source image (with face highlight box)

Default Cast storage files:

- `cast/data/people.json`
- `cast/data/faces.jsonl`
- `cast/data/review_queue.jsonl`

## Structure

```
imago/
  vhs/           <- VHS pipeline (python vhs.py ...)
  photoalbums/   <- Photo album pipeline
  cast/          <- Shared face identity module + web UI
  cast.py        <- Cast CLI entrypoint
```
