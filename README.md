# imago

Family media archive — VHS digitization, photo album preservation, and face recognition.

## Projects

- [`vhs/`](vhs/) — VHS tape digitization pipeline: bad-frame detection, gamma correction, subtitle generation, render wizard
- [`photoalbums/`](photoalbums/) — Scanned photo album pipeline: TIFF processing, metadata embedding, page stitching

## Structure

```
imago/
  vhs/           ← VHS pipeline (python vhs.py ...)
  photoalbums/   ← Photo album pipeline
  cast/          ← Shared face recognition (coming soon)
```
