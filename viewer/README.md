# viewer

Local-first static viewer for photos and videos, with support for filesystem paths.

Internal project name is `viewer`. Branding is configured through `config.json`.

## Files

- `index.html` - App shell
- `app.js` - Gallery renderer
- `styles.css` - Layout and visual style
- `config.json` - Brand/title/intro text
- `gallery.json` - Albums and media items (local `path` and/or remote `url`)
- `server.py` - Local web server that streams files from filesystem paths
- `scan_media.py` - Scans local folders and generates `gallery.json`
- `_headers` - Security headers for static hosts that support it

## Modes

- `Grid`: card gallery for all media (images and videos).
- `Photobook`: image-only two-page spread view with JS-driven mapped page turn (front/back textures, dynamic fold shading) and left/right arrow navigation.

## gallery.json shape

For local usage, set `path` to an absolute filesystem file path.
For remote usage, you can still set `url` and optional `shareUrl` / `embedUrl`.

```json
{
  "albums": [
    {
      "id": "family-1992",
      "title": "Family 1992",
      "items": [
        {
          "id": "item-1",
          "type": "image",
          "source": "local",
          "title": "Photo title",
          "caption": "Optional caption",
          "path": "C:\\Media\\Photos\\photo001.jpg"
        }
      ]
    }
  ]
}
```

## Local run

From repo root:

```powershell
python viewer/scan_media.py --root "C:\Media\Photos" --root "D:\Archive\Videos"
cd viewer
python server.py --host 0.0.0.0 --port 8095
```

Open `http://127.0.0.1:8095` (or `http://<your-lan-ip>:8095` from another device).

Optional hardening: restrict readable folders with `--allow-root`.

```powershell
python server.py --allow-root "C:\Media\Photos" --allow-root "D:\Archive\Videos"
```

## Scanner options

```powershell
python viewer/scan_media.py `
  --root "C:\Media\Photos" `
  --root "D:\Archive\Videos" `
  --max-items-per-album 500 `
  --output viewer\gallery.json
```

- `--root` can be repeated; each root becomes one album.
- `--no-recursive` scans only top-level files.
- `--max-items-per-album 0` means no limit.

Imago convention mode (what you requested):

```powershell
python viewer/scan_media.py --imago-layout
```

- Videos come from `~/Videos/VHS Clips` and `~/Videos/Videos` (if present).
- Photos come from `*_Pages` folders found under OneDrive Photo Albums roots.
- You can override discovery roots:
  - `--videos-root "<path>"`
  - `--photos-root "<path>"`

## Static deploy (remote URLs only)

### Cloudflare Pages

1. Connect this repo.
2. Build command: none.
3. Output directory: `viewer`.

### Netlify

1. Connect this repo.
2. Build command: none.
3. Publish directory: `viewer`.

### GitHub Pages

Deploy the `viewer/` directory via Actions or publish branch, then serve from that path.

## Notes

- Browser apps cannot directly load arbitrary `file://` paths from an `http://` page. `server.py` bridges this by serving local files over `/media?id=...`.
- Avoid injecting untrusted HTML into metadata values.
- The app theme intentionally matches the VHS tuner visual style.

