#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

VIEWER_DIR = Path(__file__).resolve().parent
GALLERY_PATH = VIEWER_DIR / "gallery.json"
CHUNK_SIZE = 1024 * 1024


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _load_gallery_item_paths(gallery_path: Path) -> Dict[str, str]:
    raw = json.loads(gallery_path.read_text(encoding="utf-8"))
    albums = raw.get("albums", [])
    out: Dict[str, str] = {}
    for album in albums:
        if not isinstance(album, dict):
            continue
        for item in album.get("items", []):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            item_path = item.get("path")
            if item_id and isinstance(item_path, str) and item_path.strip():
                out[item_id] = item_path.strip()
    return out


def _resolve_item_path(item_id: str, allow_roots: Iterable[Path]) -> Tuple[Optional[Path], str]:
    try:
        item_map = _load_gallery_item_paths(GALLERY_PATH)
    except FileNotFoundError:
        return None, "gallery.json not found."
    except json.JSONDecodeError as exc:
        return None, f"gallery.json parse error: {exc}"

    raw = item_map.get(item_id)
    if not raw:
        return None, f'Unknown media id "{item_id}" in gallery.json.'

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (GALLERY_PATH.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()

    roots = [Path(root).resolve() for root in allow_roots]
    if roots and not any(_is_within(candidate, root) for root in roots):
        return None, "Path is outside configured allow roots."
    if not candidate.exists():
        return None, f"File does not exist: {candidate}"
    if not candidate.is_file():
        return None, f"Path is not a file: {candidate}"
    return candidate, ""


def _parse_range(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    if not range_header or not range_header.startswith("bytes="):
        return None
    token = range_header.split("=", 1)[1].strip()
    if "," in token:
        return None
    start_s, end_s = token.split("-", 1)
    if start_s == "" and end_s == "":
        return None
    if start_s == "":
        suffix = int(end_s)
        if suffix <= 0:
            return None
        start = max(file_size - suffix, 0)
        end = file_size - 1
        return (start, end)

    start = int(start_s)
    end = file_size - 1 if end_s == "" else int(end_s)
    if start < 0 or end < start:
        return None
    end = min(end, file_size - 1)
    return (start, end)


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: Optional[str] = None, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/media":
            self._serve_media(parsed, send_body=True)
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/media":
            self._serve_media(parsed, send_body=False)
            return
        super().do_HEAD()

    def _serve_media(self, parsed, send_body: bool) -> None:
        params = parse_qs(parsed.query)
        item_id = str((params.get("id") or [""])[0]).strip()
        if not item_id:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing media id.")
            return

        target, err = _resolve_item_path(item_id, getattr(self.server, "allow_roots", []))
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND, err)
            return

        file_size = target.stat().st_size
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        range_header = self.headers.get("Range", "")
        byte_range = None

        if range_header:
            try:
                byte_range = _parse_range(range_header, file_size)
            except ValueError:
                byte_range = None
            if byte_range is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return

        if byte_range is None:
            start, end = 0, file_size - 1
            status = HTTPStatus.OK
        else:
            start, end = byte_range
            status = HTTPStatus.PARTIAL_CONTENT

        length = (end - start) + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        if not send_body:
            return

        with target.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local static server for viewer with local media paths.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8095, type=int, help="Bind port (default: 8095)")
    parser.add_argument(
        "--allow-root",
        action="append",
        default=[],
        help="Optional allowed media root directory; repeat for multiple roots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.allow_roots = [Path(p).expanduser().resolve() for p in args.allow_root]

    roots = ", ".join(str(p) for p in server.allow_roots) or "(none; all absolute paths allowed)"
    print(f"[viewer] serving {VIEWER_DIR}")
    print(f"[viewer] url: http://{args.host}:{args.port}")
    print(f"[viewer] allow roots: {roots}")
    server.serve_forever()


if __name__ == "__main__":
    main()
