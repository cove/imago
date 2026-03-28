#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import threading
from email.utils import formatdate, parsedate_to_datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

VIEWER_DIR = Path(__file__).resolve().parent
GALLERY_PATH = VIEWER_DIR / "gallery.json"
CHUNK_SIZE = 1024 * 1024
MEDIA_CACHE_CONTROL = "public, max-age=86400, stale-while-revalidate=604800"
CLIENT_DISCONNECT_ERRNOS = {
    errno.EPIPE,  # Broken pipe
    errno.ECONNRESET,  # Connection reset by peer
    errno.ECONNABORTED,  # Software caused connection abort
    10053,  # WinError WSAECONNABORTED
    10054,  # WinError WSAECONNRESET
}

_GALLERY_CACHE_LOCK = threading.Lock()
_GALLERY_CACHE_MTIME_NS: Optional[int] = None
_GALLERY_CACHE_ITEMS: Dict[str, str] = {}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _load_gallery_item_paths(gallery_path: Path) -> Dict[str, str]:
    global _GALLERY_CACHE_MTIME_NS, _GALLERY_CACHE_ITEMS
    stat = gallery_path.stat()
    mtime_ns = stat.st_mtime_ns

    with _GALLERY_CACHE_LOCK:
        if _GALLERY_CACHE_MTIME_NS == mtime_ns and _GALLERY_CACHE_ITEMS:
            return dict(_GALLERY_CACHE_ITEMS)

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

    with _GALLERY_CACHE_LOCK:
        _GALLERY_CACHE_MTIME_NS = mtime_ns
        _GALLERY_CACHE_ITEMS = out

    return out


def _make_etag(file_size: int, mtime_ns: int) -> str:
    return f'W/"{file_size:x}-{mtime_ns:x}"'


def _etag_matches(header_value: str, etag: str) -> bool:
    tokens = [token.strip() for token in header_value.split(",") if token.strip()]
    if "*" in tokens:
        return True
    strong = etag[2:] if etag.startswith("W/") else etag
    return etag in tokens or strong in tokens


def _parse_http_date(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    return dt.timestamp()


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

        stat = target.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime
        mtime_ns = stat.st_mtime_ns
        etag = _make_etag(file_size, mtime_ns)
        last_modified = formatdate(mtime, usegmt=True)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

        if_none_match = self.headers.get("If-None-Match", "")
        if if_none_match and _etag_matches(if_none_match, etag):
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.send_header("Cache-Control", MEDIA_CACHE_CONTROL)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return

        if_modified_since = self.headers.get("If-Modified-Since", "")
        if if_modified_since and not if_none_match:
            since_ts = _parse_http_date(if_modified_since)
            if since_ts is not None and int(mtime) <= int(since_ts):
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", last_modified)
                self.send_header("Cache-Control", MEDIA_CACHE_CONTROL)
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return

        range_header = self.headers.get("Range", "")
        if_range = self.headers.get("If-Range", "").strip()
        if range_header and if_range:
            if if_range.startswith(("W/", '"')):
                if not _etag_matches(if_range, etag):
                    range_header = ""
            else:
                if_range_ts = _parse_http_date(if_range)
                if if_range_ts is None or int(mtime) > int(if_range_ts):
                    range_header = ""

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
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", last_modified)
                self.send_header("Cache-Control", MEDIA_CACHE_CONTROL)
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
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", last_modified)
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Cache-Control", MEDIA_CACHE_CONTROL)
        self.end_headers()

        if not send_body:
            return

        try:
            with target.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Client canceled navigation/seek while media was streaming.
            return
        except OSError as exc:
            if exc.errno in CLIENT_DISCONNECT_ERRNOS:
                # Expected transient disconnect from browser or remote client.
                return
            raise


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
