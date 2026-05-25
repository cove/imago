"""Sync named face data from Immich to local XMP sidecars and the cast store.

Usage (via CLI):
    python -m cast immich-sync --immich-url http://immich.local:2283 --api-key <key> \\
        --photos-root /path/to/photo-albums

Environment variables (override by CLI flags):
    IMMICH_URL      Base URL of the Immich instance (no trailing slash)
    IMMICH_API_KEY  API key from Immich → Account Settings → API Keys

Strategy
--------
1. Index local photo files by lowercase filename stem.
2. Fetch all named people from Immich (paginated).
3. Optionally upsert those names into the cast TextFaceStore (people.json).
4. For each named person, fetch their assets from Immich; match to local files
   by originalFileName stem.
5. For each matched asset, fetch all face detections (named + unnamed) and
   collect the named ones.
6. Write PersonInImage and IPTC ImageRegion bounding boxes to the .xmp sidecar.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .storage import TextFaceStore
from .xmp_writer import merge_face_regions_xmp, merge_persons_xmp

log = logging.getLogger(__name__)

IMMICH_URL_ENV = "IMMICH_URL"
IMMICH_API_KEY_ENV = "IMMICH_API_KEY"

_DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
_PEOPLE_PAGE_SIZE = 500
_ASSET_PAGE_SIZE = 1000
_IMMICH_RETRY_DELAY_SECONDS = 5


# ---------------------------------------------------------------------------
# Immich API helpers (stdlib urllib — no extra deps)
# ---------------------------------------------------------------------------


def _http_request(
    method: str,
    url: str,
    api_key: str,
    *,
    params: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = json.dumps(payload).encode() if payload is not None else None
    while True:
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "x-api-key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            message = details or f"HTTP {exc.code}: {exc.reason}"
            log.exception("Immich HTTP %d %s: %s", exc.code, url, exc.reason)
            if 400 <= exc.code < 500:
                raise
            print(f"Immich request failed: {message}", flush=True)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            reason = getattr(exc, "reason", str(exc))
            print(f"Immich request failed: {reason}", flush=True)
        time.sleep(_IMMICH_RETRY_DELAY_SECONDS)


def _http_get(url: str, api_key: str, params: dict[str, str] | None = None) -> Any:
    return _http_request("GET", url, api_key, params=params)


def _http_post(url: str, api_key: str, payload: dict[str, Any]) -> Any:
    return _http_request("POST", url, api_key, payload=payload)


def fetch_all_named_people(base_url: str, api_key: str) -> list[dict[str, Any]]:
    """Return all Immich people that have a non-empty name (paginates automatically)."""
    people: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _http_get(
            f"{base_url}/api/people",
            api_key,
            params={"page": str(page), "size": str(_PEOPLE_PAGE_SIZE)},
        )
        batch: list[Any] = data.get("people", []) if isinstance(data, dict) else []
        people.extend(
            p for p in batch
            if isinstance(p, dict) and str(p.get("name") or "").strip()
        )
        if not (isinstance(data, dict) and data.get("hasNextPage") and batch):
            break
        page += 1
    return people


def fetch_person_assets(base_url: str, api_key: str, person_id: str) -> list[dict[str, Any]]:
    """Return all assets that contain a given person."""
    assets: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _http_post(
            f"{base_url}/api/search/metadata",
            api_key,
            {
                "personIds": [person_id],
                "page": page,
                "size": _ASSET_PAGE_SIZE,
            },
        )
        asset_page = data.get("assets", {}) if isinstance(data, dict) else {}
        batch = asset_page.get("items", []) if isinstance(asset_page, dict) else []
        assets.extend(asset for asset in batch if isinstance(asset, dict))
        next_page = asset_page.get("nextPage") if isinstance(asset_page, dict) else None
        if not next_page:
            break
        page = int(next_page)
    return assets


def fetch_assets_by_original_filename(base_url: str, api_key: str, original_filename: str) -> list[dict[str, Any]]:
    """Return all assets whose original filename matches exactly."""
    assets: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _http_post(
            f"{base_url}/api/search/metadata",
            api_key,
            {
                "originalFileName": original_filename,
                "page": page,
                "size": _ASSET_PAGE_SIZE,
            },
        )
        asset_page = data.get("assets", {}) if isinstance(data, dict) else {}
        batch = asset_page.get("items", []) if isinstance(asset_page, dict) else []
        assets.extend(asset for asset in batch if isinstance(asset, dict))
        next_page = asset_page.get("nextPage") if isinstance(asset_page, dict) else None
        if not next_page:
            break
        page = int(next_page)
    return assets


def fetch_asset_faces(base_url: str, api_key: str, asset_id: str) -> list[dict[str, Any]]:
    """Return all face detections for an asset (named and unnamed)."""
    data = _http_get(
        f"{base_url}/api/faces",
        api_key,
        params={"id": asset_id},
    )
    return list(data) if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Local file indexing
# ---------------------------------------------------------------------------


def build_local_index(
    photos_root: Path,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> dict[str, Path]:
    """Walk *photos_root* and return a map of lowercase filename stem → file path."""
    index: dict[str, Path] = {}
    for path in photos_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.stem.lower()
            if key not in index:
                index[key] = path
    return index


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def _bbox_to_relative(face: dict[str, Any]) -> dict[str, float] | None:
    img_w = int(face.get("imageWidth", 0))
    img_h = int(face.get("imageHeight", 0))
    if img_w <= 0 or img_h <= 0:
        return None
    x1 = int(face.get("boundingBoxX1", 0))
    y1 = int(face.get("boundingBoxY1", 0))
    x2 = int(face.get("boundingBoxX2", 0))
    y2 = int(face.get("boundingBoxY2", 0))
    pw = x2 - x1
    ph = y2 - y1
    if pw <= 0 or ph <= 0:
        return None
    return {"rx": x1 / img_w, "ry": y1 / img_h, "rw": pw / img_w, "rh": ph / img_h}


def _face_to_region(face: dict[str, Any]) -> dict[str, Any] | None:
    person = face.get("person")
    if not isinstance(person, dict):
        return None
    name = str(person.get("name") or "").strip()
    if not name:
        return None
    coords = _bbox_to_relative(face)
    if coords is None:
        return None
    return {
        "name": name,
        "image_width": int(face.get("imageWidth", 0)),
        "image_height": int(face.get("imageHeight", 0)),
        **coords,
    }


def _faces_to_regions(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Immich face dicts to relative-coordinate region dicts."""
    return [r for f in faces if (r := _face_to_region(f)) is not None]


def _local_image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        log.debug("could not read local image dimensions for %s: %s", path, exc)
        return 0, 0


def _dedupe_names(names: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for n in names:
        seen.setdefault(n.casefold(), n)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main sync entry point
# ---------------------------------------------------------------------------


def _upsert_people_to_store(people: list[dict[str, Any]], store: TextFaceStore) -> int:
    existing: set[str] = {p["display_name"].casefold() for p in store.list_people()}
    count = 0
    for person in people:
        name = str(person.get("name") or "").strip()
        if name and name.casefold() not in existing:
            store.add_person(name)
            log.info("Added to cast store: %s", name)
            existing.add(name.casefold())
            count += 1
    return count


def _write_asset_xmp(
    asset_id: str,
    local_path: Path,
    base_url: str,
    api_key: str,
    dry_run: bool,
) -> tuple[int, int]:
    raw_faces = fetch_asset_faces(base_url, api_key, asset_id)
    named = [
        f for f in raw_faces
        if isinstance(f.get("person"), dict)
        and str(f["person"].get("name") or "").strip()
    ]
    if not named:
        return 0, 1
    names = _dedupe_names([str(f["person"]["name"]).strip() for f in named])
    regions = _faces_to_regions(named)
    local_width, local_height = _local_image_dimensions(local_path)
    if local_width > 0 and local_height > 0:
        for region in regions:
            region["image_width"] = local_width
            region["image_height"] = local_height
    xmp_path = local_path.with_suffix(".xmp")
    if dry_run:
        print(f"[DRY RUN] {xmp_path.name}: {', '.join(names)}")
        return 1, 0
    merge_persons_xmp(xmp_path, names)
    if regions:
        merge_face_regions_xmp(xmp_path, regions)
    log.info("Updated %s -> %s", xmp_path.name, ", ".join(names))
    return 1, 0


def sync_immich_faces(
    base_url: str,
    api_key: str,
    photos_root: Path,
    store: TextFaceStore,
    *,
    dry_run: bool = False,
    update_castdb: bool = True,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> dict[str, int]:
    """Pull named face data from Immich and update local XMP sidecars.

    Returns a stats dict with keys:
      people_synced   - new people added to the cast store
      assets_matched  - Immich assets matched to local files
      xmp_updated     - XMP sidecars written
      xmp_skipped     - matched assets with no named faces (nothing to write)
    """
    stats: dict[str, int] = {
        "people_synced": 0,
        "assets_matched": 0,
        "xmp_updated": 0,
        "xmp_skipped": 0,
    }

    local_index = build_local_index(photos_root, extensions)
    log.info("Indexed %d local photos under %s", len(local_index), photos_root)

    people = fetch_all_named_people(base_url, api_key)
    log.info("Found %d named people in Immich", len(people))
    if not people:
        return stats

    if update_castdb and not dry_run:
        stats["people_synced"] = _upsert_people_to_store(people, store)

    asset_to_local: dict[str, Path] = {}
    query_started_at = time.monotonic()
    total_queries = len(people)
    for query_count, person in enumerate(people, start=1):
        for asset in fetch_person_assets(base_url, api_key, str(person.get("id") or "")):
            asset_id = str(asset.get("id") or "").strip()
            stem = Path(str(asset.get("originalFileName") or "")).stem.lower()
            if asset_id and stem in local_index and asset_id not in asset_to_local:
                asset_to_local[asset_id] = local_index[stem]
        elapsed = max(time.monotonic() - query_started_at, 1e-9)
        print(
            f"Immich asset queries: {query_count / total_queries:.2%} "
            f"({query_count}/{total_queries}) {query_count / elapsed:.2f} queries/s",
            end="\r" if query_count < total_queries else "\n",
            flush=True,
        )

    stats["assets_matched"] = len(asset_to_local)
    log.info("Matched %d Immich assets to local files", len(asset_to_local))

    total_face_queries = len(asset_to_local)
    if total_face_queries:
        face_query_started_at = time.monotonic()
        for face_query_count, (asset_id, local_path) in enumerate(asset_to_local.items(), start=1):
            updated, skipped = _write_asset_xmp(asset_id, local_path, base_url, api_key, dry_run)
            stats["xmp_updated"] += updated
            stats["xmp_skipped"] += skipped
            elapsed = max(time.monotonic() - face_query_started_at, 1e-9)
            print(
                f"Immich face queries: {face_query_count / total_face_queries:.2%} "
                f"({face_query_count}/{total_face_queries}) {face_query_count / elapsed:.2f} queries/s",
                end="\r" if face_query_count < total_face_queries else "\n",
                flush=True,
            )

    return stats
