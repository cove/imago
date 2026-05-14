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


# ---------------------------------------------------------------------------
# Immich API helpers (stdlib urllib — no extra deps)
# ---------------------------------------------------------------------------


def _http_get(url: str, api_key: str, params: dict[str, str] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"x-api-key": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        log.error("Immich HTTP %d %s: %s", exc.code, url, exc.reason)
        raise


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
        for person in batch:
            if isinstance(person, dict) and str(person.get("name") or "").strip():
                people.append(person)
        if not (isinstance(data, dict) and data.get("hasNextPage") and batch):
            break
        page += 1
    return people


def fetch_person_assets(base_url: str, api_key: str, person_id: str) -> list[dict[str, Any]]:
    """Return all assets that contain a given person."""
    data = _http_get(
        f"{base_url}/api/people/{urllib.parse.quote(person_id)}/assets",
        api_key,
    )
    return list(data) if isinstance(data, list) else []


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


def _faces_to_regions(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Immich face dicts to relative-coordinate region dicts."""
    regions: list[dict[str, Any]] = []
    for face in faces:
        person = face.get("person")
        if not isinstance(person, dict):
            continue
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        img_w = int(face.get("imageWidth") or 0)
        img_h = int(face.get("imageHeight") or 0)
        if img_w <= 0 or img_h <= 0:
            continue
        x1 = int(face.get("boundingBoxX1") or 0)
        y1 = int(face.get("boundingBoxY1") or 0)
        x2 = int(face.get("boundingBoxX2") or 0)
        y2 = int(face.get("boundingBoxY2") or 0)
        pw = x2 - x1
        ph = y2 - y1
        if pw <= 0 or ph <= 0:
            continue
        regions.append({
            "name": name,
            "rx": x1 / img_w,
            "ry": y1 / img_h,
            "rw": pw / img_w,
            "rh": ph / img_h,
        })
    return regions


def _dedupe_names(names: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for n in names:
        seen.setdefault(n.casefold(), n)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main sync entry point
# ---------------------------------------------------------------------------


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
      people_synced   – new people added to the cast store
      assets_matched  – Immich assets matched to local files
      xmp_updated     – XMP sidecars written
      xmp_skipped     – matched assets with no named faces (nothing to write)
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

    # Sync person names into cast store so they're available for face matching
    if update_castdb and not dry_run:
        existing: set[str] = {p["display_name"].casefold() for p in store.list_people()}
        for person in people:
            name = str(person.get("name") or "").strip()
            if name and name.casefold() not in existing:
                store.add_person(name)
                log.info("Added to cast store: %s", name)
                existing.add(name.casefold())
                stats["people_synced"] += 1

    # Collect asset_id → local path by matching Immich originalFileName stems
    asset_to_local: dict[str, Path] = {}
    for person in people:
        for asset in fetch_person_assets(base_url, api_key, str(person.get("id") or "")):
            asset_id = str(asset.get("id") or "").strip()
            stem = Path(str(asset.get("originalFileName") or "")).stem.lower()
            if asset_id and stem in local_index and asset_id not in asset_to_local:
                asset_to_local[asset_id] = local_index[stem]

    stats["assets_matched"] = len(asset_to_local)
    log.info("Matched %d Immich assets to local files", len(asset_to_local))

    # Fetch faces per matched asset and write XMP sidecars
    for asset_id, local_path in asset_to_local.items():
        raw_faces = fetch_asset_faces(base_url, api_key, asset_id)
        named = [
            f for f in raw_faces
            if isinstance(f.get("person"), dict)
            and str(f["person"].get("name") or "").strip()
        ]
        if not named:
            stats["xmp_skipped"] += 1
            continue

        names = _dedupe_names([str(f["person"]["name"]).strip() for f in named])
        regions = _faces_to_regions(named)
        xmp_path = local_path.with_suffix(".xmp")

        if dry_run:
            print(f"[DRY RUN] {xmp_path.name}: {', '.join(names)}")
            stats["xmp_updated"] += 1
            continue

        merge_persons_xmp(xmp_path, names)
        if regions:
            merge_face_regions_xmp(xmp_path, regions)
        log.info("Updated %s → %s", xmp_path.name, ", ".join(names))
        stats["xmp_updated"] += 1

    return stats
