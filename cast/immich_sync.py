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

import cv2

from .ingest import FaceIngestor, compute_arcface_embedding, estimate_face_quality
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


def build_exact_local_index(
    photos_root: Path,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in photos_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.name
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
            continue
        index[key] = path
    if duplicates:
        name, paths = next(iter(duplicates.items()))
        joined = ", ".join(str(path) for path in paths)
        raise RuntimeError(f"Duplicate local filename for Immich exact match: {name}: {joined}")
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
    return {"name": name, **coords}


def _faces_to_regions(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Immich face dicts to relative-coordinate region dicts."""
    return [r for f in faces if (r := _face_to_region(f)) is not None]


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


def _manual_region_bbox(face: dict[str, Any]) -> list[int] | None:
    x1 = int(face.get("boundingBoxX1", 0))
    y1 = int(face.get("boundingBoxY1", 0))
    x2 = int(face.get("boundingBoxX2", 0))
    y2 = int(face.get("boundingBoxY2", 0))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def _named_manual_faces(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        face
        for face in faces
        if isinstance(face.get("person"), dict) and str(face["person"].get("name") or "").strip()
    ]


def _person_id_for_name(store: TextFaceStore, name: str) -> str:
    clean = str(name or "").strip()
    for person in store.list_people():
        if str(person.get("display_name") or "").strip().casefold() == clean.casefold():
            return str(person["person_id"])
    return str(store.add_person(clean)["person_id"])


def _manual_region_candidates(
    ingestor: FaceIngestor,
    image_bgr,
    bbox: list[int],
) -> list[tuple[list[int], Any, float, list[float]]]:
    x, y, w, h = bbox
    image_h, image_w = image_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image_w, x + w)
    y1 = min(image_h, y + h)
    region = image_bgr[y0:y1, x0:x1]
    candidates: list[tuple[list[int], Any, float, list[float]]] = []
    for dx, dy, dw, dh in ingestor._detect(region, min_size=24):
        crop = region[dy : dy + dh, dx : dx + dw]
        if not ingestor.is_valid_face_crop(crop):
            continue
        embedding = compute_arcface_embedding(crop)
        if embedding is None:
            continue
        absolute_bbox = [x0 + dx, y0 + dy, dw, dh]
        candidates.append((absolute_bbox, crop, estimate_face_quality(crop), embedding))
    return candidates


def _seed_face_review(
    store: TextFaceStore,
    *,
    local_path: Path,
    person_name: str,
    reason: str,
    asset_id: str,
    region_bbox: list[int],
    candidate_count: int,
    best_quality: float | None,
) -> None:
    store.add_face_review_seed(
        source_path=str(local_path),
        person_name=person_name,
        reason=reason,
        metadata={
            "ingest": "immich_cast_import",
            "asset_id": asset_id,
            "manual_region_bbox": list(region_bbox),
            "candidate_count": int(candidate_count),
            "best_quality": best_quality,
            "supervision": "presence_only",
        },
    )


def _import_manual_face_region(
    *,
    store: TextFaceStore,
    ingestor: FaceIngestor,
    image,
    local_path: Path,
    asset_id: str,
    face: dict[str, Any],
    min_face_quality: float,
) -> tuple[int, int]:
    person = dict(face.get("person") or {})
    person_name = str(person.get("name") or "").strip()
    region_bbox = _manual_region_bbox(face)
    if region_bbox is None:
        return 0, 0

    candidates = _manual_region_candidates(ingestor, image, region_bbox)
    best_quality = max((candidate[2] for candidate in candidates), default=None)
    if not candidates:
        _seed_face_review(
            store,
            local_path=local_path,
            person_name=person_name,
            reason="presence_only_no_visible_face",
            asset_id=asset_id,
            region_bbox=region_bbox,
            candidate_count=0,
            best_quality=None,
        )
        return 0, 1
    if len(candidates) > 1:
        _seed_face_review(
            store,
            local_path=local_path,
            person_name=person_name,
            reason="ambiguous_multiple_faces",
            asset_id=asset_id,
            region_bbox=region_bbox,
            candidate_count=len(candidates),
            best_quality=best_quality,
        )
        return 0, 1

    bbox, crop, quality, _embedding = candidates[0]
    if quality < float(min_face_quality):
        _seed_face_review(
            store,
            local_path=local_path,
            person_name=person_name,
            reason="rejected_low_quality",
            asset_id=asset_id,
            region_bbox=region_bbox,
            candidate_count=1,
            best_quality=quality,
        )
        return 0, 1

    person_id = _person_id_for_name(store, person_name)
    saved = ingestor.save_arcface_face_from_crop(
        crop_bgr=crop,
        person_id=person_id,
        source_type="photo",
        source_path=str(local_path),
        bbox=bbox,
        metadata={
            "ingest": "immich_cast_import",
            "asset_id": asset_id,
            "manual_region_bbox": list(region_bbox),
            "supervision": "visible_face_from_presence_region",
            "presence_person_name": person_name,
        },
        min_quality=min_face_quality,
    )
    if saved is None:
        raise RuntimeError(f"Failed to persist validated Immich face crop for {local_path}")
    return 1, 0


def import_immich_cast_faces(
    base_url: str,
    api_key: str,
    photos_root: Path,
    store: TextFaceStore,
    *,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
    min_face_quality: float = 0.20,
) -> dict[str, int]:
    stats = {
        "assets_matched": 0,
        "faces_imported": 0,
        "review_seeds": 0,
    }
    local_index = build_exact_local_index(photos_root, extensions)
    ingestor = FaceIngestor(store, require_primary_model=True)
    ingestor._ensure_primary_model_ready()

    for filename, local_path in local_index.items():
        assets = fetch_assets_by_original_filename(base_url, api_key, filename)
        if not assets:
            continue
        if len(assets) > 1:
            raise RuntimeError(f"Immich returned multiple assets for exact filename {filename}")
        asset = assets[0]
        asset_id = str(asset.get("id") or "").strip()
        if not asset_id:
            raise RuntimeError(f"Immich asset for {filename} did not include an id")
        stats["assets_matched"] += 1

        image = cv2.imread(str(local_path))
        if image is None:
            raise RuntimeError(f"Unable to read local image for Immich cast import: {local_path}")

        for face in _named_manual_faces(fetch_asset_faces(base_url, api_key, asset_id)):
            imported, seeded = _import_manual_face_region(
                store=store,
                ingestor=ingestor,
                image=image,
                local_path=local_path,
                asset_id=asset_id,
                face=face,
                min_face_quality=min_face_quality,
            )
            stats["faces_imported"] += imported
            stats["review_seeds"] += seeded
    return stats
