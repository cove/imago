from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..naming import SCAN_NAME_RE, parse_album_filename
from .xmp_sidecar import read_ai_sidecar_state

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CAST_SIGNATURE_FILES = ("people.json", "faces.json", "review_queue.jsonl")


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _matches_album_filter(image_path: Path, album: str) -> bool:
    album_value = _clean_text(album)
    if not album_value:
        return True
    return album_value.casefold() in image_path.parent.name.casefold()


def _load_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = _clean_text(line)
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _build_manifest_lookup(manifest_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        image_path = _clean_text(row.get("image_path"))
        if not image_path:
            continue
        raw_path = Path(image_path)
        lookup[str(raw_path)] = row
        try:
            lookup[str(raw_path.resolve())] = row
        except OSError:
            pass
    return lookup


def _discover_photo_files(photos_root: str | Path, *, album: str = "") -> list[Path]:
    root = Path(photos_root)
    if not root.is_dir():
        raise ValueError(f"photos_root is not a directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not _matches_album_filter(path, album):
            continue
        files.append(path.resolve())
    files.sort()
    return files


def _is_cover_candidate(image_path: Path) -> bool:
    collection, year, book, page = parse_album_filename(image_path.name)
    if collection != "Unknown" and year != "Unknown" and book != "00":
        return page in {"00", "01"}
    upper_name = image_path.name.upper()
    return "_P00" in upper_name or "_P01" in upper_name


def _scan_page_key(image_path: Path) -> str:
    match = SCAN_NAME_RE.search(image_path.name)
    if not match:
        return ""
    return (
        f"{match.group('collection')}_{match.group('year')}_B{match.group('book')}_P{match.group('page')}"
    ).casefold()


def _scan_group_paths(image_path: Path) -> list[Path]:
    page_key = _scan_page_key(image_path)
    if not page_key:
        return [image_path]
    group_paths = [
        path.resolve()
        for path in image_path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and _scan_page_key(path) == page_key
    ]
    group_paths.sort(key=lambda path: path.name.casefold())
    return group_paths or [image_path]


def _sidecar_state(image_path: Path) -> tuple[Path, bool, dict[str, Any] | None]:
    sidecar_path = image_path.with_suffix(".xmp")
    if not sidecar_path.is_file():
        return sidecar_path, False, None
    state = read_ai_sidecar_state(sidecar_path)
    return sidecar_path, True, state if isinstance(state, dict) else None


def _sidecar_current(image_path: Path, sidecar_path: Path) -> bool:
    if not sidecar_path.is_file():
        return False
    try:
        return int(sidecar_path.stat().st_mtime_ns) >= int(image_path.stat().st_mtime_ns)
    except FileNotFoundError:
        return False


def _current_cast_signature(cast_store: str | Path) -> str:
    root = Path(cast_store)
    parts: list[str] = []
    for name in CAST_SIGNATURE_FILES:
        path = root / name
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}")
        except FileNotFoundError:
            parts.append(f"{path.name}:missing")
    return "|".join(parts)


def _image_summary(
    image_path: Path,
    manifest_row: dict[str, Any] | None,
    sidecar_path: Path,
    sidecar_state: dict[str, Any] | None,
) -> dict[str, Any]:
    scan_match = SCAN_NAME_RE.search(image_path.name)
    album_title = _clean_text((sidecar_state or {}).get("album_title"))
    return {
        "image_path": str(image_path),
        "file_name": image_path.name,
        "album_dir": image_path.parent.name,
        "page": (_clean_text(scan_match.group("page")) if scan_match else ""),
        "scan": (_clean_text(scan_match.group("scan")) if scan_match else ""),
        "is_cover_candidate": _is_cover_candidate(image_path),
        "manifest_present": manifest_row is not None,
        "manifest_state": _clean_text((manifest_row or {}).get("state")),
        "sidecar_path": str(sidecar_path),
        "sidecar_present": sidecar_path.is_file(),
        "sidecar_current": _sidecar_current(image_path, sidecar_path),
        "album_title": album_title,
    }


def query_manifest_rows(
    *,
    photos_root: str | Path,
    manifest_path: str | Path,
    album: str = "",
    state: str = "",
    file_name: str = "",
    limit: int = 100,
) -> dict[str, Any]:
    state_value = _clean_text(state)
    file_value = _clean_text(file_name)
    rows = _load_manifest_rows(manifest_path)
    matches: list[dict[str, Any]] = []
    for row in rows:
        image_path_text = _clean_text(row.get("image_path"))
        if not image_path_text:
            continue
        image_path = Path(image_path_text)
        if not _matches_album_filter(image_path, album):
            continue
        if state_value and _clean_text(row.get("state")) != state_value:
            continue
        if file_value and image_path.name.casefold() != file_value.casefold():
            continue
        sidecar_path, _sidecar_exists, sidecar_state = _sidecar_state(image_path)
        entry = dict(row)
        entry.update(_image_summary(image_path, row, sidecar_path, sidecar_state))
        matches.append(entry)
    matches.sort(
        key=lambda row: (str(row.get("album_dir") or "").casefold(), str(row.get("file_name") or "").casefold())
    )
    limited = matches[: max(0, int(limit))]
    return {
        "photos_root": str(Path(photos_root)),
        "manifest_path": str(Path(manifest_path)),
        "album_filter": _clean_text(album),
        "state_filter": state_value,
        "file_name_filter": file_value,
        "total_matches": len(matches),
        "rows": limited,
    }


def album_status(
    *,
    photos_root: str | Path,
    manifest_path: str | Path,
    album: str,
) -> dict[str, Any]:
    album_value = _clean_text(album)
    if not album_value:
        raise ValueError("Provide an album filter.")
    files = _discover_photo_files(photos_root, album=album_value)
    manifest_rows = _load_manifest_rows(manifest_path)
    manifest_lookup = _build_manifest_lookup(manifest_rows)

    cover_candidates: list[dict[str, Any]] = []
    manifest_present = 0
    sidecar_present = 0
    current_sidecars = 0
    parent_dirs = sorted({path.parent.name for path in files})

    for image_path in files:
        manifest_row = manifest_lookup.get(str(image_path))
        sidecar_path, sidecar_exists, sidecar_state = _sidecar_state(image_path)
        if manifest_row is not None:
            manifest_present += 1
        if sidecar_exists:
            sidecar_present += 1
        if _sidecar_current(image_path, sidecar_path):
            current_sidecars += 1
        if not _is_cover_candidate(image_path):
            continue
        entry = _image_summary(image_path, manifest_row, sidecar_path, sidecar_state)
        entry["ready"] = bool(entry["sidecar_present"] and entry["album_title"])
        cover_candidates.append(entry)

    cover_candidates.sort(key=lambda row: (str(row.get("page") or ""), str(row.get("scan") or ""), row["file_name"]))
    return {
        "photos_root": str(Path(photos_root)),
        "manifest_path": str(Path(manifest_path)),
        "album_filter": album_value,
        "matched_parent_dirs": parent_dirs,
        "total_images": len(files),
        "manifest_rows": manifest_present,
        "sidecars_present": sidecar_present,
        "current_sidecars": current_sidecars,
        "cover_ready": any(bool(row.get("ready")) for row in cover_candidates),
        "cover_candidates": cover_candidates,
    }


def read_job_artifacts(
    *,
    job: dict[str, Any],
    kind: str = "",
    file_name: str = "",
) -> dict[str, Any]:
    artifact_file = _clean_text(job.get("artifact_file"))
    if not artifact_file:
        return {"job_id": _clean_text(job.get("id")), "total_matches": 0, "artifacts": []}
    path = Path(artifact_file)
    if not path.exists():
        return {
            "job_id": _clean_text(job.get("id")),
            "artifact_file": str(path),
            "total_matches": 0,
            "artifacts": [],
        }
    kind_value = _clean_text(kind)
    file_value = _clean_text(file_name).casefold()
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = _clean_text(line)
        if not text:
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            continue
        if kind_value and _clean_text(row.get("kind")) != kind_value:
            continue
        if file_value:
            candidate_names = {
                Path(_clean_text(row.get("image_path"))).name.casefold(),
                Path(_clean_text(row.get("sidecar_path"))).name.casefold(),
                _clean_text(row.get("label")).casefold(),
            }
            if file_value not in candidate_names:
                continue
        matches.append(row)
    return {
        "job_id": _clean_text(job.get("id")),
        "job_name": _clean_text(job.get("name")),
        "artifact_file": str(path),
        "kind_filter": kind_value,
        "file_name_filter": _clean_text(file_name),
        "total_matches": len(matches),
        "artifacts": matches,
    }


def reprocess_audit(
    *,
    photos_root: str | Path,
    manifest_path: str | Path,
    cast_store: str | Path,
    album: str = "",
    limit: int = 100,
) -> dict[str, Any]:
    files = _discover_photo_files(photos_root, album=album)
    manifest_rows = _load_manifest_rows(manifest_path)
    manifest_lookup = _build_manifest_lookup(manifest_rows)
    current_cast_signature = _current_cast_signature(cast_store)

    matches: list[dict[str, Any]] = []
    for image_path in files:
        manifest_row = manifest_lookup.get(str(image_path))
        sidecar_path, _sidecar_exists, sidecar_state = _sidecar_state(image_path)
        reasons: list[str] = []

        if manifest_row is None:
            reasons.append("manifest_missing")
        if not sidecar_path.is_file():
            reasons.append("sidecar_missing")
        elif not _sidecar_current(image_path, sidecar_path):
            reasons.append("sidecar_older_than_image")

        if len(_scan_group_paths(image_path)) > 1:
            if _clean_text((sidecar_state or {}).get("ocr_authority_source")) != "archive_stitched":
                reasons.append("missing_stitched_authority")

        recorded_cast_signature = _clean_text((manifest_row or {}).get("cast_store_signature"))
        people_detected = (sidecar_state or {}).get("people_detected")
        if (
            manifest_row is not None
            and recorded_cast_signature != current_cast_signature
            and people_detected is not False
        ):
            reasons.append("cast_store_signature_changed")

        if not reasons:
            continue

        entry = _image_summary(image_path, manifest_row, sidecar_path, sidecar_state)
        entry["reprocess_reasons"] = reasons
        matches.append(entry)

    matches.sort(
        key=lambda row: (str(row.get("album_dir") or "").casefold(), str(row.get("file_name") or "").casefold())
    )
    limited = matches[: max(0, int(limit))]
    return {
        "photos_root": str(Path(photos_root)),
        "manifest_path": str(Path(manifest_path)),
        "cast_store": str(Path(cast_store)),
        "album_filter": _clean_text(album),
        "total_matches": len(matches),
        "rows": limited,
    }
