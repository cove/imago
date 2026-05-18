from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.lib.ai_location import _xmp_gps_to_decimal
from photoalbums.lib.xmp_sidecar import (
    read_ai_sidecar_state,
    read_locations_shown,
    read_person_in_image,
    write_xmp_sidecar,
)
from photoalbums.naming import parse_album_filename

LA_AREA_BOUNDS = {
    "lat_min": 33.65,
    "lat_max": 34.35,
    "lon_min": -118.95,
    "lon_max": -117.65,
}
TRUSTED_CITY_STATE_PAIRS = {
    ("san marino", "california"),
    ("san marino", "ca"),
    ("altadena", "california"),
    ("altadena", "ca"),
}
SNAPSHOT_FIELDS = (
    "gps_latitude",
    "gps_longitude",
    "location_city",
    "location_state",
    "location_country",
    "location_sublocation",
    "location_created",
)


def _location_in_la_area(state: dict) -> bool:
    gps_latitude = str(state.get("gps_latitude") or "").strip()
    gps_longitude = str(state.get("gps_longitude") or "").strip()
    if not gps_latitude or not gps_longitude:
        return False
    try:
        lat = float(_xmp_gps_to_decimal(gps_latitude, axis="lat"))
        lon = float(_xmp_gps_to_decimal(gps_longitude, axis="lon"))
    except ValueError:
        return False
    return (
        LA_AREA_BOUNDS["lat_min"] <= lat <= LA_AREA_BOUNDS["lat_max"]
        and LA_AREA_BOUNDS["lon_min"] <= lon <= LA_AREA_BOUNDS["lon_max"]
    )


def _is_trusted_location(state: dict) -> bool:
    city = str(state.get("location_city") or "").strip().casefold()
    region = str(state.get("location_state") or "").strip().casefold()
    return (city, region) in TRUSTED_CITY_STATE_PAIRS or _location_in_la_area(state)


def _iter_sidecars(photos_root: Path) -> list[Path]:
    return sorted(path for path in photos_root.rglob("*.xmp") if path.is_file())


def _snapshot_record(sidecar_path: Path, photos_root: Path, state: dict) -> dict[str, str]:
    return {
        "path": str(sidecar_path.relative_to(photos_root)),
        **{field: str(state.get(field) or "").strip() for field in SNAPSHOT_FIELDS},
    }


def _is_title_page(sidecar_path: Path) -> bool:
    _, _, _, page_str = parse_album_filename(sidecar_path.name)
    return page_str.isdigit() and int(page_str) == 1


def snapshot_locations(photos_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for sidecar_path in _iter_sidecars(photos_root):
        if _is_title_page(sidecar_path):
            continue
        try:
            state = read_ai_sidecar_state(sidecar_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read XMP sidecar {sidecar_path}: {exc}") from exc
        if isinstance(state, dict) and _is_trusted_location(state):
            records.append(_snapshot_record(sidecar_path, photos_root, state))
    return records


def _manual_location_payload(record: dict[str, str]) -> dict[str, str]:
    return {
        "gps_latitude": record["gps_latitude"],
        "gps_longitude": record["gps_longitude"],
        "city": record["location_city"],
        "state": record["location_state"],
        "country": record["location_country"],
        "sublocation": record["location_sublocation"],
    }


def _restore_sidecar(sidecar_path: Path, record: dict[str, str]) -> None:
    state = read_ai_sidecar_state(sidecar_path)
    if not isinstance(state, dict):
        raise ValueError(f"Could not parse regenerated XMP sidecar: {sidecar_path}")
    detections = state.get("detections")
    detections_payload = dict(detections) if isinstance(detections, dict) else {}
    dc_date_values = state.get("dc_date_values")
    detections_payload["location"] = _manual_location_payload(record)
    write_xmp_sidecar(
        sidecar_path,
        person_names=read_person_in_image(sidecar_path),
        subjects=[],
        title=str(state.get("title") or ""),
        title_source=str(state.get("title_source") or ""),
        description=str(state.get("description") or ""),
        album_title=str(state.get("album_title") or ""),
        gps_latitude=record["gps_latitude"],
        gps_longitude=record["gps_longitude"],
        location_address=record["location_created"],
        location_city=record["location_city"],
        location_state=record["location_state"],
        location_country=record["location_country"],
        location_sublocation=record["location_sublocation"],
        source_text=str(state.get("source_text") or ""),
        ocr_text=str(state.get("ocr_text") or ""),
        parent_ocr_text=str(state.get("parent_ocr_text") or ""),
        ocr_lang=str(state.get("ocr_lang") or ""),
        author_text=str(state.get("author_text") or ""),
        scene_text=str(state.get("scene_text") or ""),
        detections_payload=detections_payload,
        stitch_key=str(state.get("stitch_key") or ""),
        ocr_authority_source=str(state.get("ocr_authority_source") or ""),
        create_date=str(state.get("create_date") or ""),
        dc_date=list(dc_date_values) if isinstance(dc_date_values, list) else [],
        date_time_original=str(state.get("date_time_original") or ""),
        replace_dc_date=True,
        locations_shown=read_locations_shown(sidecar_path),
    )


def restore_locations(photos_root: Path, records: list[dict[str, str]], *, dry_run: bool) -> tuple[int, int]:
    restored = 0
    missing = 0
    for record in records:
        sidecar_path = photos_root / record["path"]
        if not sidecar_path.is_file():
            missing += 1
            print(f"MISS  {sidecar_path}")
            continue
        if dry_run:
            print(f"PLAN  {sidecar_path}")
        else:
            _restore_sidecar(sidecar_path, record)
            print(f"REST  {sidecar_path}")
        restored += 1
    return restored, missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot or restore trusted San Marino, Altadena, and LA-area XMP locations around a full regeneration."
    )
    parser.add_argument("--photos-root", default=str(PHOTO_ALBUMS_DIR), help="Photo Albums root directory.")
    parser.add_argument("--snapshot-path", required=True, help="JSON snapshot path.")
    parser.add_argument("action", choices=("snapshot", "restore"))
    parser.add_argument("--run", action="store_true", help="Write restore changes in place. Omit for a dry run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    photos_root = Path(args.photos_root)
    if not photos_root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {photos_root}")
    snapshot_path = Path(args.snapshot_path)

    if args.action == "snapshot":
        records = snapshot_locations(photos_root)
        snapshot_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"snapshot records={len(records)} path={snapshot_path}")
        return 0

    records = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Snapshot file must contain a list: {snapshot_path}")
    restored, missing = restore_locations(photos_root, records, dry_run=not bool(args.run))
    if args.run:
        print(f"done restored={restored} missing={missing}")
    else:
        print(f"dry-run would_restore={restored} missing={missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
