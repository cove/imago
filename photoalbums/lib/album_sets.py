from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import tomllib

ALBUM_SETS_PATH = Path(__file__).resolve().parents[1] / "album_sets.toml"
VALID_KINDS = {"archive", "scanwatch"}


@dataclass(frozen=True, slots=True)
class AlbumSet:
    name: str
    kind: str
    photos_root: Path
    description: str = ""
    cast_store: Path | None = None
    skill: str = ""
    title_page_location: dict[str, str] | None = None

    def supports(self) -> list[str]:
        if self.kind == "archive":
            return [
                "photoalbums_manifest_summary",
                "photoalbums_manifest_query",
                "photoalbums_album_status",
                "photoalbums_load_xmp",
                "photoalbums_reprocess_audit",
                "photoalbums_ai_index",
                "photoalbums_compress",
                "photoalbums_stitch",
            ]
        return [
            "scanwatch_start",
            "scanwatch_stop",
            "scanwatch_status",
            "scanwatch_refresh",
            "scanwatch_list_events",
            "scanwatch_get_event",
            "scanwatch_list_rescans",
            "scanwatch_apply_decision",
        ]

    def to_client_dict(self, *, default_archive_set: str, default_scan_set: str) -> dict[str, object]:
        return {
            "album_set": self.name,
            "kind": self.kind,
            "description": self.description,
            "skill": self.skill,
            "is_default": self.name == (default_archive_set if self.kind == "archive" else default_scan_set),
            "supported_tools": self.supports(),
        }


def _normalize_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"Album set field '{field_name}' must be a non-empty string: {ALBUM_SETS_PATH}")
    return text


def _resolve_path(base_dir: Path, value: Any, *, field_name: str) -> Path:
    text = _normalize_text(value, field_name=field_name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve(strict=False)
    return path


def _parse_title_page_location(name: str, payload: Any) -> dict[str, str] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise RuntimeError(f"Album set '{name}' title_page_location must be a TOML table: {ALBUM_SETS_PATH}")
    location: dict[str, str] = {}
    for field_name in ("address", "gps_latitude", "gps_longitude", "city", "state", "country", "sublocation"):
        text = str(payload.get(field_name) or "").strip()
        if text:
            location[field_name] = text
    if not location:
        return None
    if not location.get("gps_latitude") or not location.get("gps_longitude"):
        raise RuntimeError(
            f"Album set '{name}' title_page_location must define gps_latitude and gps_longitude: {ALBUM_SETS_PATH}"
        )
    return location


def _parse_album_set(name: str, payload: Any, *, base_dir: Path) -> AlbumSet:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Album set '{name}' must be a TOML table: {ALBUM_SETS_PATH}")
    kind = _normalize_text(payload.get("kind"), field_name=f"sets.{name}.kind")
    if kind not in VALID_KINDS:
        raise RuntimeError(
            f"Album set '{name}' kind must be one of {', '.join(sorted(VALID_KINDS))}: {ALBUM_SETS_PATH}"
        )

    album_set = AlbumSet(
        name=name,
        kind=kind,
        description=str(payload.get("description") or "").strip(),
        photos_root=_resolve_path(base_dir, payload.get("photos_root"), field_name=f"sets.{name}.photos_root"),
        skill=str(payload.get("skill") or "").strip(),
        cast_store=(
            _resolve_path(base_dir, payload.get("cast_store"), field_name=f"sets.{name}.cast_store")
            if kind == "archive"
            else None
        ),
        title_page_location=_parse_title_page_location(name, payload.get("title_page_location")),
    )
    return album_set


@lru_cache(maxsize=1)
def load_album_sets() -> dict[str, object]:
    with open(ALBUM_SETS_PATH, "rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Album sets config must be a TOML table: {ALBUM_SETS_PATH}")
    raw_sets = payload.get("sets")
    if not isinstance(raw_sets, dict) or not raw_sets:
        raise RuntimeError(f"Album sets config must define a non-empty [sets] table: {ALBUM_SETS_PATH}")

    base_dir = ALBUM_SETS_PATH.parent
    sets = {name: _parse_album_set(name, set_payload, base_dir=base_dir) for name, set_payload in raw_sets.items()}
    default_archive_set = _normalize_text(payload.get("default_archive_set"), field_name="default_archive_set")
    default_scan_set = _normalize_text(payload.get("default_scan_set"), field_name="default_scan_set")

    if default_archive_set not in sets:
        raise RuntimeError(f"default_archive_set '{default_archive_set}' is not defined: {ALBUM_SETS_PATH}")
    if default_scan_set not in sets:
        raise RuntimeError(f"default_scan_set '{default_scan_set}' is not defined: {ALBUM_SETS_PATH}")
    if sets[default_archive_set].kind != "archive":
        raise RuntimeError(f"default_archive_set '{default_archive_set}' must be kind='archive': {ALBUM_SETS_PATH}")
    if sets[default_scan_set].kind != "scanwatch":
        raise RuntimeError(f"default_scan_set '{default_scan_set}' must be kind='scanwatch': {ALBUM_SETS_PATH}")

    return {
        "default_archive_set": default_archive_set,
        "default_scan_set": default_scan_set,
        "sets": sets,
    }


def list_album_sets(kind: str | None = None) -> list[AlbumSet]:
    config = load_album_sets()
    if kind is not None and kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(VALID_KINDS))}")
    sets = list(config["sets"].values())
    if kind:
        sets = [album_set for album_set in sets if album_set.kind == kind]
    return sorted(sets, key=lambda album_set: album_set.name)


def get_album_set(name: str) -> AlbumSet:
    config = load_album_sets()
    album_set = config["sets"].get(name)
    if album_set is None:
        raise ValueError(f"Unknown album_set '{name}'")
    return album_set


def default_archive_set_name() -> str:
    return str(load_album_sets()["default_archive_set"])


def default_scan_set_name() -> str:
    return str(load_album_sets()["default_scan_set"])


def resolve_archive_set(name: str | None = None) -> AlbumSet:
    album_set = get_album_set(name or default_archive_set_name())
    if album_set.kind != "archive":
        raise ValueError(f"album_set '{album_set.name}' does not support archive operations")
    return album_set


def read_people_roster(album_set: AlbumSet | str | None) -> dict[str, str]:
    if album_set is None:
        return {}
    album_set_name = album_set if isinstance(album_set, str) else album_set.name
    if not str(album_set_name or "").strip():
        return {}
    with open(ALBUM_SETS_PATH, "rb") as handle:
        payload = tomllib.load(handle)
    raw_sets = payload.get("sets")
    if not isinstance(raw_sets, dict):
        return {}
    raw_set = raw_sets.get(str(album_set_name))
    if not isinstance(raw_set, dict):
        return {}
    raw_people = raw_set.get("people")
    if not isinstance(raw_people, dict):
        return {}
    roster: dict[str, str] = {}
    for key, value in raw_people.items():
        shorthand = str(key or "").strip().lower()
        full_name = str(value or "").strip()
        if shorthand and full_name:
            roster[shorthand] = full_name
    return roster


def find_archive_set_by_photos_root(photos_root: str | Path) -> AlbumSet | None:
    target = Path(photos_root).expanduser().resolve(strict=False)
    for album_set in list_album_sets(kind="archive"):
        if album_set.photos_root.resolve(strict=False) == target:
            return album_set
    return None


def resolve_scan_set(name: str | None = None) -> AlbumSet:
    album_set = get_album_set(name or default_scan_set_name())
    if album_set.kind != "scanwatch":
        raise ValueError(f"album_set '{album_set.name}' does not support scanwatch operations")
    return album_set
