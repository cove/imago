#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import requests

ALBUM_DIR_SUFFIXES = ("_Archive", "_Pages", "_Photos")
DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".webp",
    ".mov",
    ".mp4",
    ".m4v",
)
IMMICH_URL_ENV = "IMMICH_URL"
IMMICH_API_KEY_ENV = "IMMICH_API_KEY"
DEFAULT_IMMICH_URL = "http://192.168.4.26:2283"
MANAGED_DESCRIPTION = "Managed by imago immich/create_photo_albums.py. Recreated from Photo Albums directories."
ASSET_SEARCH_PAGE_SIZE = 1000
ADD_ASSETS_CHUNK_SIZE = 500
LIBRARY_SCAN_POLL_SECONDS = 5
LIBRARY_SCAN_TIMEOUT_SECONDS = 45 * 60


@dataclass(frozen=True)
class LocalAlbum:
    name: str
    directories: tuple[Path, ...]
    files: tuple[Path, ...]


def _runtime_error(message: str) -> RuntimeError:
    return RuntimeError(message)


def _exit_runtime(message: str) -> NoReturn:
    raise RuntimeError(message)


def _exit_type(message: str) -> NoReturn:
    raise TypeError(message)


class ImmichClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> Any:
        url = f"{self.base_url}{path}"
        response = requests.request(
            method,
            url,
            headers={
                "x-api-key": self.api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=payload,
            params=params,
            timeout=self.timeout,
        )
        if response.status_code not in expected:
            details = response.text.strip()
            message = details or response.reason
            error = _runtime_error(f"Immich {method} {url} failed with HTTP {response.status_code}: {message}")
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise error from exc
            raise error
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def get_albums(self) -> list[dict[str, Any]]:
        data = self.request("GET", "/api/albums")
        if not isinstance(data, list):
            _exit_type(f"Immich /api/albums returned {type(data).__name__}, expected list")
        return [album for album in data if isinstance(album, dict)]

    def get_libraries(self) -> list[dict[str, Any]]:
        data = self.request("GET", "/api/libraries")
        if not isinstance(data, list):
            _exit_type(f"Immich /api/libraries returned {type(data).__name__}, expected list")
        return [library for library in data if isinstance(library, dict)]

    def get_jobs(self) -> dict[str, Any]:
        data = self.request("GET", "/api/jobs")
        if not isinstance(data, dict):
            _exit_type(f"Immich /api/jobs returned {type(data).__name__}, expected object")
        return data

    def scan_library(self, library_id: str) -> None:
        self.request("POST", f"/api/libraries/{library_id}/scan", expected=(204,))

    def delete_album(self, album_id: str) -> None:
        self.request("DELETE", f"/api/albums/{album_id}", expected=(200, 204))

    def create_album(self, name: str) -> dict[str, Any]:
        data = self.request(
            "POST",
            "/api/albums",
            payload={"albumName": name, "albumUsers": [], "description": MANAGED_DESCRIPTION, "assetIds": []},
            expected=(200, 201),
        )
        if not isinstance(data, dict):
            _exit_type(f"Immich create album returned {type(data).__name__}, expected object")
        return data

    def add_assets_to_album(self, album_id: str, asset_ids: Sequence[str]) -> None:
        if not asset_ids:
            return
        self.request("PUT", f"/api/albums/{album_id}/assets", payload={"ids": list(asset_ids)})

    def search_assets_by_original_filename(self, original_filename: str, *, visibility: str | None = None) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        page = 1
        while True:
            payload: dict[str, Any] = {
                "originalFileName": original_filename,
                "page": page,
                "size": ASSET_SEARCH_PAGE_SIZE,
            }
            if visibility:
                payload["visibility"] = visibility
            data = self.request(
                "POST",
                "/api/search/metadata",
                payload=payload,
            )
            asset_page = data.get("assets", {}) if isinstance(data, dict) else {}
            batch = asset_page.get("items", []) if isinstance(asset_page, dict) else []
            assets.extend(asset for asset in batch if isinstance(asset, dict))
            next_page = asset_page.get("nextPage") if isinstance(asset_page, dict) else None
            if not next_page:
                break
            page = int(next_page)
        return assets


def _album_name_from_dir_name(dirname: str) -> str | None:
    for suffix in ALBUM_DIR_SUFFIXES:
        if dirname.endswith(suffix):
            return dirname
    return None


def _album_base_name(album_name: str) -> str:
    for suffix in ALBUM_DIR_SUFFIXES:
        if album_name.endswith(suffix):
            return album_name[: -len(suffix)]
    return album_name


def _legacy_immich_album_name(album_name: str) -> str:
    return album_name.replace("_", " ")


def _immich_album_name(album_name: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", album_name)
    return spaced.replace("_", " ")


def _media_files(directory: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and not path.name.startswith(".")
    )


def discover_local_albums(photos_root: Path, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> list[LocalAlbum]:
    if not photos_root.exists():
        raise FileNotFoundError(photos_root)
    if not photos_root.is_dir():
        raise NotADirectoryError(photos_root)

    extension_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    albums: list[LocalAlbum] = []
    for path in sorted(photos_root.iterdir()):
        if not path.is_dir():
            continue
        album_name = _album_name_from_dir_name(path.name)
        if album_name:
            albums.append(LocalAlbum(name=album_name, directories=(path,), files=tuple(_media_files(path, extension_set))))
    return albums


def _path_key(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _relative_key(path: Path, root: Path) -> str:
    return _path_key(path.relative_to(root))


def _asset_path_key(asset: dict[str, Any]) -> str:
    return str(asset.get("originalPath") or "").replace("\\", "/").lower()


def _asset_id_for_file(
    client: ImmichClient,
    photos_root: Path,
    file_path: Path,
    cache: dict[str, list[dict[str, Any]]],
    *,
    progress: bool = False,
) -> str | None:
    if file_path.name not in cache:
        cache[file_path.name] = client.search_assets_by_original_filename(file_path.name)
        if not cache[file_path.name]:
            cache[file_path.name] = client.search_assets_by_original_filename(file_path.name, visibility="archive")
    if progress:
        sys.stdout.write(".")
        sys.stdout.flush()
    candidates = cache[file_path.name]
    if not candidates:
        return None

    relative = _relative_key(file_path, photos_root)
    path_matches = [asset for asset in candidates if _asset_path_key(asset).endswith(relative)]
    matches = path_matches or candidates
    if len(matches) > 1:
        paths = ", ".join(str(asset.get("originalPath") or asset.get("id") or "<unknown>") for asset in matches)
        _exit_runtime(f"Multiple Immich assets match {file_path}: {paths}")

    asset_id = str(matches[0].get("id") or "").strip()
    if not asset_id:
        _exit_runtime(f"Immich asset for {file_path} did not include an id")
    return asset_id


def resolve_album_asset_ids(
    client: ImmichClient,
    photos_root: Path,
    album: LocalAlbum,
    *,
    allow_missing_assets: bool = False,
    progress: bool = False,
) -> tuple[list[str], list[Path]]:
    cache: dict[str, list[dict[str, Any]]] = {}
    asset_ids: dict[str, None] = {}
    missing: list[Path] = []
    for file_path in album.files:
        asset_id = _asset_id_for_file(client, photos_root, file_path, cache, progress=progress)
        if asset_id is None:
            missing.append(file_path)
            continue
        asset_ids.setdefault(asset_id, None)

    if missing and not allow_missing_assets:
        examples = ", ".join(str(path) for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        _exit_runtime(f"{album.name}: {len(missing)} local file(s) did not match Immich assets: {examples}{suffix}")
    return list(asset_ids), missing


def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _job_counts(jobs: dict[str, Any], job_name: str) -> dict[str, int]:
    job = jobs.get(job_name)
    counts = job.get("jobCounts", {}) if isinstance(job, dict) else {}
    if not isinstance(counts, dict):
        _exit_type(f"Immich /api/jobs {job_name}.jobCounts returned {type(counts).__name__}, expected object")
    return {key: int(counts.get(key, 0) or 0) for key in ("active", "completed", "failed", "delayed", "waiting", "paused")}


def _resolve_library_id(client: ImmichClient, library_name: str) -> str:
    libraries = client.get_libraries()
    matches = [library for library in libraries if library.get("name") == library_name]
    if not matches:
        names = ", ".join(sorted(str(library.get("name") or "<unnamed>") for library in libraries))
        _exit_runtime(f"Immich library named {library_name!r} was not found. Available libraries: {names}")
    if len(matches) > 1:
        ids = ", ".join(str(library.get("id") or "<missing id>") for library in matches)
        _exit_runtime(f"Multiple Immich libraries named {library_name!r}: {ids}")
    library_id = str(matches[0].get("id") or "").strip()
    if not library_id:
        _exit_runtime(f"Immich library named {library_name!r} did not include an id")
    return library_id


def rescan_immich_library(
    client: ImmichClient,
    library_name: str,
    *,
    poll_seconds: int = LIBRARY_SCAN_POLL_SECONDS,
    timeout_seconds: int = LIBRARY_SCAN_TIMEOUT_SECONDS,
) -> None:
    library_id = _resolve_library_id(client, library_name)
    before_counts = _job_counts(client.get_jobs(), "library")
    failed_before = before_counts["failed"]
    sys.stdout.write(f"Triggering Immich library scan for {library_name} ({library_id})\n")
    sys.stdout.flush()
    client.scan_library(library_id)

    deadline = time.monotonic() + timeout_seconds
    last_counts: tuple[int, int, int, int, int] | None = None
    while True:
        counts = _job_counts(client.get_jobs(), "library")
        current_counts = (
            counts["waiting"],
            counts["active"],
            counts["delayed"],
            counts["completed"],
            counts["failed"],
        )
        if current_counts != last_counts:
            sys.stdout.write(
                "Immich library queue: "
                f"waiting={counts['waiting']} active={counts['active']} delayed={counts['delayed']} "
                f"completed={counts['completed']} failed={counts['failed']}\n"
            )
            sys.stdout.flush()
            last_counts = current_counts
        if counts["waiting"] == 0 and counts["active"] == 0 and counts["delayed"] == 0:
            if counts["failed"] > failed_before:
                _exit_runtime(f"Immich library scan failed count rose from {failed_before} to {counts['failed']}")
            return
        if time.monotonic() > deadline:
            _exit_runtime(f"Immich library scan did not finish within {timeout_seconds} seconds")
        time.sleep(poll_seconds)


def recreate_album(
    client: ImmichClient,
    album: LocalAlbum,
    asset_ids: Sequence[str],
    *,
    existing_albums: Sequence[dict[str, Any]],
    dry_run: bool,
) -> str | None:
    immich_name = _immich_album_name(album.name)
    replace_names = {album.name, _legacy_immich_album_name(album.name), immich_name}
    matches = [item for item in existing_albums if item.get("albumName") in replace_names]
    if dry_run:
        sys.stdout.write(
            f"DRY RUN {immich_name}: would delete {len(matches)} existing album(s), "
            f"create with {len(asset_ids)} asset(s)\n"
        )
        return None

    for match in matches:
        album_id = str(match.get("id") or "").strip()
        if not album_id:
            _exit_runtime(f"Existing Immich album named {album.name} did not include an id")
        client.delete_album(album_id)

    created = client.create_album(immich_name)
    album_id = str(created.get("id") or "").strip()
    if not album_id:
        _exit_runtime(f"Created Immich album {album.name} did not include an id")
    for chunk in _chunked(list(asset_ids), ADD_ASSETS_CHUNK_SIZE):
        client.add_assets_to_album(album_id, chunk)
    return album_id


def delete_obsolete_grouped_albums(
    client: ImmichClient,
    albums: Sequence[LocalAlbum],
    *,
    existing_albums: Sequence[dict[str, Any]],
    dry_run: bool,
) -> list[str]:
    grouped_names: set[str] = set()
    for album in albums:
        base_name = _album_base_name(album.name)
        if base_name != album.name:
            grouped_names.update({base_name, _legacy_immich_album_name(base_name), _immich_album_name(base_name)})

    deleted_ids: list[str] = []
    for match in existing_albums:
        if match.get("albumName") not in grouped_names:
            continue
        album_id = str(match.get("id") or "").strip()
        if not album_id:
            _exit_runtime(f"Existing Immich grouped album named {match.get('albumName')} did not include an id")
        deleted_ids.append(album_id)
        if dry_run:
            continue
        client.delete_album(album_id)
    return deleted_ids


def sync_photo_albums(
    client: ImmichClient,
    photos_root: Path,
    *,
    album_filter: set[str] | None = None,
    allow_missing_assets: bool = False,
    dry_run: bool = False,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
) -> int:
    albums = discover_local_albums(photos_root, extensions)
    if album_filter:
        albums = [album for album in albums if album.name in album_filter or _album_base_name(album.name) in album_filter]
    if not albums:
        _exit_runtime("No matching *_Archive, *_Pages, or *_Photos album directories found")

    resolved_albums: list[tuple[LocalAlbum, list[str], list[Path]]] = []
    for index, album in enumerate(albums, start=1):
        sys.stdout.write(f"[{index}/{len(albums)}] Resolving {_immich_album_name(album.name)} ({len(album.files)} file(s)) ")
        sys.stdout.flush()
        asset_ids, missing = resolve_album_asset_ids(
            client,
            photos_root,
            album,
            allow_missing_assets=True,
            progress=True,
        )
        missing_note = f", {len(missing)} missing" if missing else ""
        sys.stdout.write(f" {len(asset_ids)} asset(s){missing_note}\n")
        sys.stdout.flush()
        resolved_albums.append((album, asset_ids, missing))

    missing_albums = [(album, missing) for album, _, missing in resolved_albums if missing]
    if missing_albums and not allow_missing_assets:
        lines = []
        for album, missing in missing_albums[:5]:
            examples = ", ".join(str(path) for path in missing[:5])
            suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
            lines.append(f"{album.name}: {len(missing)} local file(s) did not match Immich assets: {examples}{suffix}")
        suffix = "" if len(missing_albums) <= 5 else f"\n... and {len(missing_albums) - 5} more album(s)"
        _exit_runtime("Missing Immich assets; no albums were deleted or created:\n" + "\n".join(lines) + suffix)

    existing_albums = client.get_albums()
    deleted_grouped_ids = set(
        delete_obsolete_grouped_albums(
            client,
            albums,
            existing_albums=existing_albums,
            dry_run=dry_run,
        )
    )
    existing_albums = [
        album
        for album in existing_albums
        if str(album.get("id") or "").strip() not in deleted_grouped_ids
    ]
    for album, asset_ids, missing in resolved_albums:
        recreated_id = recreate_album(
            client,
            album,
            asset_ids,
            existing_albums=existing_albums,
            dry_run=dry_run,
        )
        missing_note = f", {len(missing)} missing" if missing else ""
        id_note = "" if recreated_id is None else f" ({recreated_id})"
        sys.stdout.write(f"{_immich_album_name(album.name)}: {len(asset_ids)} asset(s){missing_note}{id_note}\n")
    return 0


def _default_photos_root() -> Path:
    try:
        from photoalbums.common import get_photo_albums_dir
    except ImportError:
        return Path.cwd()
    return get_photo_albums_dir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recreate Immich albums from Photo Albums *_Archive, *_Pages, and *_Photos directories."
    )
    parser.add_argument("--photos-root", default=str(_default_photos_root()), help="Photo Albums root directory.")
    parser.add_argument(
        "--immich-url",
        default=os.environ.get(IMMICH_URL_ENV, DEFAULT_IMMICH_URL),
        help=f"Immich base URL. Defaults to {DEFAULT_IMMICH_URL}.",
    )
    parser.add_argument("--api-key", default=os.environ.get(IMMICH_API_KEY_ENV, ""), help="Immich API key.")
    parser.add_argument("--album", action="append", default=[], help="Album name to process; repeat to process several.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve local assets without deleting or creating albums.")
    parser.add_argument(
        "--rescan-library",
        action="store_true",
        help="Trigger and wait for an Immich external library scan before resolving assets.",
    )
    parser.add_argument(
        "--library-name",
        default="cordell",
        help="Immich external library name to scan when --rescan-library is set.",
    )
    parser.add_argument(
        "--allow-missing-assets",
        action="store_true",
        help="Create albums with matched assets even when some local files are not in Immich.",
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="Media extension to include. Repeat to override the default extension set.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_url = str(args.immich_url or "").rstrip("/")
    api_key = str(args.api_key or "")
    if not base_url:
        parser.error(f"--immich-url is required, or set {IMMICH_URL_ENV}")
    if not api_key:
        parser.error(f"--api-key is required, or set {IMMICH_API_KEY_ENV}")

    extensions = tuple(args.extension) if args.extension else DEFAULT_EXTENSIONS
    client = ImmichClient(base_url, api_key)
    if args.rescan_library:
        rescan_immich_library(client, str(args.library_name or ""))
    return sync_photo_albums(
        client,
        Path(args.photos_root).expanduser(),
        album_filter=set(args.album) if args.album else None,
        allow_missing_assets=bool(args.allow_missing_assets),
        dry_run=bool(args.dry_run),
        extensions=extensions,
    )


if __name__ == "__main__":
    sys.exit(main())
