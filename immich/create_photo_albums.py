#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
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
        self.request("POST", f"/api/albums/{album_id}/assets", payload={"ids": list(asset_ids)})

    def search_assets_by_original_filename(self, original_filename: str) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self.request(
                "POST",
                "/api/search/metadata",
                payload={"originalFileName": original_filename, "page": page, "size": ASSET_SEARCH_PAGE_SIZE},
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
            return dirname[: -len(suffix)]
    return None


def _immich_album_name(album_name: str) -> str:
    return album_name.replace("_", " ")


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
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(photos_root.iterdir()):
        if not path.is_dir():
            continue
        album_name = _album_name_from_dir_name(path.name)
        if album_name:
            grouped[album_name].append(path)

    albums: list[LocalAlbum] = []
    for name, directories in sorted(grouped.items()):
        files: list[Path] = []
        for directory in directories:
            files.extend(_media_files(directory, extension_set))
        albums.append(LocalAlbum(name=name, directories=tuple(directories), files=tuple(sorted(files))))
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
) -> str | None:
    if file_path.name not in cache:
        cache[file_path.name] = client.search_assets_by_original_filename(file_path.name)
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
) -> tuple[list[str], list[Path]]:
    cache: dict[str, list[dict[str, Any]]] = {}
    asset_ids: dict[str, None] = {}
    missing: list[Path] = []
    for file_path in album.files:
        asset_id = _asset_id_for_file(client, photos_root, file_path, cache)
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


def recreate_album(
    client: ImmichClient,
    album: LocalAlbum,
    asset_ids: Sequence[str],
    *,
    existing_albums: Sequence[dict[str, Any]],
    dry_run: bool,
) -> str | None:
    immich_name = _immich_album_name(album.name)
    replace_names = {album.name, immich_name}
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
        albums = [album for album in albums if album.name in album_filter]
    if not albums:
        _exit_runtime("No matching *_Archive, *_Pages, or *_Photos album directories found")

    existing_albums = client.get_albums()
    for album in albums:
        asset_ids, missing = resolve_album_asset_ids(
            client,
            photos_root,
            album,
            allow_missing_assets=allow_missing_assets,
        )
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
