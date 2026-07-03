#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import http.server
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast, override

import requests
from PIL import Image, ImageOps

from photoalbums.common import get_photo_albums_dir
from photoalbums.lib.image_limits import allow_large_pillow_images
from photoalbums.naming import (
    ALBUM_DIR_SUFFIX_PAGES,
    ALBUM_DIR_SUFFIX_PHOTOS,
    album_dir_base_name,
    album_dir_suffix,
)

GOOGLE_PHOTOS_ACCESS_TOKEN_ENV = "GOOGLE_PHOTOS_ACCESS_TOKEN"
GOOGLE_PHOTOS_CLIENT_SECRETS_ENV = "GOOGLE_PHOTOS_CLIENT_SECRETS"
GOOGLE_PHOTOS_TOKEN_CACHE_ENV = "GOOGLE_PHOTOS_TOKEN_CACHE"
DEFAULT_GOOGLE_ACCOUNT = "dolorescordell@gmail.com"
DEFAULT_STAGING_ROOT = Path.home() / "PycharmProjects" / "Family Photos"
DEFAULT_CLIENT_SECRETS_PATH = Path.home() / ".config" / "imago" / "google-photos-oauth-client.json"
DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".webp",
)
ALBUM_DIR_SUFFIX_ORDER = {
    ALBUM_DIR_SUFFIX_PAGES: 0,
    ALBUM_DIR_SUFFIX_PHOTOS: 1,
}
GOOGLE_PHOTOS_API_URL = "https://photoslibrary.googleapis.com/v1"
UPLOADS_URL = f"{GOOGLE_PHOTOS_API_URL}/uploads"
GOOGLE_OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_PHOTOS_SCOPES = (
    "openid",
    "email",
    "https://www.googleapis.com/auth/photoslibrary.appendonly",
    "https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata",
)
BATCH_CREATE_LIMIT = 50
JPEG_QUALITY = 95
TOKEN_REFRESH_SKEW_SECONDS = 60
UPLOAD_TOKEN_TTL_SECONDS = 23 * 60 * 60
REQUEST_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
REQUEST_RETRY_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)
REQUEST_MAX_ATTEMPTS = 5
REQUEST_MAX_BACKOFF_SECONDS = 60
REQUEST_CONNECT_TIMEOUT_SECONDS = 15


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _log_status(message: str) -> None:
    sys.stderr.write(f"[{_timestamp()}] {message}\n")
    sys.stderr.flush()


def _vlog(verbose: bool, message: str) -> None:
    if verbose:
        _log_status(message)


def _request_label(method: str, url: str, context: str) -> str:
    endpoint = url.rsplit("/", 1)[-1].split("?", 1)[0]
    label = f"{method} {endpoint}"
    return f"{label} ({context})" if context else label


def _status_retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after and retry_after.isdecimal():
        return float(retry_after)
    return min(2**attempt, REQUEST_MAX_BACKOFF_SECONDS)


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


@dataclass(frozen=True)
class LocalAlbum:
    name: str
    directories: tuple[Path, ...]
    files: tuple[Path, ...]


@dataclass(frozen=True)
class PreparedMedia:
    source_path: Path
    staged_path: Path
    upload_name: str
    sha256: str = ""


@dataclass(frozen=True)
class ExistingMediaItem:
    id: str
    filename: str


@dataclass(frozen=True)
class GoogleOAuthCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    email: str


class GooglePhotosAlbumClient(Protocol):
    def create_album(self, title: str) -> dict[str, Any]: ...

    def list_albums(self) -> list[dict[str, Any]]: ...

    def list_album_media_items(self, album_id: str) -> list[ExistingMediaItem]: ...

    def upload_media(self, media_path: Path) -> str: ...

    def batch_create_media_items(self, album_id: str, items: Sequence[tuple[str, str]]) -> list[str]: ...

    def batch_add_media_items(self, album_id: str, media_item_ids: Sequence[str]) -> None: ...


def _exit_runtime(message: str) -> NoReturn:
    raise RuntimeError(message)


def _rewind_request_data(data: Any | None) -> None:
    seek = getattr(data, "seek", None)
    if callable(seek):
        seek(0)


def _parse_response(method: str, url: str, response: requests.Response, expected: tuple[int, ...]) -> Any:
    if response.status_code not in expected:
        details = response.text.strip()
        message = details or response.reason
        error = RuntimeError(f"Google Photos {method} {url} failed with HTTP {response.status_code}: {message}")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise error from exc
        raise error
    if response.status_code == 204 or not response.content:
        return None
    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        return response.text
    return response.json()


class GooglePhotosClient:
    def __init__(
        self,
        access_token: str,
        timeout: int = 60,
        refresh_access_token: Callable[[], str] | None = None,
        verbose: bool = False,
    ) -> None:
        self.access_token: str = access_token
        self.timeout: int = timeout
        self.refresh_access_token: Callable[[], str] | None = refresh_access_token
        self.verbose: bool = verbose

    def request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        expected: tuple[int, ...] = (200,),
        context: str = "",
    ) -> Any:
        label = _request_label(method, url, context)
        response: requests.Response | None = None
        refreshed_access_token = False
        for attempt in range(REQUEST_MAX_ATTEMPTS):
            if attempt:
                _rewind_request_data(data)
            request_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
            }
            request_headers.update(headers or {})
            _vlog(self.verbose, f"{label}: attempt {attempt + 1}/{REQUEST_MAX_ATTEMPTS}")
            started = time.monotonic()
            try:
                response = requests.request(
                    method,
                    url,
                    json=payload,
                    data=data,
                    headers=request_headers,
                    timeout=(REQUEST_CONNECT_TIMEOUT_SECONDS, self.timeout),
                )
            except REQUEST_RETRY_EXCEPTIONS as exc:
                self._handle_send_error(label, exc, attempt, started)
                continue
            _vlog(self.verbose, f"{label}: HTTP {response.status_code} in {time.monotonic() - started:.1f}s")
            if response.status_code == 401 and self.refresh_access_token is not None and not refreshed_access_token:
                _log_status(f"{label}: HTTP 401; refreshing access token")
                self.access_token = self.refresh_access_token()
                refreshed_access_token = True
                continue
            if not self._retry_after_backoff(label, response, attempt):
                break
        if response is None:
            _exit_runtime(f"Google Photos {method} {url} did not return a response")
        return _parse_response(method, url, response, expected)

    def _handle_send_error(self, label: str, exc: Exception, attempt: int, started: float) -> None:
        if attempt == REQUEST_MAX_ATTEMPTS - 1:
            _log_status(f"{label}: network error after {attempt + 1} attempts: {type(exc).__name__}: {exc}")
            raise exc
        delay = min(2**attempt, REQUEST_MAX_BACKOFF_SECONDS)
        _log_status(
            f"{label}: {type(exc).__name__} after {time.monotonic() - started:.1f}s; "
            f"retrying in {delay:.0f}s ({attempt + 1}/{REQUEST_MAX_ATTEMPTS})"
        )
        time.sleep(delay)

    def _retry_after_backoff(self, label: str, response: requests.Response, attempt: int) -> bool:
        if response.status_code not in REQUEST_RETRY_STATUS_CODES or attempt == REQUEST_MAX_ATTEMPTS - 1:
            return False
        delay = _status_retry_delay(response, attempt)
        throttled = " (throttled)" if response.status_code == 429 else ""
        _log_status(
            f"{label}: HTTP {response.status_code}{throttled}; "
            f"retrying in {delay:.0f}s ({attempt + 1}/{REQUEST_MAX_ATTEMPTS})"
        )
        time.sleep(delay)
        return True

    def create_album(self, title: str) -> dict[str, Any]:
        data = self.request(
            "POST",
            f"{GOOGLE_PHOTOS_API_URL}/albums",
            payload={"album": {"title": title}},
            headers={"Content-Type": "application/json"},
            expected=(200, 201),
        )
        if not isinstance(data, dict):
            raise TypeError(f"Google Photos album create returned {type(data).__name__}, expected object")
        album_id = str(data.get("id") or "").strip()
        if not album_id:
            _exit_runtime(f"Created Google Photos album {title!r} did not include an id")
        return data

    def list_albums(self) -> list[dict[str, Any]]:
        albums: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params = {"pageSize": "50"}
            if page_token:
                params["pageToken"] = page_token
            data = self.request("GET", f"{GOOGLE_PHOTOS_API_URL}/albums?{urllib.parse.urlencode(params)}")
            if not isinstance(data, dict):
                raise TypeError(f"Google Photos albums list returned {type(data).__name__}, expected object")
            page_albums = data.get("albums", [])
            if not isinstance(page_albums, list):
                raise TypeError("Google Photos albums list albums field was not a list")
            albums.extend(album for album in page_albums if isinstance(album, dict))
            page_token = str(data.get("nextPageToken") or "").strip()
            if not page_token:
                return albums

    def list_album_media_items(self, album_id: str) -> list[ExistingMediaItem]:
        media_items: list[ExistingMediaItem] = []
        page_token = ""
        while True:
            payload = {"albumId": album_id, "pageSize": 100}
            if page_token:
                payload["pageToken"] = page_token
            data = self.request(
                "POST",
                f"{GOOGLE_PHOTOS_API_URL}/mediaItems:search",
                payload=payload,
                headers={"Content-Type": "application/json"},
            )
            if not isinstance(data, dict):
                raise TypeError(f"Google Photos mediaItems search returned {type(data).__name__}, expected object")
            page_media = data.get("mediaItems", [])
            if not isinstance(page_media, list):
                raise TypeError("Google Photos mediaItems search mediaItems field was not a list")
            for media_item in page_media:
                if not isinstance(media_item, dict):
                    continue
                media_id = str(media_item.get("id") or "").strip()
                filename = str(media_item.get("filename") or "").strip()
                if media_id and filename:
                    media_items.append(ExistingMediaItem(id=media_id, filename=filename))
            page_token = str(data.get("nextPageToken") or "").strip()
            if not page_token:
                return media_items

    def upload_media(self, media_path: Path) -> str:
        content_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
        with media_path.open("rb") as handle:
            data = self.request(
                "POST",
                UPLOADS_URL,
                data=handle,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Goog-Upload-Content-Type": content_type,
                    "X-Goog-Upload-Protocol": "raw",
                },
                context=media_path.name,
            )
        token = str(data or "").strip()
        if not token:
            _exit_runtime(f"Google Photos upload for {media_path} did not return an upload token")
        return token

    def batch_create_media_items(self, album_id: str, items: Sequence[tuple[str, str]]) -> list[str]:
        payload = {
            "albumId": album_id,
            "newMediaItems": [
                {"simpleMediaItem": {"fileName": file_name, "uploadToken": upload_token}}
                for file_name, upload_token in items
            ],
        }
        data = self.request(
            "POST",
            f"{GOOGLE_PHOTOS_API_URL}/mediaItems:batchCreate",
            payload=payload,
            headers={"Content-Type": "application/json"},
        )
        if not isinstance(data, dict):
            raise TypeError(f"Google Photos batchCreate returned {type(data).__name__}, expected object")
        results = data.get("newMediaItemResults", [])
        if not isinstance(results, list):
            raise TypeError("Google Photos batchCreate newMediaItemResults was not a list")
        media_ids: list[str] = []
        for index, result in enumerate(results):
            if not isinstance(result, dict):
                _exit_runtime(f"Google Photos batchCreate result {index} was {type(result).__name__}, expected object")
            status = result.get("status", {})
            code = int(status.get("code", 0) or 0) if isinstance(status, dict) else 0
            if code:
                message = status.get("message", "") if isinstance(status, dict) else ""
                _exit_runtime(f"Google Photos batchCreate failed for item {index}: code={code} message={message}")
            media_item = result.get("mediaItem", {})
            media_id = str(media_item.get("id") or "").strip() if isinstance(media_item, dict) else ""
            if not media_id:
                _exit_runtime(f"Google Photos batchCreate result {index} did not include a mediaItem id")
            media_ids.append(media_id)
        return media_ids

    def batch_add_media_items(self, album_id: str, media_item_ids: Sequence[str]) -> None:
        _ = self.request(
            "POST",
            f"{GOOGLE_PHOTOS_API_URL}/albums/{album_id}:batchAddMediaItems",
            payload={"mediaItemIds": list(media_item_ids)},
            headers={"Content-Type": "application/json"},
            context=f"{len(media_item_ids)} existing item(s)",
        )


class OAuthCallbackServer(http.server.HTTPServer):
    authorization_response: dict[str, str]

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[http.server.BaseHTTPRequestHandler],
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.authorization_response = {}


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    server_version: str = "ImagoGooglePhotosOAuth/1.0"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        server = cast(OAuthCallbackServer, self.server)
        server.authorization_response = {key: values[0] for key, values in params.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        _ = self.wfile.write(b"Google Photos authorization received. You can close this window.")

    @override
    def log_message(self, format: str, *args: Any) -> None:
        _ = (format, args)
        return


def _default_token_cache_path(account_email: str) -> Path:
    account_key = account_email.replace("@", "_").replace(".", "_")
    return Path.home() / ".config" / "imago" / f"google-photos-{account_key}.token.json"


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} contained {type(data).__name__}, expected object")
    return data


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    with contextlib.suppress(ValueError, json.JSONDecodeError):
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded)
        if isinstance(data, dict):
            return data
    return {}


def _load_oauth_client(client_secrets_path: Path) -> dict[str, str]:
    data = _read_json_file(client_secrets_path)
    client_data = data.get("installed") or data.get("web")
    if not isinstance(client_data, dict):
        _exit_runtime(f"{client_secrets_path} must contain an installed or web OAuth client")
    client_id = str(client_data.get("client_id") or "").strip()
    client_secret = str(client_data.get("client_secret") or "").strip()
    auth_uri = str(client_data.get("auth_uri") or GOOGLE_OAUTH_AUTH_URL).strip()
    token_uri = str(client_data.get("token_uri") or GOOGLE_OAUTH_TOKEN_URL).strip()
    if not client_id:
        _exit_runtime(f"{client_secrets_path} did not include client_id")
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "auth_uri": auth_uri or GOOGLE_OAUTH_AUTH_URL,
        "token_uri": token_uri or GOOGLE_OAUTH_TOKEN_URL,
    }


def _post_oauth_token(token_uri: str, payload: dict[str, str]) -> dict[str, Any]:
    response = requests.post(token_uri, data=payload, timeout=60)
    if response.status_code != 200:
        details = response.text.strip() or response.reason
        raise RuntimeError(f"Google OAuth token request failed with HTTP {response.status_code}: {details}")
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"Google OAuth token response was {type(data).__name__}, expected object")
    return data


def _token_email(access_token: str, id_token: str) -> str:
    payload = _decode_jwt_payload(id_token) if id_token else {}
    email = str(payload.get("email") or "").strip()
    if email:
        return email
    response = requests.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
    if response.status_code != 200:
        details = response.text.strip() or response.reason
        raise RuntimeError(f"Google OAuth userinfo request failed with HTTP {response.status_code}: {details}")
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"Google OAuth userinfo response was {type(data).__name__}, expected object")
    return str(data.get("email") or "").strip()


def _credentials_from_token_response(
    response: dict[str, Any],
    *,
    cached_refresh_token: str,
    expected_account: str,
) -> GoogleOAuthCredentials:
    access_token = str(response.get("access_token") or "").strip()
    if not access_token:
        _exit_runtime("Google OAuth token response did not include access_token")
    refresh_token = str(response.get("refresh_token") or cached_refresh_token).strip()
    if not refresh_token:
        _exit_runtime("Google OAuth token response did not include refresh_token; revoke access and authorize again")
    expires_in = int(response.get("expires_in") or 3600)
    email = _token_email(access_token, str(response.get("id_token") or ""))
    if email.lower() != expected_account.lower():
        _exit_runtime(f"Google OAuth returned account {email!r}, expected {expected_account!r}")
    return GoogleOAuthCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + expires_in,
        email=email,
    )


def _refresh_google_oauth_credentials(
    client: dict[str, str],
    token_cache: dict[str, Any],
    expected_account: str,
) -> GoogleOAuthCredentials:
    refresh_token = str(token_cache.get("refresh_token") or "").strip()
    if not refresh_token:
        _exit_runtime("Cached Google OAuth credentials did not include refresh_token")
    response = _post_oauth_token(
        client["token_uri"],
        {
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    response["refresh_token"] = refresh_token
    response["id_token"] = response.get("id_token") or token_cache.get("id_token") or ""
    return _credentials_from_token_response(
        response,
        cached_refresh_token=refresh_token,
        expected_account=expected_account,
    )


def _run_google_oauth_flow(
    client: dict[str, str],
    *,
    account_email: str,
    open_browser: bool,
) -> GoogleOAuthCredentials:
    state = secrets.token_urlsafe(32)
    server = OAuthCallbackServer(("127.0.0.1", 0), OAuthCallbackHandler)
    redirect_uri = f"http://127.0.0.1:{server.server_port}/"
    params = {
        "client_id": client["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(GOOGLE_PHOTOS_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "login_hint": account_email,
        "state": state,
    }
    auth_url = f"{client['auth_uri']}?{urllib.parse.urlencode(params)}"
    if open_browser:
        _ = webbrowser.open(auth_url)
    print(f"Open this Google authorization URL if your browser did not open:\n{auth_url}", file=sys.stderr)
    server.handle_request()
    response = server.authorization_response
    server.server_close()
    if response.get("state") != state:
        _exit_runtime("Google OAuth callback state did not match")
    if response.get("error"):
        _exit_runtime(f"Google OAuth authorization failed: {response['error']}")
    code = str(response.get("code") or "").strip()
    if not code:
        _exit_runtime("Google OAuth callback did not include an authorization code")
    token_response = _post_oauth_token(
        client["token_uri"],
        {
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
    )
    return _credentials_from_token_response(
        token_response,
        cached_refresh_token="",
        expected_account=account_email,
    )


def load_google_photos_credentials(
    *,
    account_email: str,
    client_secrets_path: Path,
    token_cache_path: Path,
    open_browser: bool,
    force_auth: bool,
) -> GoogleOAuthCredentials:
    client = _load_oauth_client(client_secrets_path)
    if token_cache_path.is_file() and not force_auth:
        token_cache = _read_json_file(token_cache_path)
        email = str(token_cache.get("email") or "").strip()
        if email.lower() != account_email.lower():
            _exit_runtime(f"Cached Google OAuth account is {email!r}, expected {account_email!r}")
        access_token = str(token_cache.get("access_token") or "").strip()
        refresh_token = str(token_cache.get("refresh_token") or "").strip()
        expires_at = float(token_cache.get("expires_at") or 0)
        if access_token and refresh_token and expires_at - time.time() > TOKEN_REFRESH_SKEW_SECONDS:
            return GoogleOAuthCredentials(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                email=email,
            )
        credentials = _refresh_google_oauth_credentials(client, token_cache, account_email)
    else:
        credentials = _run_google_oauth_flow(client, account_email=account_email, open_browser=open_browser)

    _write_json_file(
        token_cache_path,
        {
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "expires_at": credentials.expires_at,
            "email": credentials.email,
            "scopes": list(GOOGLE_PHOTOS_SCOPES),
        },
    )
    return credentials


def refresh_cached_google_photos_access_token(
    *,
    account_email: str,
    client_secrets_path: Path,
    token_cache_path: Path,
) -> str:
    client = _load_oauth_client(client_secrets_path)
    token_cache = _read_json_file(token_cache_path)
    credentials = _refresh_google_oauth_credentials(client, token_cache, account_email)
    _write_json_file(
        token_cache_path,
        {
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "expires_at": credentials.expires_at,
            "email": credentials.email,
            "scopes": list(GOOGLE_PHOTOS_SCOPES),
        },
    )
    return credentials.access_token


def _google_album_name(album_name: str) -> str:
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
    grouped: dict[str, list[Path]] = {}
    for path in sorted(photos_root.iterdir()):
        if not path.is_dir():
            continue
        suffix = album_dir_suffix(path)
        if suffix not in ALBUM_DIR_SUFFIX_ORDER:
            continue
        grouped.setdefault(album_dir_base_name(path), []).append(path)

    albums: list[LocalAlbum] = []
    for album_name, directories in sorted(grouped.items()):
        ordered_dirs = tuple(sorted(directories, key=lambda path: (ALBUM_DIR_SUFFIX_ORDER[album_dir_suffix(path)], path.name)))
        files = tuple(file_path for directory in ordered_dirs for file_path in _media_files(directory, extension_set))
        albums.append(LocalAlbum(name=album_name, directories=ordered_dirs, files=files))
    return albums


def _assert_staging_outside_source(photos_root: Path, staging_root: Path) -> None:
    source = photos_root.resolve(strict=False)
    target = staging_root.resolve(strict=False)
    if target == source or source in target.parents:
        _exit_runtime(f"Staging root must not be inside the read-only Photo Albums source tree: {staging_root}")


def _staged_path(source_path: Path, staging_root: Path, album: LocalAlbum) -> Path:
    return staging_root / album.name / source_path.with_suffix(".jpg").name


def _save_jpeg(source_path: Path, staged_path: Path) -> None:
    allow_large_pillow_images(Image)
    with Image.open(source_path) as image:
        transposed = ImageOps.exif_transpose(image)
        if transposed.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", transposed.size, "white")
            background.paste(transposed, mask=transposed.getchannel("A"))
            output = background
        else:
            output = transposed.convert("RGB")
        output.save(staged_path, format="JPEG", quality=JPEG_QUALITY)


def _ensure_staged_path_writable(staged_path: Path) -> None:
    if staged_path.exists():
        staged_path.chmod(staged_path.stat().st_mode | 0o200)


def _copy_media_for_google_photos(source_path: Path, staged_path: Path) -> None:
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_staged_path_writable(staged_path)
    if album_dir_suffix(source_path.parent) == ALBUM_DIR_SUFFIX_PAGES or source_path.suffix.lower() != ".jpg":
        _save_jpeg(source_path, staged_path)
        shutil.copystat(source_path, staged_path)
        _ensure_staged_path_writable(staged_path)
        return
    shutil.copy2(source_path, staged_path)
    _ensure_staged_path_writable(staged_path)


def _embed_xmp_sidecar(source_path: Path, staged_path: Path) -> bool:
    sidecar_path = source_path.with_suffix(".xmp")
    if not sidecar_path.is_file():
        return False
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-TagsFromFile",
            str(sidecar_path),
            "-all:all",
            str(staged_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_album_copy(staging_root: Path, album: LocalAlbum) -> list[PreparedMedia]:
    prepared: list[PreparedMedia] = []
    seen: set[Path] = set()
    for source_path in album.files:
        staged_path = _staged_path(source_path, staging_root, album)
        if staged_path in seen:
            _exit_runtime(f"Multiple source files would write the same Google Photos copy: {staged_path}")
        seen.add(staged_path)
        _copy_media_for_google_photos(source_path, staged_path)
        _embed_xmp_sidecar(source_path, staged_path)
        prepared.append(
            PreparedMedia(
                source_path=source_path,
                staged_path=staged_path,
                upload_name=staged_path.name,
                sha256=_file_sha256(staged_path),
            )
        )
    return prepared


def _chunked(items: Sequence[PreparedMedia], size: int) -> Iterable[Sequence[PreparedMedia]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _chunked_media_item_ids(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _default_resume_state_path(staging_root: Path) -> Path:
    return staging_root / ".google-photos-upload-state.json"


def _new_upload_state(account_email: str) -> dict[str, Any]:
    return {
        "version": 1,
        "account_email": account_email,
        "albums": {},
    }


def load_upload_state(path: Path, account_email: str) -> dict[str, Any]:
    if not path.is_file():
        return _new_upload_state(account_email)
    data = _read_json_file(path)
    state_account = str(data.get("account_email") or "").strip()
    if state_account.lower() != account_email.lower():
        _exit_runtime(f"Upload state account is {state_account!r}, expected {account_email!r}")
    albums = data.setdefault("albums", {})
    if not isinstance(albums, dict):
        raise TypeError(f"{path} albums field was {type(albums).__name__}, expected object")
    return data


def save_upload_state(path: Path, state: dict[str, Any]) -> None:
    _write_json_file(path, state)


def _album_state(state: dict[str, Any], album: LocalAlbum) -> dict[str, Any]:
    albums = state.setdefault("albums", {})
    if not isinstance(albums, dict):
        raise TypeError("Upload state albums field was not an object")
    album_state = albums.setdefault(
        album.name,
        {
            "title": _google_album_name(album.name),
            "album_id": "",
            "items": {},
        },
    )
    if not isinstance(album_state, dict):
        raise TypeError(f"Upload state album {album.name} was not an object")
    items = album_state.setdefault("items", {})
    if not isinstance(items, dict):
        raise TypeError(f"Upload state album {album.name} items field was not an object")
    return album_state


def _prepared_state_key(item: PreparedMedia, staging_root: Path) -> str:
    return item.staged_path.relative_to(staging_root).as_posix()


def _matching_item_state(album_state: dict[str, Any], state_key: str, item: PreparedMedia) -> dict[str, Any] | None:
    items = album_state["items"]
    item_state = items.get(state_key)
    if not isinstance(item_state, dict):
        return None
    if str(item_state.get("sha256") or "") != item.sha256:
        return None
    return item_state


def _find_existing_album_id(client: GooglePhotosAlbumClient, title: str) -> str:
    try:
        albums = client.list_albums()
    except RuntimeError as exc:
        if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(exc):
            return ""
        raise
    matches = [str(album.get("id") or "").strip() for album in albums if album.get("title") == title]
    matches = [album_id for album_id in matches if album_id]
    if len(matches) > 1:
        _exit_runtime(f"Found multiple Google Photos albums titled {title!r}; cannot safely resume")
    return matches[0] if matches else ""


def _ensure_google_album_id(
    client: GooglePhotosAlbumClient,
    album: LocalAlbum,
    album_state: dict[str, Any],
    state_path: Path,
    state: dict[str, Any],
) -> str:
    title = _google_album_name(album.name)
    album_id = str(album_state.get("album_id") or "").strip()
    if album_id:
        return album_id
    album_id = _find_existing_album_id(client, title)
    if not album_id:
        album_id = str(client.create_album(title)["id"])
    album_state["title"] = title
    album_state["album_id"] = album_id
    save_upload_state(state_path, state)
    return album_id


def _ensure_google_album_title(
    client: GooglePhotosAlbumClient,
    title: str,
    album_state: dict[str, Any],
    state_path: Path,
    state: dict[str, Any],
) -> str:
    album_id = str(album_state.get("album_id") or "").strip()
    if album_id:
        return album_id
    album_id = _find_existing_album_id(client, title)
    if not album_id:
        album_id = str(client.create_album(title)["id"])
    album_state["title"] = title
    album_state["album_id"] = album_id
    save_upload_state(state_path, state)
    return album_id


def _mark_existing_album_media(
    client: GooglePhotosAlbumClient,
    album_id: str,
    album_state: dict[str, Any],
    prepared_media: Sequence[PreparedMedia],
    staging_root: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    try:
        existing_media = client.list_album_media_items(album_id)
    except RuntimeError as exc:
        if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(exc):
            return
        raise
    existing_by_filename = {item.filename: item.id for item in existing_media}
    changed = False
    for item in prepared_media:
        media_id = existing_by_filename.get(item.upload_name, "")
        if not media_id:
            continue
        state_key = _prepared_state_key(item, staging_root)
        items = album_state["items"]
        items[state_key] = {
            "filename": item.upload_name,
            "sha256": item.sha256,
            "media_item_id": media_id,
        }
        changed = True
    if changed:
        save_upload_state(state_path, state)


def _fresh_upload_token(item_state: dict[str, Any]) -> str:
    token = str(item_state.get("upload_token") or "").strip()
    created_at = float(item_state.get("upload_token_created_at") or 0)
    if token and time.time() - created_at < UPLOAD_TOKEN_TTL_SECONDS:
        return token
    return ""


def _existing_album_state(upload_state: dict[str, Any], album_name: str) -> dict[str, Any] | None:
    albums_state = upload_state.get("albums")
    if not isinstance(albums_state, dict):
        return None
    album_state = albums_state.get(album_name)
    return album_state if isinstance(album_state, dict) else None


def _aggregate_albums_state(upload_state: dict[str, Any]) -> dict[str, Any]:
    aggregate_albums = upload_state.setdefault("aggregate_albums", {})
    if not isinstance(aggregate_albums, dict):
        raise TypeError("Upload state aggregate_albums field was not an object")
    return aggregate_albums


def _aggregate_album_state(upload_state: dict[str, Any], title: str) -> dict[str, Any]:
    aggregate_albums = _aggregate_albums_state(upload_state)
    album_state = aggregate_albums.setdefault(title, {"title": title, "album_id": "", "media_item_ids": []})
    if not isinstance(album_state, dict):
        raise TypeError(f"Upload state aggregate album {title} was not an object")
    media_item_ids = album_state.setdefault("media_item_ids", [])
    if not isinstance(media_item_ids, list):
        raise TypeError(f"Upload state aggregate album {title} media_item_ids field was not a list")
    return album_state


def _uploaded_media_ids_for_album_names(upload_state: dict[str, Any], album_names: set[str]) -> list[str]:
    albums_state = upload_state.get("albums")
    if not isinstance(albums_state, dict):
        return []
    media_ids: list[str] = []
    seen: set[str] = set()
    for album_name in sorted(album_names):
        album_state = albums_state.get(album_name)
        if not isinstance(album_state, dict):
            continue
        items = album_state.get("items")
        if not isinstance(items, dict):
            continue
        for state_key in sorted(items):
            item_state = items[state_key]
            media_id = str(item_state.get("media_item_id") or "").strip() if isinstance(item_state, dict) else ""
            if media_id and media_id not in seen:
                media_ids.append(media_id)
                seen.add(media_id)
    return media_ids


def _google_media_ids_for_album_names(
    client: GooglePhotosAlbumClient,
    upload_state: dict[str, Any],
    album_names: set[str],
    *,
    verbose: bool = False,
) -> list[str]:
    albums_state = upload_state.get("albums")
    if not isinstance(albums_state, dict):
        return []
    media_ids: list[str] = []
    seen: set[str] = set()
    for album_name in sorted(album_names):
        album_state = albums_state.get(album_name)
        if not isinstance(album_state, dict):
            continue
        album_title = str(album_state.get("title") or _google_album_name(album_name)).strip()
        album_id = str(album_state.get("album_id") or "").strip()
        if not album_id:
            _vlog(verbose, f"{album_title}: skipping aggregate source with no recorded album_id")
            continue
        source_media = client.list_album_media_items(album_id)
        _vlog(verbose, f"{album_title}: found {len(source_media)} item(s) in Google Photos source album")
        for item in source_media:
            if item.id not in seen:
                media_ids.append(item.id)
                seen.add(item.id)
    return media_ids


def _aggregate_album_names(upload_state: dict[str, Any]) -> dict[str, set[str]]:
    albums_state = upload_state.get("albums")
    if not isinstance(albums_state, dict):
        return {"Family": set(), "Travel": set()}
    source_album_names = {name for name, album_state in albums_state.items() if isinstance(name, str) and isinstance(album_state, dict)}
    family_album_names = {name for name in source_album_names if name.startswith("Family")}
    return {
        "Family": family_album_names,
        "Travel": source_album_names - family_album_names,
    }


def _is_invalid_media_item_id_error(exc: RuntimeError) -> bool:
    return "Request contains an invalid media item id" in str(exc)


def _add_media_items_with_invalid_id_isolation(
    client: GooglePhotosAlbumClient,
    album_id: str,
    media_item_ids: Sequence[str],
    *,
    title: str,
) -> tuple[list[str], list[str]]:
    try:
        client.batch_add_media_items(album_id, media_item_ids)
    except RuntimeError as exc:
        if not _is_invalid_media_item_id_error(exc):
            raise
        if len(media_item_ids) == 1:
            invalid_id = media_item_ids[0]
            _log_status(f"{title}: skipping invalid Google Photos media item id {invalid_id}")
            return [], [invalid_id]
        midpoint = len(media_item_ids) // 2
        added_left, invalid_left = _add_media_items_with_invalid_id_isolation(
            client,
            album_id,
            media_item_ids[:midpoint],
            title=title,
        )
        added_right, invalid_right = _add_media_items_with_invalid_id_isolation(
            client,
            album_id,
            media_item_ids[midpoint:],
            title=title,
        )
        return added_left + added_right, invalid_left + invalid_right
    return list(media_item_ids), []


def sync_google_aggregate_album(
    client: GooglePhotosAlbumClient,
    title: str,
    media_item_ids: Sequence[str],
    *,
    state_path: Path,
    state: dict[str, Any],
    verbose: bool = False,
) -> str:
    album_state = _aggregate_album_state(state, title)
    album_id = _ensure_google_album_title(client, title, album_state, state_path, state)
    existing_ids = {item.id for item in client.list_album_media_items(album_id)}
    pending_ids = [media_id for media_id in media_item_ids if media_id not in existing_ids]
    _vlog(
        verbose,
        f"{title}: album_id={album_id}; {len(pending_ids)} item(s) to add, "
        f"{len(media_item_ids) - len(pending_ids)} already present",
    )
    for batch_index, batch in enumerate(_chunked_media_item_ids(pending_ids, BATCH_CREATE_LIMIT), start=1):
        _vlog(verbose, f"{title}: batchAddMediaItems {len(batch)} item(s) (batch {batch_index})")
        added_ids, invalid_ids = _add_media_items_with_invalid_id_isolation(client, album_id, batch, title=title)
        existing_ids.update(added_ids)
        album_state["media_item_ids"] = sorted(existing_ids)
        if invalid_ids:
            aggregate_invalid_ids = album_state.setdefault("invalid_media_item_ids", [])
            if not isinstance(aggregate_invalid_ids, list):
                raise TypeError(f"Upload state aggregate album {title} invalid_media_item_ids field was not a list")
            aggregate_invalid_ids.extend(media_id for media_id in invalid_ids if media_id not in aggregate_invalid_ids)
        save_upload_state(state_path, state)
        sys.stdout.write("." * len(added_ids))
        if invalid_ids:
            sys.stdout.write("!" * len(invalid_ids))
        sys.stdout.flush()
    if pending_ids:
        sys.stdout.write("\n")
    return album_id


def sync_google_aggregate_albums(
    client: GooglePhotosAlbumClient | None,
    *,
    state_path: Path,
    state: dict[str, Any],
    dry_run: bool,
    verbose: bool = False,
) -> None:
    aggregate_names = _aggregate_album_names(state)
    for title in ("Family", "Travel"):
        source_album_names = aggregate_names[title]
        media_item_ids = _uploaded_media_ids_for_album_names(state, source_album_names)
        source_titles = ", ".join(_google_album_name(name) for name in sorted(source_album_names))
        if dry_run:
            sys.stdout.write(
                f"DRY RUN {title}: would include {len(media_item_ids)} item(s) "
                f"from {len(source_album_names)} uploaded album(s)"
            )
            if verbose and source_titles:
                sys.stdout.write(f": {source_titles}")
            sys.stdout.write("\n")
            continue
        if client is None:
            _exit_runtime("Google Photos client is required unless --dry-run or --prepare-only is set")
        _log_status(f"{title}: syncing aggregate album from {len(source_album_names)} uploaded album(s)")
        aggregate_started = time.monotonic()
        media_item_ids = _google_media_ids_for_album_names(client, state, source_album_names, verbose=verbose)
        album_id = sync_google_aggregate_album(
            client,
            title,
            media_item_ids,
            state_path=state_path,
            state=state,
            verbose=verbose,
        )
        sys.stdout.write(
            f"{title}: synced {len(media_item_ids)} item(s) ({album_id}) in {time.monotonic() - aggregate_started:.1f}s\n"
        )


def _album_is_complete(album: LocalAlbum, album_state: dict[str, Any] | None) -> bool:
    if album_state is None:
        return False
    album_id = str(album_state.get("album_id") or "").strip()
    if not album_id:
        return False
    items = album_state.get("items")
    if not isinstance(items, dict):
        return False
    if len(items) != len(album.files):
        return False
    return all(
        isinstance(item_state, dict) and str(item_state.get("media_item_id") or "").strip()
        for item_state in items.values()
    )


def _record_uploaded_media_batch(
    media_ids: Sequence[str],
    batch_states: Sequence[tuple[str, PreparedMedia, dict[str, Any]]],
    state_path: Path,
    state: dict[str, Any],
) -> None:
    for media_id, (_state_key, item, item_state) in zip(media_ids, batch_states, strict=True):
        item_state["filename"] = item.upload_name
        item_state["sha256"] = item.sha256
        item_state["media_item_id"] = media_id
        item_state.pop("upload_token", None)
        item_state.pop("upload_token_created_at", None)
        sys.stdout.write(".")
        sys.stdout.flush()
    save_upload_state(state_path, state)


def sync_google_album_with_media(
    client: GooglePhotosAlbumClient,
    album: LocalAlbum,
    prepared_media: Sequence[PreparedMedia],
    *,
    staging_root: Path,
    state_path: Path,
    state: dict[str, Any],
    dry_run: bool,
    verbose: bool = False,
) -> str | None:
    album_title = _google_album_name(album.name)
    if dry_run:
        sys.stdout.write(f"DRY RUN {album_title}: would create with {len(prepared_media)} item(s)\n")
        return None
    album_state = _album_state(state, album)
    album_id = _ensure_google_album_id(client, album, album_state, state_path, state)
    _mark_existing_album_media(client, album_id, album_state, prepared_media, staging_root, state_path, state)
    pending_media: list[PreparedMedia] = []
    for item in prepared_media:
        state_key = _prepared_state_key(item, staging_root)
        item_state = _matching_item_state(album_state, state_key, item)
        if item_state and str(item_state.get("media_item_id") or "").strip():
            continue
        pending_media.append(item)

    _vlog(
        verbose,
        f"{album_title}: album_id={album_id}; {len(pending_media)} item(s) to upload, "
        f"{len(prepared_media) - len(pending_media)} already present",
    )
    uploaded = 0
    for batch_index, batch in enumerate(_chunked(pending_media, BATCH_CREATE_LIMIT), start=1):
        upload_items: list[tuple[str, str]] = []
        batch_states: list[tuple[str, PreparedMedia, dict[str, Any]]] = []
        for item in batch:
            state_key = _prepared_state_key(item, staging_root)
            items = album_state["items"]
            item_state = items.setdefault(state_key, {})
            if not isinstance(item_state, dict):
                raise TypeError(f"Upload state item {state_key} was not an object")
            if str(item_state.get("sha256") or "") != item.sha256:
                item_state.clear()
                item_state.update({"filename": item.upload_name, "sha256": item.sha256})
            upload_token = _fresh_upload_token(item_state)
            if not upload_token:
                uploaded += 1
                _vlog(
                    verbose,
                    f"{album_title}: uploading {uploaded}/{len(pending_media)} "
                    f"{item.upload_name} ({_format_size(item.staged_path.stat().st_size)})",
                )
                upload_token = client.upload_media(item.staged_path)
                item_state["upload_token"] = upload_token
                item_state["upload_token_created_at"] = time.time()
                save_upload_state(state_path, state)
            upload_items.append((item.upload_name, upload_token))
            batch_states.append((state_key, item, item_state))

        _vlog(verbose, f"{album_title}: batchCreate {len(upload_items)} item(s) (batch {batch_index})")
        media_ids = client.batch_create_media_items(album_id, upload_items)
        _record_uploaded_media_batch(media_ids, batch_states, state_path, state)
    if pending_media:
        sys.stdout.write("\n")
    return album_id


def create_google_album_with_media(
    client: GooglePhotosAlbumClient,
    album: LocalAlbum,
    prepared_media: Sequence[PreparedMedia],
    *,
    dry_run: bool,
) -> str | None:
    album_title = _google_album_name(album.name)
    if dry_run:
        sys.stdout.write(f"DRY RUN {album_title}: would create with {len(prepared_media)} item(s)\n")
        return None
    album_id = str(client.create_album(album_title)["id"])
    for batch in _chunked(list(prepared_media), BATCH_CREATE_LIMIT):
        upload_items = [(item.upload_name, client.upload_media(item.staged_path)) for item in batch]
        client.batch_create_media_items(album_id, upload_items)
    return album_id


def sync_photo_albums(
    client: GooglePhotosAlbumClient | None,
    photos_root: Path,
    staging_root: Path,
    *,
    album_filter: set[str] | None = None,
    dry_run: bool = False,
    prepare_only: bool = False,
    resume_state_path: Path | None = None,
    account_email: str = DEFAULT_GOOGLE_ACCOUNT,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    verbose: bool = False,
    sync_aggregate_albums_enabled: bool = False,
) -> int:
    _assert_staging_outside_source(photos_root, staging_root)
    if prepare_only and sync_aggregate_albums_enabled:
        _exit_runtime("--sync-aggregate-albums cannot be combined with --prepare-only")
    albums = discover_local_albums(photos_root, extensions)
    if album_filter:
        albums = [album for album in albums if album.name in album_filter]
    if not albums:
        _exit_runtime("No matching *_Pages or *_Photos album directories found")

    state_path = resume_state_path or _default_resume_state_path(staging_root)
    upload_state = (
        _new_upload_state(account_email)
        if (dry_run and not sync_aggregate_albums_enabled) or prepare_only
        else load_upload_state(state_path, account_email)
    )
    for index, album in enumerate(albums, start=1):
        album_state = None if dry_run or prepare_only else _existing_album_state(upload_state, album.name)
        if _album_is_complete(album, album_state):
            sys.stdout.write(f"[{index}/{len(albums)}] {_google_album_name(album.name)}: already synced ({len(album.files)} item(s))\n")
            continue
        album_title = _google_album_name(album.name)
        sys.stdout.write(f"[{index}/{len(albums)}] Preparing {album_title} ({len(album.files)} file(s))\n")
        sys.stdout.flush()
        prepare_started = time.monotonic()
        if dry_run:
            prepared_media: list[PreparedMedia] = []
        else:
            _vlog(verbose, f"[{index}/{len(albums)}] {album_title}: staging {len(album.files)} file(s)")
            prepared_media = prepare_album_copy(staging_root, album)
            _vlog(
                verbose,
                f"[{index}/{len(albums)}] {album_title}: "
                f"staged {len(prepared_media)} item(s) in {time.monotonic() - prepare_started:.1f}s",
            )
        if prepare_only or dry_run:
            sys.stdout.write(f"{album_title}: prepared {len(prepared_media)} item(s)\n")
            continue
        if client is None:
            _exit_runtime("Google Photos client is required unless --dry-run or --prepare-only is set")
        sync_started = time.monotonic()
        album_id = sync_google_album_with_media(
            client,
            album,
            prepared_media,
            staging_root=staging_root,
            state_path=state_path,
            state=upload_state,
            dry_run=False,
            verbose=verbose,
        )
        sys.stdout.write(
            f"{album_title}: synced {len(prepared_media)} item(s) ({album_id}) in {time.monotonic() - sync_started:.1f}s\n"
        )
    if sync_aggregate_albums_enabled:
        sync_google_aggregate_albums(
            client,
            state_path=state_path,
            state=upload_state,
            dry_run=dry_run,
            verbose=verbose,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Google Photos albums from Photo Albums *_Pages and *_Photos directories."
    )
    parser.add_argument("--photos-root", default=str(get_photo_albums_dir()), help="Read-only Photo Albums root directory.")
    parser.add_argument(
        "--staging-root",
        default=str(DEFAULT_STAGING_ROOT),
        help="Directory where per-album Google Photos JPEG copies with embedded XMP are written.",
    )
    parser.add_argument("--access-token", default=os.environ.get(GOOGLE_PHOTOS_ACCESS_TOKEN_ENV, ""), help="OAuth access token.")
    parser.add_argument(
        "--google-account",
        default=DEFAULT_GOOGLE_ACCOUNT,
        help="Google account email to authorize and sync into.",
    )
    parser.add_argument(
        "--client-secrets",
        default=os.environ.get(GOOGLE_PHOTOS_CLIENT_SECRETS_ENV, str(DEFAULT_CLIENT_SECRETS_PATH)),
        help="Google OAuth desktop client JSON path.",
    )
    parser.add_argument(
        "--oauth-token-cache",
        default=os.environ.get(GOOGLE_PHOTOS_TOKEN_CACHE_ENV, ""),
        help="Cached OAuth token JSON path.",
    )
    parser.add_argument(
        "--resume-state",
        default="",
        help="Upload resume state JSON path. Defaults to .google-photos-upload-state.json in the staging root.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Print the OAuth URL instead of opening a browser.")
    parser.add_argument("--force-auth", action="store_true", help="Ignore cached OAuth credentials and authorize again.")
    parser.add_argument("--album", action="append", default=[], help="Base album name to process; repeat to process several.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log per-request timing, upload progress, and retry/backoff details to stderr.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover matching albums without copying or uploading.")
    parser.add_argument("--prepare-only", action="store_true", help="Create the Google Photos staging copy without uploading.")
    parser.add_argument(
        "--sync-aggregate-albums",
        action="store_true",
        help="Create or update aggregate Google Photos albums: Family from Family* uploads, Travel from other uploads.",
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
    photos_root = Path(args.photos_root).expanduser()
    staging_root = Path(args.staging_root).expanduser()
    extensions = tuple(args.extension) if args.extension else DEFAULT_EXTENSIONS
    access_token = str(args.access_token or "").strip()
    google_account = str(args.google_account or DEFAULT_GOOGLE_ACCOUNT).strip()
    client_secrets_path = Path(str(args.client_secrets)).expanduser()
    token_cache_path = (
        Path(str(args.oauth_token_cache)).expanduser()
        if str(args.oauth_token_cache or "").strip()
        else _default_token_cache_path(google_account)
    )
    resume_state_path = (
        Path(str(args.resume_state)).expanduser() if str(args.resume_state or "").strip() else _default_resume_state_path(staging_root)
    )
    dry_run = bool(args.dry_run)
    prepare_only = bool(args.prepare_only)
    verbose = bool(args.verbose)
    refresh_access_token: Callable[[], str] | None = None
    if not access_token and not (dry_run or prepare_only):
        credentials = load_google_photos_credentials(
            account_email=google_account,
            client_secrets_path=client_secrets_path,
            token_cache_path=token_cache_path,
            open_browser=not bool(args.no_browser),
            force_auth=bool(args.force_auth),
        )
        access_token = credentials.access_token

        def refresh_access_token_from_cache() -> str:
            return refresh_cached_google_photos_access_token(
                account_email=google_account,
                client_secrets_path=client_secrets_path,
                token_cache_path=token_cache_path,
            )

        refresh_access_token = refresh_access_token_from_cache
    client = (
        None
        if not access_token
        else GooglePhotosClient(access_token, refresh_access_token=refresh_access_token, verbose=verbose)
    )
    return sync_photo_albums(
        client,
        photos_root,
        staging_root,
        album_filter=set(args.album) if args.album else None,
        dry_run=dry_run,
        prepare_only=prepare_only,
        resume_state_path=resume_state_path,
        account_email=google_account,
        extensions=extensions,
        verbose=verbose,
        sync_aggregate_albums_enabled=bool(args.sync_aggregate_albums),
    )


if __name__ == "__main__":
    sys.exit(main())
