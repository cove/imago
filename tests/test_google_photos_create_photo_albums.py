from collections.abc import Sequence
from pathlib import Path
from typing import cast
from unittest import mock

import pytest
import requests
from PIL import Image

from photoalbums.scripts.create_google_photos_albums import (
    ExistingMediaItem,
    GooglePhotosClient,
    LocalAlbum,
    PreparedMedia,
    create_google_album_with_media,
    discover_local_albums,
    prepare_album_copy,
    sync_google_aggregate_album,
    sync_google_aggregate_albums,
    sync_google_album_with_media,
    sync_photo_albums,
)

INVALID_MEDIA_ITEM_ID_MESSAGE = "Request contains an invalid media item id."


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2, 2), "white")
    image.save(path)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2, 2), "blue")
    image.save(path)


def test_discovers_base_albums_from_pages_and_photos_only(tmp_path: Path) -> None:
    root = tmp_path / "Photo Albums"
    (root / "Family_1975_B01_Archive").mkdir(parents=True)
    (root / "Family_1975_B01_Pages").mkdir()
    (root / "Family_1975_B01_Photos").mkdir()
    _write_jpeg(root / "Family_1975_B01_Archive" / "Family_1975_B01_P01_S01.tif")
    _write_jpeg(root / "Family_1975_B01_Pages" / "Family_1975_B01_P01_V.jpg")
    _write_jpeg(root / "Family_1975_B01_Photos" / "Family_1975_B01_P01_D01-00_V.jpg")

    albums = discover_local_albums(root)

    assert [album.name for album in albums] == ["Family_1975_B01"]
    assert [path.name for path in albums[0].directories] == ["Family_1975_B01_Pages", "Family_1975_B01_Photos"]
    assert [path.name for path in albums[0].files] == [
        "Family_1975_B01_P01_V.jpg",
        "Family_1975_B01_P01_D01-00_V.jpg",
    ]


def test_prepare_album_copy_converts_pages_to_jpg_and_embeds_xmp_on_copy(tmp_path: Path) -> None:
    root = tmp_path / "Photo Albums"
    pages_dir = root / "Family_1975_B01_Pages"
    photos_dir = root / "Family_1975_B01_Photos"
    page = pages_dir / "Family_1975_B01_P01_V.png"
    photo = photos_dir / "Family_1975_B01_P01_D01-00_V.jpg"
    _write_png(page)
    _write_jpeg(photo)
    _ = page.with_suffix(".xmp").write_text("<xmp>page</xmp>", encoding="utf-8")
    _ = photo.with_suffix(".xmp").write_text("<xmp>photo</xmp>", encoding="utf-8")
    staging_root = tmp_path / "googlephotos"
    album = LocalAlbum("Family_1975_B01", (pages_dir, photos_dir), (page, photo))

    with mock.patch("photoalbums.scripts.create_google_photos_albums.subprocess.run") as run_mock:
        prepared = prepare_album_copy(staging_root, album)

    assert [item.staged_path.relative_to(staging_root) for item in prepared] == [
        Path("Family_1975_B01/Family_1975_B01_P01_V.jpg"),
        Path("Family_1975_B01/Family_1975_B01_P01_D01-00_V.jpg"),
    ]
    assert prepared[0].staged_path.suffix == ".jpg"
    assert page.read_bytes() != prepared[0].staged_path.read_bytes()
    assert photo.read_bytes() == prepared[1].staged_path.read_bytes()
    assert run_mock.call_args_list == [
        mock.call(
            [
                "exiftool",
                "-overwrite_original",
                "-TagsFromFile",
                str(page.with_suffix(".xmp")),
                "-all:all",
                str(prepared[0].staged_path),
            ],
            stdout=mock.ANY,
            stderr=mock.ANY,
            check=True,
        ),
        mock.call(
            [
                "exiftool",
                "-overwrite_original",
                "-TagsFromFile",
                str(photo.with_suffix(".xmp")),
                "-all:all",
                str(prepared[1].staged_path),
            ],
            stdout=mock.ANY,
            stderr=mock.ANY,
            check=True,
        ),
    ]


def test_sync_rejects_staging_inside_source_tree(tmp_path: Path) -> None:
    root = tmp_path / "Photo Albums"
    (root / "Family_1975_B01_Pages").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Staging root must not be inside"):
        _ = sync_photo_albums(None, root, root / "googlephotos", dry_run=True)


class FakeGooglePhotosClient:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.uploaded: list[Path] = []
        self.batches: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def create_album(self, title: str) -> dict[str, str]:
        self.created.append(title)
        return {"id": f"album-{title}"}

    def list_albums(self) -> list[dict[str, str]]:
        return []

    def list_album_media_items(self, album_id: str) -> list[ExistingMediaItem]:
        _ = album_id
        return []

    def upload_media(self, media_path: Path) -> str:
        self.uploaded.append(media_path)
        return f"token-{media_path.name}"

    def batch_create_media_items(self, album_id: str, items: Sequence[tuple[str, str]]) -> list[str]:
        self.batches.append((album_id, tuple(items)))
        return [f"media-{index}" for index, _ in enumerate(items)]

    def batch_add_media_items(self, album_id: str, media_item_ids: Sequence[str]) -> None:
        _ = (album_id, media_item_ids)


def test_create_google_album_with_media_creates_album_uploads_then_batch_creates(tmp_path: Path) -> None:
    staged = tmp_path / "Family_1975_B01_Pages" / "Family_1975_B01_P01_V.jpg"
    _write_jpeg(staged)
    client = FakeGooglePhotosClient()
    album = LocalAlbum("Family_1975_B01", (), ())
    prepared = [PreparedMedia(source_path=staged, staged_path=staged, upload_name=staged.name)]

    album_id = create_google_album_with_media(client, album, prepared, dry_run=False)

    assert album_id == "album-Family 1975 B01"
    assert client.created == ["Family 1975 B01"]
    assert client.uploaded == [staged]
    assert client.batches == [
        ("album-Family 1975 B01", (("Family_1975_B01_P01_V.jpg", "token-Family_1975_B01_P01_V.jpg"),))
    ]


class ResumeFakeGooglePhotosClient:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.uploaded: list[Path] = []
        self.batches: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def create_album(self, title: str) -> dict[str, str]:
        self.created.append(title)
        return {"id": f"album-{title}"}

    def list_albums(self) -> list[dict[str, str]]:
        return [{"id": "existing-album", "title": "Family 1975 B01"}]

    def list_album_media_items(self, album_id: str) -> list[ExistingMediaItem]:
        assert album_id == "existing-album"
        return [ExistingMediaItem(id="existing-media-1", filename="Family_1975_B01_P01_V.jpg")]

    def upload_media(self, media_path: Path) -> str:
        self.uploaded.append(media_path)
        return f"token-{media_path.name}"

    def batch_create_media_items(self, album_id: str, items: Sequence[tuple[str, str]]) -> list[str]:
        self.batches.append((album_id, tuple(items)))
        return [f"media-{file_name}" for file_name, _token in items]

    def batch_add_media_items(self, album_id: str, media_item_ids: Sequence[str]) -> None:
        _ = (album_id, media_item_ids)


def test_sync_google_album_with_media_resumes_existing_album_and_media(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    staging_root = tmp_path / "googlephotos"
    first = staging_root / "Family_1975_B01" / "Family_1975_B01_P01_V.jpg"
    second = staging_root / "Family_1975_B01" / "Family_1975_B01_P01_D01-00_V.jpg"
    _write_jpeg(first)
    _write_jpeg(second)
    prepared = [
        PreparedMedia(source_path=first, staged_path=first, upload_name=first.name, sha256="sha-first"),
        PreparedMedia(source_path=second, staged_path=second, upload_name=second.name, sha256="sha-second"),
    ]
    client = ResumeFakeGooglePhotosClient()
    album = LocalAlbum("Family_1975_B01", (), ())
    state_path = tmp_path / "state.json"
    state: dict[str, object] = {"version": 1, "account_email": "dolorescordell@gmail.com", "albums": {}}

    album_id = sync_google_album_with_media(
        client,
        album,
        prepared,
        staging_root=staging_root,
        state_path=state_path,
        state=state,
        dry_run=False,
    )

    assert album_id == "existing-album"
    assert client.created == []
    assert client.uploaded == [second]
    assert client.batches == [
        ("existing-album", (("Family_1975_B01_P01_D01-00_V.jpg", "token-Family_1975_B01_P01_D01-00_V.jpg"),))
    ]
    albums = cast(dict[str, dict[str, object]], state["albums"])
    album_state = albums["Family_1975_B01"]
    items = cast(dict[str, dict[str, object]], album_state["items"])
    assert album_state["album_id"] == "existing-album"
    assert items["Family_1975_B01/Family_1975_B01_P01_V.jpg"]["media_item_id"] == "existing-media-1"
    assert (
        items["Family_1975_B01/Family_1975_B01_P01_D01-00_V.jpg"]["media_item_id"]
        == "media-Family_1975_B01_P01_D01-00_V.jpg"
    )
    assert capsys.readouterr().out == ".\n"


class AggregateFakeGooglePhotosClient:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.added: list[tuple[str, tuple[str, ...]]] = []

    def create_album(self, title: str) -> dict[str, str]:
        self.created.append(title)
        return {"id": f"album-{title}"}

    def list_albums(self) -> list[dict[str, str]]:
        return []

    def list_album_media_items(self, album_id: str) -> list[ExistingMediaItem]:
        if album_id == "album-Family":
            return [ExistingMediaItem(id="family-existing", filename="family-existing.jpg")]
        if album_id == "source-family":
            return [
                ExistingMediaItem(id="family-new", filename="Family_1975_B01_P01_D01-00_V.jpg"),
                ExistingMediaItem(id="family-existing", filename="Family_1975_B01_P01_V.jpg"),
            ]
        if album_id == "source-travel":
            return [ExistingMediaItem(id="travel-new", filename="Egypt_1975_B00_P01_V.jpg")]
        return []

    def upload_media(self, media_path: Path) -> str:
        _ = media_path
        raise AssertionError

    def batch_create_media_items(self, album_id: str, items: Sequence[tuple[str, str]]) -> list[str]:
        _ = (album_id, items)
        raise AssertionError

    def batch_add_media_items(self, album_id: str, media_item_ids: Sequence[str]) -> None:
        self.added.append((album_id, tuple(media_item_ids)))


def test_sync_google_aggregate_albums_adds_uploaded_family_and_travel_media(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state: dict[str, object] = {
        "version": 1,
        "account_email": "dolorescordell@gmail.com",
        "albums": {
            "Egypt_1975_B00": {
                "album_id": "source-travel",
                "items": {
                    "Egypt_1975_B00/Egypt_1975_B00_P01_V.jpg": {"media_item_id": "travel-new"},
                }
            },
            "Family_1975_B01": {
                "album_id": "source-family",
                "items": {
                    "Family_1975_B01/Family_1975_B01_P01_D01-00_V.jpg": {"media_item_id": "family-new"},
                    "Family_1975_B01/Family_1975_B01_P01_V.jpg": {"media_item_id": "family-existing"},
                }
            },
        },
    }
    client = AggregateFakeGooglePhotosClient()

    sync_google_aggregate_albums(client, state_path=tmp_path / "state.json", state=state, dry_run=False)

    assert client.created == ["Family", "Travel"]
    assert client.added == [
        ("album-Family", ("family-new",)),
        ("album-Travel", ("travel-new",)),
    ]
    aggregate_albums = cast(dict[str, dict[str, object]], state["aggregate_albums"])
    assert aggregate_albums["Family"]["album_id"] == "album-Family"
    assert aggregate_albums["Travel"]["album_id"] == "album-Travel"
    output = capsys.readouterr().out
    assert "Family: synced 2 item(s) (album-Family)" in output
    assert "Travel: synced 1 item(s) (album-Travel)" in output


class InvalidAggregateFakeGooglePhotosClient(AggregateFakeGooglePhotosClient):
    def list_album_media_items(self, album_id: str) -> list[ExistingMediaItem]:
        if album_id == "album-Family":
            return []
        return []

    def batch_add_media_items(self, album_id: str, media_item_ids: Sequence[str]) -> None:
        self.added.append((album_id, tuple(media_item_ids)))
        if "bad-id" in media_item_ids:
            raise RuntimeError(INVALID_MEDIA_ITEM_ID_MESSAGE)


def test_sync_google_aggregate_album_skips_only_invalid_media_item_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state: dict[str, object] = {"version": 1, "account_email": "dolorescordell@gmail.com", "albums": {}}
    client = InvalidAggregateFakeGooglePhotosClient()

    album_id = sync_google_aggregate_album(
        client,
        "Family",
        ["good-1", "bad-id", "good-2"],
        state_path=tmp_path / "state.json",
        state=state,
    )

    assert album_id == "album-Family"
    assert ("album-Family", ("good-1",)) in client.added
    assert ("album-Family", ("good-2",)) in client.added
    aggregate_albums = cast(dict[str, dict[str, object]], state["aggregate_albums"])
    assert aggregate_albums["Family"]["media_item_ids"] == ["good-1", "good-2"]
    assert aggregate_albums["Family"]["invalid_media_item_ids"] == ["bad-id"]
    assert capsys.readouterr().out == "..!\n"


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code: int = status_code
        self.payload: object = payload
        self.text: str = "{}"
        self.reason: str = "reason"
        self.headers: dict[str, str] = {"Content-Type": "application/json"}
        self.content: bytes = b"{}"

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError


def test_google_photos_client_retries_read_timeout_then_succeeds() -> None:
    calls: list[int] = []

    def request_mock(
        _method: str,
        _url: str,
        *,
        json: object | None,
        data: object | None,
        headers: dict[str, str],
        timeout: object,
    ) -> FakeResponse:
        _ = (json, data, headers, timeout)
        calls.append(1)
        if len(calls) < 3:
            raise requests.ReadTimeout
        return FakeResponse(200, {"albums": []})

    client = GooglePhotosClient("token")

    with (
        mock.patch("photoalbums.scripts.create_google_photos_albums.requests.request", side_effect=request_mock),
        mock.patch("photoalbums.scripts.create_google_photos_albums.time.sleep") as sleep_mock,
    ):
        albums = client.list_albums()

    assert albums == []
    assert len(calls) == 3
    assert sleep_mock.call_count == 2


def test_google_photos_client_raises_after_exhausting_retries() -> None:
    def request_mock(
        _method: str,
        _url: str,
        *,
        json: object | None,
        data: object | None,
        headers: dict[str, str],
        timeout: object,
    ) -> FakeResponse:
        _ = (json, data, headers, timeout)
        raise requests.ConnectionError

    client = GooglePhotosClient("token")

    with (
        mock.patch("photoalbums.scripts.create_google_photos_albums.requests.request", side_effect=request_mock),
        mock.patch("photoalbums.scripts.create_google_photos_albums.time.sleep"),
        pytest.raises(requests.ConnectionError),
    ):
        _ = client.list_albums()


def test_google_photos_client_refreshes_access_token_on_unauthorized() -> None:
    calls: list[str] = []

    def refresh_access_token() -> str:
        return "fresh-token"

    def request_mock(
        _method: str,
        _url: str,
        *,
        json: object | None,
        data: object | None,
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        _ = (json, data, timeout)
        calls.append(headers["Authorization"])
        return FakeResponse(401, {}) if len(calls) == 1 else FakeResponse(200, {"albums": []})

    client = GooglePhotosClient("expired-token", refresh_access_token=refresh_access_token)

    with mock.patch("photoalbums.scripts.create_google_photos_albums.requests.request", side_effect=request_mock):
        albums = client.list_albums()

    assert albums == []
    assert calls == ["Bearer expired-token", "Bearer fresh-token"]
