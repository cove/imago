"""Tests for cast.immich_sync."""
from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest

from cast.immich_sync import (
    _dedupe_names,
    _faces_to_regions,
    _http_get,
    build_exact_local_index,
    build_local_index,
    fetch_assets_by_original_filename,
    fetch_person_assets,
    import_immich_cast_faces,
    sync_immich_faces,
)
from cast.storage import TextFaceStore
from cast.xmp_writer import read_person_in_image

# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------


def test_build_local_index_indexes_by_stem(tmp_path: Path) -> None:
    (tmp_path / "IMG_001.jpg").write_bytes(b"")
    (tmp_path / "IMG_002.JPG").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "IMG_003.tif").write_bytes(b"")

    index = build_local_index(tmp_path)

    assert index["img_001"] == tmp_path / "IMG_001.jpg"
    assert index["img_002"] == tmp_path / "IMG_002.JPG"
    assert index["img_003"] == tmp_path / "sub" / "IMG_003.tif"


def test_build_local_index_skips_non_photo_files(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_bytes(b"")
    (tmp_path / "photo.xmp").write_bytes(b"")
    (tmp_path / "README.txt").write_bytes(b"")

    index = build_local_index(tmp_path)

    assert "photo" in index
    assert "readme" not in index
    assert len([k for k in index if k != "photo"]) == 0


def test_build_local_index_first_match_wins(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first = a / "IMG_001.jpg"
    first.write_bytes(b"")
    (b / "IMG_001.jpg").write_bytes(b"")

    index = build_local_index(tmp_path)

    assert index["img_001"] in (a / "IMG_001.jpg", b / "IMG_001.jpg")


def test_build_exact_local_index_fails_on_duplicate_filenames(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "IMG_001.jpg").write_bytes(b"")
    (tmp_path / "b" / "IMG_001.jpg").write_bytes(b"")

    with pytest.raises(RuntimeError, match="Duplicate local filename"):
        build_exact_local_index(tmp_path)


def test_faces_to_regions_converts_pixel_coords() -> None:
    faces = [
        {
            "imageWidth": 1000,
            "imageHeight": 500,
            "boundingBoxX1": 100,
            "boundingBoxY1": 50,
            "boundingBoxX2": 300,
            "boundingBoxY2": 200,
            "person": {"id": "p1", "name": "Alice"},
        }
    ]

    regions = _faces_to_regions(faces)

    assert len(regions) == 1
    r = regions[0]
    assert r["name"] == "Alice"
    assert abs(float(r["rx"]) - 0.1) < 1e-6
    assert abs(float(r["ry"]) - 0.1) < 1e-6
    assert abs(float(r["rw"]) - 0.2) < 1e-6
    assert abs(float(r["rh"]) - 0.3) < 1e-6


def test_faces_to_regions_skips_unnamed() -> None:
    faces = [
        {
            "imageWidth": 100, "imageHeight": 100,
            "boundingBoxX1": 0, "boundingBoxY1": 0,
            "boundingBoxX2": 50, "boundingBoxY2": 50,
            "person": {"id": "p1", "name": ""},
        },
        {
            "imageWidth": 100, "imageHeight": 100,
            "boundingBoxX1": 0, "boundingBoxY1": 0,
            "boundingBoxX2": 50, "boundingBoxY2": 50,
            "person": None,
        },
        {
            "imageWidth": 100, "imageHeight": 100,
            "boundingBoxX1": 10, "boundingBoxY1": 10,
            "boundingBoxX2": 60, "boundingBoxY2": 60,
            "person": {"id": "p2", "name": "Bob"},
        },
    ]

    regions = _faces_to_regions(faces)

    assert len(regions) == 1
    assert regions[0]["name"] == "Bob"


def test_faces_to_regions_skips_zero_size_bbox() -> None:
    faces = [
        {
            "imageWidth": 100, "imageHeight": 100,
            "boundingBoxX1": 10, "boundingBoxY1": 10,
            "boundingBoxX2": 10,  # zero width
            "boundingBoxY2": 50,
            "person": {"id": "p1", "name": "Alice"},
        },
    ]

    assert _faces_to_regions(faces) == []


def test_faces_to_regions_skips_zero_image_dimensions() -> None:
    faces = [
        {
            "imageWidth": 0, "imageHeight": 100,
            "boundingBoxX1": 10, "boundingBoxY1": 10,
            "boundingBoxX2": 50, "boundingBoxY2": 50,
            "person": {"id": "p1", "name": "Alice"},
        },
    ]

    assert _faces_to_regions(faces) == []


def test_dedupe_names_case_insensitive() -> None:
    result = _dedupe_names(["Alice", "alice", "ALICE", "Bob"])
    assert result == ["Alice", "Bob"]


def test_http_get_prints_immich_error_and_retries_until_recovery(capsys) -> None:
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = b'{"people":[]}'
    error = urllib.error.HTTPError(
        _BASE,
        500,
        "server error",
        {},
        io.BytesIO(b'{"error":"database unavailable"}'),
    )
    with (
        patch("cast.immich_sync.urllib.request.urlopen", side_effect=[error, response]),
        patch("cast.immich_sync.time.sleep") as sleep_mock,
    ):
        payload = _http_get(_BASE, _KEY)

    assert payload == {"people": []}
    assert 'Immich request failed: {"error":"database unavailable"}' in capsys.readouterr().out
    sleep_mock.assert_called_once()


def test_http_get_raises_on_client_error_without_retry() -> None:
    error = urllib.error.HTTPError(
        _BASE,
        404,
        "not found",
        {},
        io.BytesIO(b'{"error":"not found"}'),
    )
    with (
        patch("cast.immich_sync.urllib.request.urlopen", side_effect=error),
        patch("cast.immich_sync.time.sleep") as sleep_mock,
        pytest.raises(urllib.error.HTTPError),
    ):
        _http_get(_BASE, _KEY)

    sleep_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Integration-style tests for sync_immich_faces (HTTP calls mocked)
# ---------------------------------------------------------------------------

_BASE = "http://immich.local:2283"
_KEY = "test-key"

_PERSON_ALICE = {"id": "person-alice", "name": "Alice"}
_ASSET_1 = {"id": "asset-001", "originalFileName": "IMG_001.jpg"}
_FACE_ALICE = {
    "imageWidth": 1000,
    "imageHeight": 800,
    "boundingBoxX1": 100,
    "boundingBoxY1": 80,
    "boundingBoxX2": 300,
    "boundingBoxY2": 320,
    "person": {"id": "person-alice", "name": "Alice"},
}


def _make_http_mocks(people, person_assets, asset_faces):
    """Return side_effect functions for patching Immich GET and POST helpers."""
    def _get(url, api_key, params=None):
        if url.endswith("/api/people"):
            return {"people": people, "hasNextPage": False}
        if "/api/faces" in url:
            asset_id = (params or {}).get("id", "")
            return asset_faces.get(asset_id, [])
        return {}

    def _post(url, api_key, payload):
        if url.endswith("/api/search/metadata"):
            person_id = payload["personIds"][0]
            return {"assets": {"items": person_assets.get(person_id, [])}}
        return {}

    return _get, _post


def test_sync_writes_xmp_and_updates_castdb(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    photo = tmp_path / "IMG_001.jpg"
    photo.write_bytes(b"")

    get_mock, post_mock = _make_http_mocks(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["people_synced"] == 1
    assert stats["assets_matched"] == 1
    assert stats["xmp_updated"] == 1
    assert stats["xmp_skipped"] == 0

    xmp = tmp_path / "IMG_001.xmp"
    assert xmp.is_file()
    assert "Alice" in read_person_in_image(xmp)

    people = store.list_people()
    assert any(p["display_name"] == "Alice" for p in people)


def test_sync_dry_run_does_not_write(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    get_mock, post_mock = _make_http_mocks(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store, dry_run=True)

    assert stats["xmp_updated"] == 1
    assert not (tmp_path / "IMG_001.xmp").exists()
    assert store.list_people() == []


def test_sync_skip_castdb_does_not_add_person(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    get_mock, post_mock = _make_http_mocks(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        sync_immich_faces(_BASE, _KEY, tmp_path, store, update_castdb=False)

    assert store.list_people() == []


def test_sync_no_named_people_returns_early(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    get_mock, post_mock = _make_http_mocks(people=[], person_assets={}, asset_faces={})
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats == {"people_synced": 0, "assets_matched": 0, "xmp_updated": 0, "xmp_skipped": 0}


def test_sync_no_local_file_match(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    # No local photo file present — nothing to match

    get_mock, post_mock = _make_http_mocks(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["assets_matched"] == 0
    assert stats["xmp_updated"] == 0


def test_sync_asset_with_no_named_faces_skipped(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    unnamed_face = {**_FACE_ALICE, "person": None}
    get_mock, post_mock = _make_http_mocks(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [unnamed_face]},
    )
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["xmp_skipped"] == 1
    assert stats["xmp_updated"] == 0
    assert not (tmp_path / "IMG_001.xmp").exists()


def test_sync_existing_person_not_duplicated(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    store.add_person("Alice")
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    get_mock, post_mock = _make_http_mocks(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["people_synced"] == 0
    assert len([p for p in store.list_people() if p["display_name"] == "Alice"]) == 1


def test_sync_reports_asset_query_progress_while_people_are_processed(tmp_path: Path, capsys) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    people = [_PERSON_ALICE, {"id": "person-bob", "name": "Bob"}]
    get_mock, post_mock = _make_http_mocks(people=people, person_assets={}, asset_faces={})
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
        patch("cast.immich_sync.time.monotonic", side_effect=[10.0, 11.0, 12.0]),
    ):
        sync_immich_faces(_BASE, _KEY, tmp_path, store)

    output = capsys.readouterr().out
    assert "Immich asset queries: 50.00% (1/2) 1.00 queries/s\r" in output
    assert "Immich asset queries: 100.00% (2/2) 1.00 queries/s\n" in output


def test_sync_reports_face_query_progress_while_matched_assets_are_processed(
    tmp_path: Path,
    capsys,
) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    get_mock, post_mock = _make_http_mocks(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
        patch("cast.immich_sync.time.monotonic", side_effect=[10.0, 11.0, 20.0, 21.0]),
    ):
        sync_immich_faces(_BASE, _KEY, tmp_path, store)

    output = capsys.readouterr().out
    assert "Immich face queries: 100.00% (1/1) 1.00 queries/s\n" in output


def test_fetch_all_named_people_paginates(tmp_path: Path) -> None:
    """People fetcher follows hasNextPage across multiple API calls."""
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    call_count = 0

    def paging_mock(url, api_key, params=None):
        nonlocal call_count
        if url.endswith("/api/people"):
            call_count += 1
            page = int((params or {}).get("page", 1))
            if page == 1:
                return {"people": [{"id": "p1", "name": "Alice"}], "hasNextPage": True}
            if page == 2:
                return {"people": [{"id": "p2", "name": "Bob"}], "hasNextPage": False}
        return {}

    with (
        patch("cast.immich_sync._http_get", side_effect=paging_mock),
        patch("cast.immich_sync._http_post", return_value={"assets": {"items": []}}),
    ):
        sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert call_count == 2
    names = {p["display_name"] for p in store.list_people()}
    assert names == {"Alice", "Bob"}


def test_fetch_person_assets_follows_search_metadata_pages() -> None:
    def paging_mock(url, api_key, payload):
        assert url.endswith("/api/search/metadata")
        if payload["page"] == 1:
            return {"assets": {"items": [_ASSET_1], "nextPage": "2"}}
        return {"assets": {"items": [{"id": "asset-002", "originalFileName": "IMG_002.jpg"}]}}

    with patch("cast.immich_sync._http_post", side_effect=paging_mock):
        assets = fetch_person_assets(_BASE, _KEY, "person-alice")

    assert [asset["id"] for asset in assets] == ["asset-001", "asset-002"]


def test_fetch_assets_by_original_filename_follows_search_metadata_pages() -> None:
    def paging_mock(url, api_key, payload):
        assert url.endswith("/api/search/metadata")
        assert payload["originalFileName"] == "IMG_001.jpg"
        if payload["page"] == 1:
            return {"assets": {"items": [_ASSET_1], "nextPage": "2"}}
        return {"assets": {"items": [{"id": "asset-002", "originalFileName": "IMG_001.jpg"}]}}

    with patch("cast.immich_sync._http_post", side_effect=paging_mock):
        assets = fetch_assets_by_original_filename(_BASE, _KEY, "IMG_001.jpg")

    assert [asset["id"] for asset in assets] == ["asset-001", "asset-002"]


def test_sync_multiple_people_in_same_photo(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    person_bob = {"id": "person-bob", "name": "Bob"}
    face_bob = {
        "imageWidth": 1000, "imageHeight": 800,
        "boundingBoxX1": 500, "boundingBoxY1": 80,
        "boundingBoxX2": 700, "boundingBoxY2": 320,
        "person": {"id": "person-bob", "name": "Bob"},
    }

    def get_mock(url, api_key, params=None):
        if url.endswith("/api/people"):
            return {"people": [_PERSON_ALICE, person_bob], "hasNextPage": False}
        if "/api/faces" in url:
            return [_FACE_ALICE, face_bob]
        return {}

    def post_mock(url, api_key, payload):
        if url.endswith("/api/search/metadata"):
            return {"assets": {"items": [_ASSET_1]}}
        return {}

    with (
        patch("cast.immich_sync._http_get", side_effect=get_mock),
        patch("cast.immich_sync._http_post", side_effect=post_mock),
    ):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["assets_matched"] == 1
    assert stats["xmp_updated"] == 1

    xmp = tmp_path / "IMG_001.xmp"
    names = read_person_in_image(xmp)
    assert "Alice" in names
    assert "Bob" in names


def test_immich_cast_import_uses_review_seed_for_uncertain_region(tmp_path: Path, monkeypatch) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    image_path = tmp_path / "IMG_001.jpg"
    import cv2
    import numpy as np

    cv2.imwrite(str(image_path), np.zeros((120, 120, 3), dtype=np.uint8))

    class _FakeIngestor:
        def __init__(self, _store, *, require_primary_model):
            assert require_primary_model is True

        @staticmethod
        def _ensure_primary_model_ready():
            return None

        @staticmethod
        def _detect(_image, *, min_size):
            assert min_size == 24
            return []

        @staticmethod
        def is_valid_face_crop(_crop):
            return True

    monkeypatch.setattr("cast.immich_sync.FaceIngestor", _FakeIngestor)
    monkeypatch.setattr(
        "cast.immich_sync.fetch_assets_by_original_filename",
        lambda *_args, **_kwargs: [_ASSET_1],
    )
    monkeypatch.setattr("cast.immich_sync.fetch_asset_faces", lambda *_args, **_kwargs: [_FACE_ALICE])

    stats = import_immich_cast_faces(_BASE, _KEY, tmp_path, store)

    assert stats == {"assets_matched": 1, "faces_imported": 0, "review_seeds": 1}
    assert store.list_faces() == []
    assert store.list_face_review_seeds()[0]["reason"] == "presence_only_no_visible_face"


def test_immich_cast_import_saves_only_clear_visible_arcface_crop(tmp_path: Path, monkeypatch) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    image_path = tmp_path / "IMG_001.jpg"
    import cv2
    import numpy as np

    image = np.full((120, 120, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    class _FakeIngestor:
        def __init__(self, face_store, *, require_primary_model):
            assert require_primary_model is True
            self.store = face_store

        @staticmethod
        def _ensure_primary_model_ready():
            return None

        @staticmethod
        def _detect(_image, *, min_size):
            assert min_size == 24
            return [(10, 10, 40, 40)]

        @staticmethod
        def is_valid_face_crop(_crop):
            return True

        def save_arcface_face_from_crop(self, **kwargs):
            return self.store.add_face(
                embedding=[1.0, 0.0, 0.0],
                person_id=kwargs["person_id"],
                source_type=kwargs["source_type"],
                source_path=kwargs["source_path"],
                bbox=kwargs["bbox"],
                quality=0.9,
                metadata=kwargs["metadata"],
            )

    monkeypatch.setattr("cast.immich_sync.FaceIngestor", _FakeIngestor)
    monkeypatch.setattr("cast.immich_sync.compute_arcface_embedding", lambda _crop: [1.0, 0.0, 0.0])
    monkeypatch.setattr("cast.immich_sync.estimate_face_quality", lambda _crop: 0.9)
    monkeypatch.setattr(
        "cast.immich_sync.fetch_assets_by_original_filename",
        lambda *_args, **_kwargs: [_ASSET_1],
    )
    monkeypatch.setattr("cast.immich_sync.fetch_asset_faces", lambda *_args, **_kwargs: [_FACE_ALICE])

    stats = import_immich_cast_faces(_BASE, _KEY, tmp_path, store)

    assert stats == {"assets_matched": 1, "faces_imported": 1, "review_seeds": 0}
    faces = store.list_faces()
    assert len(faces) == 1
    assert faces[0]["metadata"]["supervision"] == "visible_face_from_presence_region"
