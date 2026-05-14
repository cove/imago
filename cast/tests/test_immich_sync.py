"""Tests for cast.immich_sync."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cast.immich_sync import (
    _dedupe_names,
    _faces_to_regions,
    build_local_index,
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


def _make_http_mock(people, person_assets, asset_faces):
    """Return a side_effect function for patching _http_get."""
    def _get(url, api_key, params=None):
        if "/api/people" in url and not any(pid in url for pid in ["alice", "bob"]):
            return {"people": people, "hasNextPage": False}
        for pid, assets in person_assets.items():
            if f"/api/people/{pid}" in url:
                return assets
        if "/api/faces" in url:
            asset_id = (params or {}).get("id", "")
            return asset_faces.get(asset_id, [])
        return {}
    return _get


def test_sync_writes_xmp_and_updates_castdb(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    photo = tmp_path / "IMG_001.jpg"
    photo.write_bytes(b"")

    http_mock = _make_http_mock(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with patch("cast.immich_sync._http_get", side_effect=http_mock):
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

    http_mock = _make_http_mock(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with patch("cast.immich_sync._http_get", side_effect=http_mock):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store, dry_run=True)

    assert stats["xmp_updated"] == 1
    assert not (tmp_path / "IMG_001.xmp").exists()
    assert store.list_people() == []


def test_sync_skip_castdb_does_not_add_person(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    http_mock = _make_http_mock(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with patch("cast.immich_sync._http_get", side_effect=http_mock):
        sync_immich_faces(_BASE, _KEY, tmp_path, store, update_castdb=False)

    assert store.list_people() == []


def test_sync_no_named_people_returns_early(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    http_mock = _make_http_mock(people=[], person_assets={}, asset_faces={})
    with patch("cast.immich_sync._http_get", side_effect=http_mock):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats == {"people_synced": 0, "assets_matched": 0, "xmp_updated": 0, "xmp_skipped": 0}


def test_sync_no_local_file_match(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    # No local photo file present — nothing to match

    http_mock = _make_http_mock(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with patch("cast.immich_sync._http_get", side_effect=http_mock):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["assets_matched"] == 0
    assert stats["xmp_updated"] == 0


def test_sync_asset_with_no_named_faces_skipped(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    unnamed_face = {**_FACE_ALICE, "person": None}
    http_mock = _make_http_mock(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [unnamed_face]},
    )
    with patch("cast.immich_sync._http_get", side_effect=http_mock):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["xmp_skipped"] == 1
    assert stats["xmp_updated"] == 0
    assert not (tmp_path / "IMG_001.xmp").exists()


def test_sync_existing_person_not_duplicated(tmp_path: Path) -> None:
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    store.add_person("Alice")
    (tmp_path / "IMG_001.jpg").write_bytes(b"")

    http_mock = _make_http_mock(
        people=[_PERSON_ALICE],
        person_assets={"person-alice": [_ASSET_1]},
        asset_faces={"asset-001": [_FACE_ALICE]},
    )
    with patch("cast.immich_sync._http_get", side_effect=http_mock):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["people_synced"] == 0
    assert len([p for p in store.list_people() if p["display_name"] == "Alice"]) == 1


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
        if "/api/people/p1/assets" in url or "/api/people/p2/assets" in url:
            return []
        return {}

    with patch("cast.immich_sync._http_get", side_effect=paging_mock):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert call_count == 2
    names = {p["display_name"] for p in store.list_people()}
    assert names == {"Alice", "Bob"}


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

    def multi_mock(url, api_key, params=None):
        if "/api/people" in url and "person-" not in url:
            return {"people": [_PERSON_ALICE, person_bob], "hasNextPage": False}
        if "/api/people/person-alice/assets" in url:
            return [_ASSET_1]
        if "/api/people/person-bob/assets" in url:
            return [_ASSET_1]
        if "/api/faces" in url:
            return [_FACE_ALICE, face_bob]
        return {}

    with patch("cast.immich_sync._http_get", side_effect=multi_mock):
        stats = sync_immich_faces(_BASE, _KEY, tmp_path, store)

    assert stats["assets_matched"] == 1
    assert stats["xmp_updated"] == 1

    xmp = tmp_path / "IMG_001.xmp"
    names = read_person_in_image(xmp)
    assert "Alice" in names
    assert "Bob" in names
