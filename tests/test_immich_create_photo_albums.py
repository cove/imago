from pathlib import Path

import pytest

from immich.create_photo_albums import (
    LocalAlbum,
    _album_date_iso,
    _asset_date_iso,
    create_album_with_assets,
    delete_all_albums,
    discover_local_albums,
    resolve_album_asset_ids,
    set_album_asset_dates,
)

SIDECAR_TEMPLATE = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:imago="https://imago.local/ns/1.0/">'
    "<imago:ViewerSortDate>{date}</imago:ViewerSortDate>"
    "</rdf:Description></rdf:RDF></x:xmpmeta>"
)


def _write_sidecar(image_path: Path, date: str) -> None:
    image_path.with_suffix(".xmp").write_text(SIDECAR_TEMPLATE.format(date=date), encoding="utf-8")


class FakeImmichClient:
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.deleted: list[str] = []
        self.created: list[str] = []
        self.added: list[tuple[str, tuple[str, ...]]] = []
        self.bulk_updates: list[tuple[tuple[str, ...], str]] = []
        self.search_results: dict[str, list[dict]] = {}

    def search_assets_by_original_filename(
        self, original_filename: str, *, visibility: str | None = None
    ) -> list[dict]:
        self.searches.append(original_filename)
        return self.search_results.get(original_filename, [])

    def delete_album(self, album_id: str) -> None:
        self.deleted.append(album_id)

    def create_album(self, name: str) -> dict:
        self.created.append(name)
        return {"id": f"album-{name}"}

    def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        self.added.append((album_id, tuple(asset_ids)))

    def bulk_update_assets(self, asset_ids, *, date_time_original: str) -> None:
        self.bulk_updates.append((tuple(asset_ids), date_time_original))


def test_discovers_logical_albums_from_archive_pages_and_photos_dirs(tmp_path: Path) -> None:
    root = tmp_path / "Photo Albums"
    (root / "Family_1975_B01_Archive").mkdir(parents=True)
    (root / "Family_1975_B01_Pages").mkdir()
    (root / "Family_1975_B01_Photos").mkdir()
    (root / "Family_1975_B01_View").mkdir()
    (root / "Family_1975_B01_Archive" / "Family_1975_B01_P01.tif").write_text("archive", encoding="utf-8")
    (root / "Family_1975_B01_Pages" / "Family_1975_B01_P01.jpg").write_text("page", encoding="utf-8")
    (root / "Family_1975_B01_Photos" / "Family_1975_B01_P01_D01_01.jpg").write_text("photo", encoding="utf-8")
    (root / "Family_1975_B01_View" / "ignored.jpg").write_text("view", encoding="utf-8")

    albums = discover_local_albums(root)

    assert [album.name for album in albums] == ["Family_1975_B01"]
    assert [path.name for path in albums[0].directories] == [
        "Family_1975_B01_Archive",
        "Family_1975_B01_Pages",
        "Family_1975_B01_Photos",
    ]
    assert [path.name for path in albums[0].files] == [
        "Family_1975_B01_P01.tif",
        "Family_1975_B01_P01.jpg",
        "Family_1975_B01_P01_D01_01.jpg",
    ]


def test_resolves_assets_by_original_filename_and_path_suffix(tmp_path: Path) -> None:
    root = tmp_path / "Photo Albums"
    album_dir = root / "Family_1975_B01_Pages"
    album_dir.mkdir(parents=True)
    file_path = album_dir / "Family_1975_B01_P01.jpg"
    file_path.write_text("page", encoding="utf-8")
    client = FakeImmichClient()
    client.search_results[file_path.name] = [
        {"id": "wrong", "originalPath": "/library/Other/Family_1975_B01_P01.jpg"},
        {"id": "right", "originalPath": f"/library/Cordell, Leslie & Audrey/Photo Albums/{album_dir.name}/{file_path.name}"},
    ]
    album = LocalAlbum("Family_1975_B01", (album_dir,), (file_path,))

    asset_ids, missing, asset_files = resolve_album_asset_ids(client, root, album)

    assert asset_ids == ["right"]
    assert missing == []
    assert asset_files == {"right": file_path}


def test_resolve_assets_fails_loud_when_local_file_is_missing_from_immich(tmp_path: Path) -> None:
    root = tmp_path / "Photo Albums"
    album_dir = root / "Family_1975_B01_Pages"
    album_dir.mkdir(parents=True)
    file_path = album_dir / "Family_1975_B01_P01.jpg"
    file_path.write_text("page", encoding="utf-8")
    album = LocalAlbum("Family_1975_B01", (album_dir,), (file_path,))

    with pytest.raises(RuntimeError, match="did not match Immich assets"):
        resolve_album_asset_ids(FakeImmichClient(), root, album)


def test_resolve_assets_caches_original_filename_searches(tmp_path: Path) -> None:
    root = tmp_path / "Photo Albums"
    album_dir = root / "Family_1975_B01_Pages"
    nested = album_dir / "Nested"
    nested.mkdir(parents=True)
    first_path = album_dir / "duplicate.jpg"
    second_path = nested / "duplicate.jpg"
    first_path.write_text("first", encoding="utf-8")
    second_path.write_text("second", encoding="utf-8")
    client = FakeImmichClient()
    client.search_results["duplicate.jpg"] = [
        {"id": "asset-1", "originalPath": f"/library/Photo Albums/{album_dir.name}/{first_path.name}"},
        {"id": "asset-2", "originalPath": f"/library/Photo Albums/{album_dir.name}/Nested/{second_path.name}"},
    ]
    album = LocalAlbum("Family_1975_B01", (album_dir,), (first_path, second_path))

    asset_ids, missing, asset_files = resolve_album_asset_ids(client, root, album)

    assert asset_ids == ["asset-1", "asset-2"]
    assert missing == []
    assert client.searches == ["duplicate.jpg"]
    assert asset_files == {"asset-1": first_path, "asset-2": second_path}


def test_delete_all_albums_deletes_every_existing_album() -> None:
    client = FakeImmichClient()

    deleted_ids = delete_all_albums(
        client,
        existing_albums=[
            {"id": "old-1", "albumName": "Family_1975_B01"},
            {"id": "old-2", "albumName": "Family 1975 B01"},
            {"id": "other", "albumName": "Unrelated"},
        ],
        dry_run=False,
    )

    assert deleted_ids == ["old-1", "old-2", "other"]
    assert client.deleted == ["old-1", "old-2", "other"]


def test_create_album_with_assets_creates_then_adds_assets() -> None:
    client = FakeImmichClient()
    album = LocalAlbum("Family_1975_B01", (), ())

    album_id = create_album_with_assets(
        client,
        album,
        ["asset-1", "asset-2"],
        dry_run=False,
    )

    assert album_id == "album-Family 1975 B01"
    assert client.created == ["Family 1975 B01"]
    assert client.added == [("album-Family 1975 B01", ("asset-1", "asset-2"))]
    assert client.deleted == []


def test_album_date_iso_extracts_first_year_from_name() -> None:
    assert _album_date_iso("Egypt_1975_B00") == "1975-01-01T00:00:00.000Z"
    assert _album_date_iso("Family_1907-1946_B01") == "1907-01-01T00:00:00.000Z"
    assert _album_date_iso("NoYearHere") is None


def test_set_album_asset_dates_bulk_updates_with_album_year() -> None:
    client = FakeImmichClient()
    album = LocalAlbum("Family_1907-1946_B01", (), ())

    date_iso = set_album_asset_dates(client, album, ["a1", "a2"], dry_run=False)

    assert date_iso == "1907-01-01T00:00:00.000Z"
    assert client.bulk_updates == [(("a1", "a2"), "1907-01-01T00:00:00.000Z")]


def test_set_album_asset_dates_skips_when_name_has_no_year() -> None:
    client = FakeImmichClient()
    album = LocalAlbum("Misc_Photos_NoDate", (), ())

    date_iso = set_album_asset_dates(client, album, ["a1"], dry_run=False)

    assert date_iso is None
    assert client.bulk_updates == []


def test_asset_date_iso_reads_viewer_sort_date_from_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "TravelPostCards_1973-1988_B02_P01_D01-00_V.jpg"
    image.write_text("img", encoding="utf-8")
    _write_sidecar(image, "1973-07-03T02:15:00")

    assert _asset_date_iso(image) == "1973-07-03T02:15:00.000Z"
    assert _asset_date_iso(tmp_path / "no_sidecar.jpg") is None


def test_set_album_asset_dates_applies_per_asset_sidecar_dates(tmp_path: Path) -> None:
    page01 = tmp_path / "TravelPostCards_1973-1988_B02_P01_D01-00_V.jpg"
    page31 = tmp_path / "TravelPostCards_1973-1988_B02_P31_D01-00_V.jpg"
    for image in (page01, page31):
        image.write_text("img", encoding="utf-8")
    _write_sidecar(page01, "1973-07-03T02:15:00")
    _write_sidecar(page31, "1988-07-01T21:45:00")
    client = FakeImmichClient()
    album = LocalAlbum("TravelPostCards_1973-1988_B02", (tmp_path,), (page01, page31))
    asset_files = {"a01": page01, "a31": page31}

    earliest = set_album_asset_dates(
        client, album, ["a01", "a31"], asset_files=asset_files, dry_run=False
    )

    assert earliest == "1973-07-03T02:15:00.000Z"
    assert sorted(client.bulk_updates) == [
        (("a01",), "1973-07-03T02:15:00.000Z"),
        (("a31",), "1988-07-01T21:45:00.000Z"),
    ]


def test_set_album_asset_dates_falls_back_to_album_year_without_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "Family_1907-1946_B01_P01_V.jpg"
    image.write_text("img", encoding="utf-8")
    client = FakeImmichClient()
    album = LocalAlbum("Family_1907-1946_B01", (tmp_path,), (image,))

    earliest = set_album_asset_dates(
        client, album, ["a1"], asset_files={"a1": image}, dry_run=False
    )

    assert earliest == "1907-01-01T00:00:00.000Z"
    assert client.bulk_updates == [(("a1",), "1907-01-01T00:00:00.000Z")]
