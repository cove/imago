from __future__ import annotations

from pathlib import Path

import pytest

from photoalbums.lib.crop_number_repair import repair_album_crop_numbers
from photoalbums.lib.xmp_sidecar import read_ai_sidecar_state, write_xmp_sidecar
from photoalbums.lib.xmpmm_provenance import assign_document_id, write_derived_from


def _build_album(
    tmp_path: Path,
    *,
    archive_numbers: list[int],
    crop_numbers: list[int],
) -> tuple[Path, dict[int, str]]:
    album_id = "Family_1907-1946_B01"
    page_token = "P40"
    archive_dir = tmp_path / f"{album_id}_Archive"
    pages_dir = tmp_path / f"{album_id}_Pages"
    photos_dir = tmp_path / f"{album_id}_Photos"
    archive_dir.mkdir()
    pages_dir.mkdir()
    photos_dir.mkdir()

    page_image = pages_dir / f"{album_id}_{page_token}_V.jpg"
    page_image.write_bytes(b"page")
    page_sidecar = page_image.with_suffix(".xmp")
    write_xmp_sidecar(
        page_sidecar,
        person_names=[],
        subjects=[],
        description="Page 40",
        ocr_text="",
    )
    page_doc_id = assign_document_id(page_sidecar)

    for number in archive_numbers:
        suffix = ".png" if number % 2 else ".tif"
        (archive_dir / f"{album_id}_{page_token}_D{number:02d}-01{suffix}").write_bytes(b"archive-derived")

    crop_doc_ids: dict[int, str] = {}
    for number in crop_numbers:
        crop_image = photos_dir / f"{album_id}_{page_token}_D{number:02d}-00_V.jpg"
        crop_image.write_bytes(f"crop-{number}".encode("utf-8"))
        crop_sidecar = crop_image.with_suffix(".xmp")
        write_xmp_sidecar(
            crop_sidecar,
            person_names=[],
            subjects=[],
            description=f"Crop {number}",
            ocr_text="",
        )
        crop_doc_ids[number] = assign_document_id(crop_sidecar)
        write_derived_from(crop_sidecar, page_doc_id, source_path=f"../{album_id}_Pages/{album_id}_{page_token}_V.jpg")

    return photos_dir, crop_doc_ids


def test_repair_album_crop_numbers_fixes_collision_shape(tmp_path: Path) -> None:
    photos_dir, crop_doc_ids = _build_album(tmp_path, archive_numbers=[1, 2, 3], crop_numbers=[1, 2, 3, 4, 5])

    result = repair_album_crop_numbers(tmp_path, album_id="Family_1907-1946_B01", page="40")

    assert result["pages_scanned"] == 1
    assert result["pages_changed"] == 1
    assert result["files_scanned"] == 5
    assert result["files_changed"] == 5
    assert len(result["renames"]) == 5

    for number in range(1, 4):
        assert not (photos_dir / f"Family_1907-1946_B01_P40_D{number:02d}-00_V.jpg").exists()
    for number in range(4, 9):
        assert (photos_dir / f"Family_1907-1946_B01_P40_D{number:02d}-00_V.jpg").exists()
        assert (photos_dir / f"Family_1907-1946_B01_P40_D{number:02d}-00_V.xmp").exists()

    repaired_state = read_ai_sidecar_state(photos_dir / "Family_1907-1946_B01_P40_D04-00_V.xmp")
    assert repaired_state is not None
    assert repaired_state["description"] == "Crop 1"
    assert crop_doc_ids[1] in (photos_dir / "Family_1907-1946_B01_P40_D04-00_V.xmp").read_text(encoding="utf-8")


def test_repair_album_crop_numbers_leaves_canonical_page_unchanged(tmp_path: Path) -> None:
    _build_album(tmp_path, archive_numbers=[1, 2, 3], crop_numbers=[4, 5])

    result = repair_album_crop_numbers(tmp_path, album_id="Family_1907-1946_B01", page="40")

    assert result["pages_scanned"] == 1
    assert result["pages_changed"] == 0
    assert result["files_scanned"] == 2
    assert result["files_changed"] == 0
    assert result["renames"] == []


def test_repair_album_crop_numbers_fails_on_incomplete_crop_pair(tmp_path: Path) -> None:
    photos_dir, _ = _build_album(tmp_path, archive_numbers=[1, 2, 3], crop_numbers=[1, 2, 3, 4, 5])
    (photos_dir / "Family_1907-1946_B01_P40_D03-00_V.xmp").unlink()

    with pytest.raises(FileNotFoundError, match="missing companion file"):
        repair_album_crop_numbers(tmp_path, album_id="Family_1907-1946_B01", page="40")
