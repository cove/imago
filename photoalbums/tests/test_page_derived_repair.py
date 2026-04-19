from __future__ import annotations

from pathlib import Path

from PIL import Image

from photoalbums.lib.page_derived_repair import repair_page_derived_views


def _write_solid_image(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), rgb).save(path)


def test_repair_page_derived_views_renames_without_rerendering_and_removes_orphans(tmp_path: Path) -> None:
    album_id = "Family_1907-1946_B01"
    archive_dir = tmp_path / f"{album_id}_Archive"
    pages_dir = tmp_path / f"{album_id}_Pages"
    archive_dir.mkdir()
    pages_dir.mkdir()

    _write_solid_image(archive_dir / f"{album_id}_P40_D01-01.png", (200, 20, 20))
    _write_solid_image(archive_dir / f"{album_id}_P40_D02-01.png", (20, 200, 20))

    wrong_one = pages_dir / f"{album_id}_P40_D02-01_V.jpg"
    wrong_two = pages_dir / f"{album_id}_P40_D01-01_V.jpg"
    orphan = pages_dir / f"{album_id}_P40_D01-02_V.jpg"

    _write_solid_image(wrong_one, (200, 20, 20))
    wrong_one.with_suffix(".xmp").write_text("<xmp>one</xmp>", encoding="utf-8")
    _write_solid_image(wrong_two, (20, 200, 20))
    wrong_two.with_suffix(".xmp").write_text("<xmp>two</xmp>", encoding="utf-8")
    _write_solid_image(orphan, (20, 20, 200))
    orphan.with_suffix(".xmp").write_text("<xmp>orphan</xmp>", encoding="utf-8")

    result = repair_page_derived_views(tmp_path, album_id=album_id, page="40")

    assert result == {
        "archives_scanned": 1,
        "files_scanned": 3,
        "files_renamed": 2,
        "orphan_files_removed": 2,
        "missing_expected_views": 0,
    }
    repaired_one = pages_dir / f"{album_id}_P40_D01-01_V.jpg"
    repaired_two = pages_dir / f"{album_id}_P40_D02-01_V.jpg"
    assert repaired_one.exists()
    assert repaired_two.exists()
    assert repaired_one.with_suffix(".xmp").read_text(encoding="utf-8") == "<xmp>one</xmp>"
    assert repaired_two.with_suffix(".xmp").read_text(encoding="utf-8") == "<xmp>two</xmp>"
    assert not orphan.exists()
    assert not orphan.with_suffix(".xmp").exists()


def test_repair_page_derived_views_reports_missing_expected_views_without_rendering(tmp_path: Path) -> None:
    album_id = "Family_1907-1946_B01"
    archive_dir = tmp_path / f"{album_id}_Archive"
    pages_dir = tmp_path / f"{album_id}_Pages"
    archive_dir.mkdir()
    pages_dir.mkdir()

    _write_solid_image(archive_dir / f"{album_id}_P40_D01-01.png", (200, 20, 20))
    _write_solid_image(archive_dir / f"{album_id}_P40_D02-01.png", (20, 200, 20))
    _write_solid_image(pages_dir / f"{album_id}_P40_D09-01_V.jpg", (200, 20, 20))
    (pages_dir / f"{album_id}_P40_D09-01_V.xmp").write_text("<xmp>one</xmp>", encoding="utf-8")

    result = repair_page_derived_views(tmp_path, album_id=album_id, page="40")

    assert result == {
        "archives_scanned": 1,
        "files_scanned": 1,
        "files_renamed": 1,
        "orphan_files_removed": 0,
        "missing_expected_views": 1,
    }
    assert (pages_dir / f"{album_id}_P40_D01-01_V.jpg").exists()
    assert not (pages_dir / f"{album_id}_P40_D02-01_V.jpg").exists()


def test_repair_page_derived_views_groups_candidates_by_page(tmp_path: Path) -> None:
    album_id = "MainlandChina_1986_B02"
    archive_dir = tmp_path / f"{album_id}_Archive"
    pages_dir = tmp_path / f"{album_id}_Pages"
    archive_dir.mkdir()
    pages_dir.mkdir()

    _write_solid_image(archive_dir / f"{album_id}_P16_D01-01.png", (200, 20, 20))
    _write_solid_image(archive_dir / f"{album_id}_P29_D01-01.png", (20, 200, 20))
    _write_solid_image(pages_dir / f"{album_id}_P16_D01-01_V.jpg", (200, 20, 20))
    (pages_dir / f"{album_id}_P16_D01-01_V.xmp").write_text("<xmp>p16</xmp>", encoding="utf-8")
    _write_solid_image(pages_dir / f"{album_id}_P29_D01-01_V.jpg", (20, 200, 20))
    (pages_dir / f"{album_id}_P29_D01-01_V.xmp").write_text("<xmp>p29</xmp>", encoding="utf-8")

    result = repair_page_derived_views(tmp_path, album_id=album_id)

    assert result == {
        "archives_scanned": 1,
        "files_scanned": 2,
        "files_renamed": 0,
        "orphan_files_removed": 0,
        "missing_expected_views": 0,
    }
    assert (pages_dir / f"{album_id}_P16_D01-01_V.jpg").exists()
    assert (pages_dir / f"{album_id}_P29_D01-01_V.jpg").exists()
