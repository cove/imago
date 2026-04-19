from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "move_page_derived_to_photos.py"
SPEC = importlib.util.spec_from_file_location("move_page_derived_to_photos_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("move_page_derived_to_photos_script", MODULE)
SPEC.loader.exec_module(MODULE)


def test_iter_move_operations_targets_only_derived_page_outputs(tmp_path: Path) -> None:
    pages_dir = tmp_path / "Egypt_1975_B00_Pages"
    photos_dir = tmp_path / "Egypt_1975_B00_Photos"
    pages_dir.mkdir()
    photos_dir.mkdir()

    derived_jpg = pages_dir / "Egypt_1975_B00_P26_D01-01_V.jpg"
    derived_xmp = pages_dir / "Egypt_1975_B00_P26_D01-01_V.xmp"
    page_jpg = pages_dir / "Egypt_1975_B00_P26_V.jpg"
    page_xmp = pages_dir / "Egypt_1975_B00_P26_V.xmp"
    derived_jpg.write_bytes(b"jpg")
    derived_xmp.write_text("<xmp />", encoding="utf-8")
    page_jpg.write_bytes(b"page")
    page_xmp.write_text("<xmp />", encoding="utf-8")

    operations = MODULE._iter_move_operations(tmp_path)

    assert operations == [
        MODULE.MoveOperation(derived_jpg, photos_dir / derived_jpg.name),
        MODULE.MoveOperation(derived_xmp, photos_dir / derived_xmp.name),
    ]


def test_iter_move_operations_includes_derived_media_outputs(tmp_path: Path) -> None:
    pages_dir = tmp_path / "Egypt_1975_B00_Pages"
    photos_dir = tmp_path / "Egypt_1975_B00_Photos"
    pages_dir.mkdir()
    photos_dir.mkdir()

    derived_mp4 = pages_dir / "Egypt_1975_B00_P26_D01-01.mp4"
    page_mp4 = pages_dir / "Egypt_1975_B00_P26.mp4"
    derived_mp4.write_bytes(b"media")
    page_mp4.write_bytes(b"page-media")

    operations = MODULE._iter_move_operations(tmp_path)

    assert operations == [MODULE.MoveOperation(derived_mp4, photos_dir / derived_mp4.name)]


def test_execute_move_operations_moves_files_to_photos_dir(tmp_path: Path) -> None:
    pages_dir = tmp_path / "Egypt_1975_B00_Pages"
    pages_dir.mkdir()
    derived_jpg = pages_dir / "Egypt_1975_B00_P26_D01-01_V.jpg"
    derived_xmp = pages_dir / "Egypt_1975_B00_P26_D01-01_V.xmp"
    derived_jpg.write_bytes(b"jpg")
    derived_xmp.write_text("<xmp />", encoding="utf-8")

    operations = MODULE._iter_move_operations(tmp_path)
    MODULE._validate_move_operations(operations)
    moved, planned = MODULE._execute_move_operations(operations, dry_run=False)

    target_dir = tmp_path / "Egypt_1975_B00_Photos"
    assert moved == 2
    assert planned == 0
    assert not derived_jpg.exists()
    assert not derived_xmp.exists()
    assert (target_dir / derived_jpg.name).read_bytes() == b"jpg"
    assert (target_dir / derived_xmp.name).read_text(encoding="utf-8") == "<xmp />"


def test_validate_move_operations_rejects_existing_target(tmp_path: Path) -> None:
    pages_dir = tmp_path / "Egypt_1975_B00_Pages"
    photos_dir = tmp_path / "Egypt_1975_B00_Photos"
    pages_dir.mkdir()
    photos_dir.mkdir()

    derived_jpg = pages_dir / "Egypt_1975_B00_P26_D01-01_V.jpg"
    derived_jpg.write_bytes(b"jpg")
    (photos_dir / derived_jpg.name).write_bytes(b"existing")

    operations = MODULE._iter_move_operations(tmp_path)

    try:
        MODULE._validate_move_operations(operations)
    except FileExistsError as exc:
        assert str(photos_dir / derived_jpg.name) in str(exc)
    else:
        raise AssertionError("Expected FileExistsError for pre-existing target")
