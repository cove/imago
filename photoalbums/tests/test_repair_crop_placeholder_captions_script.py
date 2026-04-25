from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from photoalbums.lib.xmp_sidecar import read_ai_sidecar_state, write_region_list, write_xmp_sidecar
from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repair_crop_placeholder_captions.py"
SPEC = importlib.util.spec_from_file_location("repair_crop_placeholder_captions_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("repair_crop_placeholder_captions_script", MODULE)
SPEC.loader.exec_module(MODULE)


def _write_album(tmp_path: Path, *, region_name: str, crop_description: str) -> Path:
    pages_dir = tmp_path / "Russia_1984_B02_Pages"
    photos_dir = tmp_path / "Russia_1984_B02_Photos"
    pages_dir.mkdir()
    photos_dir.mkdir()

    page_sidecar = pages_dir / "Russia_1984_B02_P36_V.xmp"
    write_xmp_sidecar(
        page_sidecar,
        person_names=[],
        subjects=[],
        description="Canal walk",
        ocr_text="",
    )
    write_region_list(
        page_sidecar,
        [
            RegionWithCaption(
                RegionResult(index=0, x=0, y=0, width=100, height=100, caption_hint="", person_names=()),
                caption=region_name,
            )
        ],
        100,
        100,
    )

    crop_sidecar = photos_dir / "Russia_1984_B02_P36_D01-00_V.xmp"
    write_xmp_sidecar(
        crop_sidecar,
        person_names=[],
        subjects=[],
        description=crop_description,
        ocr_text="",
    )
    return crop_sidecar


def test_repair_sidecar_updates_only_placeholder_captions(tmp_path: Path) -> None:
    crop_sidecar = _write_album(tmp_path, region_name="photo_1", crop_description="photo_1")

    outcome = MODULE._repair_sidecar(crop_sidecar, dry_run=False)

    assert outcome == "fixed"
    state = read_ai_sidecar_state(crop_sidecar)
    assert state is not None
    assert state["description"] == "Page caption: Canal walk"


def test_repair_sidecar_skips_non_placeholder_region_names(tmp_path: Path) -> None:
    crop_sidecar = _write_album(tmp_path, region_name="Canal boat", crop_description="Canal boat")

    outcome = MODULE._repair_sidecar(crop_sidecar, dry_run=False)

    assert outcome == "skip"
    state = read_ai_sidecar_state(crop_sidecar)
    assert state is not None
    assert state["description"] == "Canal boat"


def test_repair_sidecar_does_not_overwrite_manual_crop_caption(tmp_path: Path) -> None:
    crop_sidecar = _write_album(tmp_path, region_name="photo_1", crop_description="Handwritten custom note")

    outcome = MODULE._repair_sidecar(crop_sidecar, dry_run=False)

    assert outcome == "skip"
    state = read_ai_sidecar_state(crop_sidecar)
    assert state is not None
    assert state["description"] == "Handwritten custom note"


def test_iter_target_sidecars_skips_non_crop_derived_views(tmp_path: Path) -> None:
    photos_dir = tmp_path / "TheOrient_1974_B00_Photos"
    photos_dir.mkdir()
    crop_sidecar = photos_dir / "TheOrient_1974_B00_P44_D03-00_V.xmp"
    page_derived_sidecar = photos_dir / "TheOrient_1974_B00_P44_D01-01_V.xmp"
    crop_sidecar.write_text("<xmp />", encoding="utf-8")
    page_derived_sidecar.write_text("<xmp />", encoding="utf-8")

    targets = MODULE._iter_target_sidecars(tmp_path, "theorient_1974_b00", "44")

    assert targets == [crop_sidecar]


def test_repair_sidecar_skips_when_parent_region_is_missing(tmp_path: Path) -> None:
    pages_dir = tmp_path / "England_1983_B01_Pages"
    photos_dir = tmp_path / "England_1983_B01_Photos"
    pages_dir.mkdir()
    photos_dir.mkdir()

    page_sidecar = pages_dir / "England_1983_B01_P03_V.xmp"
    write_xmp_sidecar(
        page_sidecar,
        person_names=[],
        subjects=[],
        description="Village walk",
        ocr_text="",
    )

    crop_sidecar = photos_dir / "England_1983_B01_P03_D01-00_V.xmp"
    write_xmp_sidecar(
        crop_sidecar,
        person_names=[],
        subjects=[],
        description="Village walk",
        ocr_text="",
    )

    outcome = MODULE._repair_sidecar(crop_sidecar, dry_run=False)

    assert outcome == "skip"
