from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from photoalbums.lib.crop_source_repair import (
    find_crop_sidecars_needing_source_repair,
    repair_album_crop_sources,
    repair_crop_sidecar_source,
)
from photoalbums.lib.xmp_sidecar import read_ai_sidecar_state, write_xmp_sidecar
from photoalbums.lib.xmpmm_provenance import assign_document_id, write_derived_from


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _build_album(tmp_path: Path) -> tuple[Path, Path]:
    archive_dir = tmp_path / "Portugal_1988_B00_Archive"
    pages_dir = tmp_path / "Portugal_1988_B00_Pages"
    photos_dir = tmp_path / "Portugal_1988_B00_Photos"
    archive_dir.mkdir()
    pages_dir.mkdir()
    photos_dir.mkdir()
    (archive_dir / "Portugal_1988_B00_P23_S01.tif").write_bytes(b"scan")
    (archive_dir / "Portugal_1988_B00_P23_S02.tif").write_bytes(b"scan")
    page_image = pages_dir / "Portugal_1988_B00_P23_V.jpg"
    page_image.write_bytes(b"page")
    page_sidecar = page_image.with_suffix(".xmp")
    write_xmp_sidecar(
        page_sidecar,
        creator_tool="imago-test",
        person_names=[],
        subjects=[],
        description="",
        album_title="PANAMA CANAL & MEXICO 1987 PORTUGAL 1988",
        source_text=(
            "PANAMA CANAL & MEXICO 1987 PORTUGAL 1988 Page 23 "
            "Scan(s) S01 S02; Portugal_1988_B00_P23_S01.tif; Portugal_1988_B00_P23_S02.tif"
        ),
        ocr_text="",
        detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
        image_width=200,
        image_height=100,
    )
    assign_document_id(page_sidecar)

    write_xmp_sidecar(
        archive_dir / "Portugal_1988_B00_P23_S01.xmp",
        creator_tool="imago-test",
        person_names=[],
        subjects=[],
        description="",
        album_title="PANAMA CANAL & MEXICO 1987 PORTUGAL 1988",
        source_text=(
            "PANAMA CANAL & MEXICO 1987 PORTUGAL 1988 Page 23 "
            "Scan(s) S01 S02; Portugal_1988_B00_P23_S01.tif; Portugal_1988_B00_P23_S02.tif"
        ),
        ocr_text="",
    )

    crop_image = photos_dir / "Portugal_1988_B00_P23_D01-00_V.jpg"
    crop_image.write_bytes(b"crop")
    crop_sidecar = crop_image.with_suffix(".xmp")
    write_xmp_sidecar(
        crop_sidecar,
        creator_tool="imago-test",
        person_names=[],
        subjects=[],
        description="",
        source_text="Page 23 Scan(s) S01 S02; Portugal_1988_B00_P23_S01.tif; Portugal_1988_B00_P23_S02.tif",
        ocr_text="",
        detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
        image_width=100,
        image_height=100,
    )
    page_doc_id = assign_document_id(page_sidecar)
    write_derived_from(crop_sidecar, page_doc_id, source_path="../Portugal_1988_B00_Pages/Portugal_1988_B00_P23_V.jpg")
    return crop_sidecar, page_sidecar


def test_find_crop_sidecars_needing_source_repair_reports_stale_crop(tmp_path: Path) -> None:
    crop_sidecar, _ = _build_album(tmp_path)

    matches = find_crop_sidecars_needing_source_repair(tmp_path, album_id="Portugal_1988_B00", page="23")

    assert matches == [crop_sidecar]


def test_repair_crop_sidecar_source_rewrites_album_title_and_dc_source(tmp_path: Path) -> None:
    crop_sidecar, _ = _build_album(tmp_path)

    changed = repair_crop_sidecar_source(crop_sidecar)

    assert changed is True
    state = read_ai_sidecar_state(crop_sidecar)
    assert state is not None
    assert state["album_title"] == "PANAMA CANAL & MEXICO 1987 PORTUGAL 1988"
    assert (
        state["source_text"]
        == "PANAMA CANAL & MEXICO 1987 PORTUGAL 1988 Page 23 Scan(s) S01 S02; "
        "Portugal_1988_B00_P23_S01.tif; Portugal_1988_B00_P23_S02.tif"
    )


def test_repair_album_crop_sources_only_counts_matching_crop_sidecars(tmp_path: Path) -> None:
    _build_album(tmp_path)
    untouched = _write_text(
        tmp_path / "Portugal_1988_B00_Pages" / "Portugal_1988_B00_P23_V.xmp",
        """
        <?xml version="1.0" encoding="utf-8"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
          <rdf:RDF>
            <rdf:Description rdf:about="" />
          </rdf:RDF>
        </x:xmpmeta>
        """,
    )

    result = repair_album_crop_sources(tmp_path, album_id="Portugal_1988_B00")

    assert result == {"files_scanned": 1, "files_changed": 1}
    assert untouched.read_text(encoding="utf-8")
