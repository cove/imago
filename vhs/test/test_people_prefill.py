from __future__ import annotations

import json
from pathlib import Path

from vhs_pipeline import people_prefill


def _write_chapters(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1001/30000\nSTART=0\nEND=900\nTITLE=Example Chapter\n",
        encoding="utf-8",
    )


def _write_split_chapters_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "parent_chapter\tstart\tend\ttimebase\ttitle\nExample Parent\t0\t900\t1001/30000\tExample Split\n",
        encoding="utf-8",
    )


def test_prefill_people_from_cast_uses_chapter_clip_matches(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_dir = tmp_path / "archive"
    cast_dir = tmp_path / "cast_data"
    _write_chapters(metadata_dir / "demo_archive" / "chapters.ffmetadata")

    cast_dir.mkdir(parents=True, exist_ok=True)
    (cast_dir / "people.json").write_text(
        json.dumps(
            {
                "people": [
                    {"person_id": "p1", "display_name": "Jim"},
                    {"person_id": "p2", "display_name": "Linda"},
                ]
            }
        ),
        encoding="utf-8",
    )
    faces = [
        {
            "source_type": "vhs",
            "source_path": str(tmp_path / "Clips" / "Example Chapter.mp4"),
            "timestamp": "00:00:10.000",
            "person_id": "p1",
            "quality": 0.9,
        },
        {
            "source_type": "vhs",
            "source_path": str(tmp_path / "Clips" / "Example Chapter.mp4"),
            "timestamp": "00:00:10.000",
            "person_id": "p2",
            "quality": 0.9,
        },
    ]
    (cast_dir / "faces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in faces) + "\n",
        encoding="utf-8",
    )

    old_meta = people_prefill.METADATA_DIR
    old_archive = people_prefill.ARCHIVE_DIR
    try:
        people_prefill.METADATA_DIR = metadata_dir
        people_prefill.ARCHIVE_DIR = archive_dir
        result = people_prefill.prefill_people_from_cast(
            archive="demo_archive",
            chapter_title="Example Chapter",
            cast_store_dir=cast_dir,
            min_quality=0.1,
        )
    finally:
        people_prefill.METADATA_DIR = old_meta
        people_prefill.ARCHIVE_DIR = old_archive

    assert result.entries
    first = result.entries[0]
    assert first["people"] == "Jim | Linda"
    assert float(first["end_seconds"]) > float(first["start_seconds"])


def test_prefill_people_from_cast_falls_back_to_chapters_tsv_for_split_titles(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_dir = tmp_path / "archive"
    cast_dir = tmp_path / "cast_data"
    _write_chapters(metadata_dir / "demo_archive" / "chapters.ffmetadata")
    _write_split_chapters_tsv(metadata_dir / "demo_archive" / "chapters.tsv")

    cast_dir.mkdir(parents=True, exist_ok=True)
    (cast_dir / "people.json").write_text(
        json.dumps({"people": [{"person_id": "p1", "display_name": "Jim"}]}),
        encoding="utf-8",
    )
    (cast_dir / "faces.jsonl").write_text(
        json.dumps(
            {
                "source_type": "vhs",
                "source_path": str(tmp_path / "Clips" / "Example Split.mp4"),
                "timestamp": "00:00:10.000",
                "person_id": "p1",
                "quality": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    old_meta = people_prefill.METADATA_DIR
    old_archive = people_prefill.ARCHIVE_DIR
    try:
        people_prefill.METADATA_DIR = metadata_dir
        people_prefill.ARCHIVE_DIR = archive_dir
        result = people_prefill.prefill_people_from_cast(
            archive="demo_archive",
            chapter_title="Example Split",
            cast_store_dir=cast_dir,
            min_quality=0.1,
        )
    finally:
        people_prefill.METADATA_DIR = old_meta
        people_prefill.ARCHIVE_DIR = old_archive

    assert result.entries
    assert result.entries[0]["people"] == "Jim"


def test_apply_prefill_entries_to_people_tsv_replaces_chapter_overlap(
    tmp_path: Path,
) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_dir = tmp_path / "archive"
    archive_name = "demo_archive"
    chapter = "Example Chapter"
    _write_chapters(metadata_dir / archive_name / "chapters.ffmetadata")

    people_tsv = metadata_dir / archive_name / "people.tsv"
    people_tsv.parent.mkdir(parents=True, exist_ok=True)
    people_tsv.write_text(
        "start_frame\tend_frame\tpeople\n150\t360\tOld Person\n1199\t1349\tOutside Chapter\n",
        encoding="utf-8",
    )

    entries = [
        {
            "start_seconds": 10.0,
            "end_seconds": 12.0,
            "people": "Jim | Linda",
        }
    ]

    old_meta = people_prefill.METADATA_DIR
    old_archive = people_prefill.ARCHIVE_DIR
    try:
        people_prefill.METADATA_DIR = metadata_dir
        people_prefill.ARCHIVE_DIR = archive_dir
        out_path, written = people_prefill.apply_prefill_entries_to_people_tsv(
            archive=archive_name,
            chapter_title=chapter,
            entries=entries,
        )
    finally:
        people_prefill.METADATA_DIR = old_meta
        people_prefill.ARCHIVE_DIR = old_archive

    assert out_path == people_tsv
    assert written == 1
    text = people_tsv.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "start\tend\tpeople"
    assert "Old Person" not in text
    assert "00:00:10.000\t00:00:12.000\tJim | Linda" in text
    outside_start = people_prefill._to_timestamp(people_prefill._frame_to_seconds(1199))
    outside_end = people_prefill._to_timestamp(people_prefill._frame_to_seconds(1349))
    assert f"{outside_start}\t{outside_end}\tOutside Chapter" in text
