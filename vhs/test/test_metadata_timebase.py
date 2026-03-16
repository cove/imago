from pathlib import Path

from vhs_pipeline.metadata import (
    ffmetadata_to_chapters_tsv,
    generate_ffmetadata_from_chapters_tsv,
    generate_mkv_chapters_xml,
    generate_tsv_metadata,
)


def _write_ffmetadata(path: Path, *, timebase: str, start: int, end: int) -> None:
    path.write_text(
        "\n".join(
            [
                ";FFMETADATA1",
                "title=Unit Archive",
                "author=Unit Tester",
                "",
                "[CHAPTER]",
                f"TIMEBASE={timebase}",
                f"START={start}",
                f"END={end}",
                "title=Unit Chapter",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_master_chapters_tsv(
    path: Path, *, timebase: str, start: int, end: int
) -> Path:
    ffmeta = path.parent / "chapters.ffmetadata"
    _write_ffmetadata(ffmeta, timebase=timebase, start=start, end=end)
    ffmetadata_to_chapters_tsv(ffmeta, path)
    return path


def test_generate_tsv_metadata_uses_normalized_seconds_for_1001_30000(
    tmp_path: Path,
) -> None:
    chapters_tsv = tmp_path / "chapters.tsv"
    out = tmp_path / "markers.tsv"
    _write_master_chapters_tsv(chapters_tsv, timebase="1001/30000", start=101, end=303)

    generate_tsv_metadata(chapters_tsv, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    row = lines[1].split("\t")
    assert row[3] == "3.37"
    assert row[4] == "10.11"


def test_generate_tsv_metadata_uses_normalized_seconds_for_1_100(
    tmp_path: Path,
) -> None:
    chapters_tsv = tmp_path / "chapters.tsv"
    out = tmp_path / "markers.tsv"
    _write_master_chapters_tsv(chapters_tsv, timebase="1/100", start=6315, end=6603)

    generate_tsv_metadata(chapters_tsv, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    row = lines[1].split("\t")
    assert row[3] == "63.15"
    assert row[4] == "66.03"


def test_generate_mkv_chapters_xml_uses_normalized_seconds(tmp_path: Path) -> None:
    chapters_tsv = tmp_path / "chapters.tsv"
    out = tmp_path / "markers.mkvchapters.xml"
    _write_master_chapters_tsv(chapters_tsv, timebase="1001/30000", start=101, end=303)

    generate_mkv_chapters_xml(chapters_tsv, out)
    xml = out.read_text(encoding="utf-8")
    assert "<ChapterTimeStart>00:00:03.370</ChapterTimeStart>" in xml
    assert "<ChapterTimeEnd>00:00:10.110</ChapterTimeEnd>" in xml


def test_chapters_tsv_round_trip_recreates_ffmetadata_exactly(tmp_path: Path) -> None:
    # Source ffmetadata has per-chapter column variation and an empty comment= field.
    # After round-trip: empty fields are dropped, and column order is normalized to
    # the TSV-header order (first-seen across all chapters).
    source = (
        ";FFMETADATA1\n"
        "title=Unit Archive (VHS Tape)\n"
        "author=Unit Tester\n"
        "\n"
        "[CHAPTER]\n"
        "TIMEBASE=1001/30000\n"
        "START=101\n"
        "END=303\n"
        "title=Unit Chapter\n"
        "creation_time=2001\n"
        "comment=\n"
        "\n"
        "[CHAPTER]\n"
        "START=6315\n"
        "END=6603\n"
        "TIMEBASE=1/100\n"
        "title=Second Chapter\n"
        "location=Pasadena, CA\n"
    )
    # Expected: empty comment= dropped; chapter 2 column order follows TSV-header order
    # (TIMEBASE before START/END because that's the first-seen order from chapter 1).
    expected = (
        ";FFMETADATA1\n"
        "title=Unit Archive (VHS Tape)\n"
        "author=Unit Tester\n"
        "\n"
        "[CHAPTER]\n"
        "TIMEBASE=1001/30000\n"
        "START=101\n"
        "END=303\n"
        "title=Unit Chapter\n"
        "creation_time=2001\n"
        "\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/100\n"
        "START=6315\n"
        "END=6603\n"
        "title=Second Chapter\n"
        "location=Pasadena, CA\n"
    )
    ffmeta = tmp_path / "chapters.ffmetadata"
    chapters_tsv = tmp_path / "chapters.tsv"
    roundtrip = tmp_path / "chapters.roundtrip.ffmetadata"
    ffmeta.write_text(source, encoding="utf-8")

    ffmetadata_to_chapters_tsv(ffmeta, chapters_tsv)
    generate_ffmetadata_from_chapters_tsv(chapters_tsv, roundtrip)

    assert roundtrip.read_text(encoding="utf-8") == expected
