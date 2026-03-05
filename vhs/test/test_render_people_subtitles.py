from __future__ import annotations

from pathlib import Path

import pytest

from vhs_pipeline import render_pipeline


def _frame_seconds(frames: int) -> float:
    return float(int(frames) * 1001) / 30000.0


def _write_people_tsv(path: Path, rows: list[tuple[int, int, str]]) -> None:
    lines = ["start_frame\tend_frame\tpeople"]
    for start_frame, end_frame, people in rows:
        lines.append(f"{int(start_frame)}\t{int(end_frame)}\t{people}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_people_entries_for_chapter_clips_archive_frame_ranges(tmp_path: Path) -> None:
    people_tsv = tmp_path / "people.tsv"
    _write_people_tsv(
        people_tsv,
        [
            (90, 105, "Lynda"),
            (110, 140, "Jim | Linda"),
            (150, 160, "Outside"),
        ],
    )

    entries = render_pipeline.load_people_entries_for_chapter(people_tsv, 100, 120)

    assert entries == [
        (_frame_seconds(0), _frame_seconds(5), "Lynda"),
        (_frame_seconds(10), _frame_seconds(20), "Jim | Linda"),
    ]


def test_load_people_entries_for_chapter_skips_invalid_rows_and_fixes_single_frame(tmp_path: Path) -> None:
    people_tsv = tmp_path / "people.tsv"
    people_tsv.write_text(
        "start_frame\tend_frame\tpeople\n"
        "# comment row\n"
        "bad\t120\tInvalid Start\n"
        "100\tbad\tInvalid End\n"
        "-1\t120\tNegative Start\n"
        "120\t100\tReverse Range\n"
        "110\t110\tSingle Frame\n"
        "111\t114\t Jim|  Linda \n",
        encoding="utf-8",
    )

    entries = render_pipeline.load_people_entries_for_chapter(people_tsv, 100, 120)

    assert entries == [
        (_frame_seconds(10), _frame_seconds(11), "Single Frame"),
        (_frame_seconds(11), _frame_seconds(14), "Jim | Linda"),
    ]


def test_merge_people_entries_into_srt_replaces_prior_people_line_and_dedupes(tmp_path: Path) -> None:
    srt_path = tmp_path / "chapter.srt"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:03,000\n"
        "Hello there\n"
        "[Old Person]\n",
        encoding="utf-8",
    )

    merged = render_pipeline.merge_people_entries_into_srt(
        srt_path,
        [
            (0.100, 1.000, "Jim|Linda"),
            (1.200, 2.100, "jim | linda"),
            (2.200, 2.900, "Audrey"),
        ],
    )
    assert merged is True

    merged_text = srt_path.read_text(encoding="utf-8")
    assert "Hello there\n[Jim | Linda | Audrey]" in merged_text
    assert "[Old Person]" not in merged_text
    assert merged_text.count("[") == 1


def test_merge_people_entries_into_srt_returns_false_for_missing_or_invalid_srt(tmp_path: Path) -> None:
    missing = tmp_path / "missing.srt"
    assert render_pipeline.merge_people_entries_into_srt(missing, [(0.0, 1.0, "Jim")]) is False

    invalid = tmp_path / "invalid.srt"
    invalid.write_text("not an srt file", encoding="utf-8")
    assert render_pipeline.merge_people_entries_into_srt(invalid, [(0.0, 1.0, "Jim")]) is False
    assert invalid.read_text(encoding="utf-8") == "not an srt file"


def test_write_people_entries_to_srt_vtt_wraps_and_sorts_entries(tmp_path: Path) -> None:
    srt_path = tmp_path / "people.srt"
    vtt_path = tmp_path / "people.vtt"
    wrote = render_pipeline.write_people_entries_to_srt_vtt(
        [
            (_frame_seconds(10), _frame_seconds(12), "Lynda"),
            (_frame_seconds(2), _frame_seconds(4), "Audrey"),
            (_frame_seconds(6), _frame_seconds(7), " "),
        ],
        srt_path,
        vtt_path,
        wrap_in_brackets=True,
    )
    assert wrote is True

    srt_text = srt_path.read_text(encoding="utf-8")
    vtt_text = vtt_path.read_text(encoding="utf-8")
    assert "[Audrey]" in srt_text
    assert "[Lynda]" in srt_text
    assert srt_text.find("[Audrey]") < srt_text.find("[Lynda]")
    assert "[ ]" not in srt_text
    assert "WEBVTT" in vtt_text


def test_write_people_entries_to_srt_vtt_returns_false_when_all_entries_empty(tmp_path: Path) -> None:
    srt_path = tmp_path / "people.srt"
    vtt_path = tmp_path / "people.vtt"
    wrote = render_pipeline.write_people_entries_to_srt_vtt(
        [(_frame_seconds(1), _frame_seconds(2), "  ")],
        srt_path,
        vtt_path,
        wrap_in_brackets=True,
    )
    assert wrote is False
    assert not srt_path.exists()
    assert not vtt_path.exists()


def test_tsv_people_to_srt_vtt_uses_frame_clipping_and_brackets(tmp_path: Path) -> None:
    people_tsv = tmp_path / "people.tsv"
    _write_people_tsv(
        people_tsv,
        [
            (90, 105, "Lynda"),
            (110, 120, "Jim"),
            (140, 150, "Outside"),
        ],
    )
    srt_path = tmp_path / "chapter.people.srt"
    vtt_path = tmp_path / "chapter.people.vtt"

    wrote = render_pipeline.tsv_people_to_srt_vtt(
        people_tsv,
        srt_path,
        vtt_path,
        clip_start_frame=100,
        clip_end_frame=118,
    )
    assert wrote is True

    srt_text = srt_path.read_text(encoding="utf-8")
    first_span = f"{render_pipeline._to_srt_time(_frame_seconds(0))} --> {render_pipeline._to_srt_time(_frame_seconds(5))}"
    second_span = f"{render_pipeline._to_srt_time(_frame_seconds(10))} --> {render_pipeline._to_srt_time(_frame_seconds(18))}"
    assert first_span in srt_text
    assert second_span in srt_text
    assert "[Lynda]" in srt_text
    assert "[Jim]" in srt_text
    assert "[Outside]" not in srt_text


def test_tsv_people_to_ass_writes_italic_people_lines_with_frame_clipping(tmp_path: Path) -> None:
    people_tsv = tmp_path / "people.tsv"
    _write_people_tsv(
        people_tsv,
        [
            (90, 105, "Lynda"),
            (110, 120, "Jim | Linda"),
            (140, 150, "Outside"),
        ],
    )
    ass_path = tmp_path / "chapter.people.ass"

    wrote = render_pipeline.tsv_people_to_ass(
        people_tsv,
        ass_path,
        clip_start_frame=100,
        clip_end_frame=118,
    )
    assert wrote is True

    ass_text = ass_path.read_text(encoding="utf-8")
    assert r"{\i1}Lynda{\i0}" in ass_text
    assert r"{\i1}Jim | Linda{\i0}" in ass_text
    assert r"{\i1}Outside{\i0}" not in ass_text


def test_srt_to_ass_italicizes_people_bracket_lines(tmp_path: Path) -> None:
    srt_path = tmp_path / "chapter.srt"
    ass_path = tmp_path / "chapter.ass"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "Hello there\n"
        "[Jim | Linda]\n",
        encoding="utf-8",
    )

    render_pipeline.srt_to_ass(srt_path, ass_path)

    ass_text = ass_path.read_text(encoding="utf-8")
    assert r"Hello there\N{\rPeople}Jim | Linda{\rDefault}" in ass_text
    assert "[Jim | Linda]" not in ass_text


def test_srt_to_ass_scales_people_font_to_70_percent_of_dialogue_size(tmp_path: Path) -> None:
    srt_path = tmp_path / "chapter.srt"
    ass_path = tmp_path / "chapter.ass"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "Dialogue line\n"
        "[Person Name]\n",
        encoding="utf-8",
    )

    render_pipeline.srt_to_ass(srt_path, ass_path, fontsize=50)
    ass_text = ass_path.read_text(encoding="utf-8")
    assert "Style: People,Calibri,35," in ass_text
    assert r"{\rPeople}Person Name{\rDefault}" in ass_text
