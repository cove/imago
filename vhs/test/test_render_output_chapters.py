from __future__ import annotations

import argparse
from pathlib import Path

from vhs_pipeline import render_pipeline
from vhs_pipeline.metadata import _read_chapters_tsv_rows, _sort_rows_by_index


def _write_multi_chapter_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "__chapter_index\tffmeta_title\tffmeta_author\tTIMEBASE\tSTART\tEND\ttitle\tcreation_time\n"
        "1\tUnit Archive\tUnit Tester\t1001/30000\t300\t420\tFull Movie\t2001\n"
        "2\tUnit Archive\tUnit Tester\t1001/30000\t300\t340\tIntro\t2001\n"
        "3\tUnit Archive\tUnit Tester\t1001/30000\t340\t420\tOutro\t2001\n",
        encoding="utf-8",
    )


def _write_overlapping_multi_chapter_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle\n"
        "1\tUnit Archive\t1001/30000\t300\t420\tFull Movie\n"
        "2\tUnit Archive\t1001/30000\t300\t340\tIntro\n"
        "3\tUnit Archive\t1001/30000\t310\t340\tIntro Duplicate\n"
        "4\tUnit Archive\t1001/30000\t340\t420\tOutro\n",
        encoding="utf-8",
    )


def _write_prefixed_multi_chapter_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle\n"
        "1\tUnit Archive\t1001/30000\t300\t420\t2001 - Dilbeck's Movie - 01 Full Movie\n"
        "2\tUnit Archive\t1001/30000\t300\t340\t2001 - Dilbeck's Movie - 02 Intro & Coyote Crossing At Agua Caliente\n"
        "3\tUnit Archive\t1001/30000\t340\t420\t2001 - Dilbeck's Movie - 03 Outro\n",
        encoding="utf-8",
    )


def test_write_output_chapter_ffmetadata_keeps_single_matching_chapter(
    tmp_path: Path,
) -> None:
    chapters_tsv = tmp_path / "chapters.tsv"
    _write_multi_chapter_tsv(chapters_tsv)
    header, rows = _read_chapters_tsv_rows(chapters_tsv)
    rows = _sort_rows_by_index(rows)
    _ffm, chapters = render_pipeline._load_chapters_from_tsv(chapters_tsv)
    out = tmp_path / "single-output.ffmetadata"

    count = render_pipeline.write_output_chapter_ffmetadata(
        header,
        rows,
        chapters,
        clip_start_frame=300,
        clip_end_frame=340,
        out_path=out,
    )

    assert count == 1
    assert out.read_text(encoding="utf-8") == (
        ";FFMETADATA1\n"
        "title=Unit Archive\n"
        "author=Unit Tester\n"
        "\n"
        "[CHAPTER]\n"
        "TIMEBASE=1001/30000\n"
        "START=0\n"
        "END=40\n"
        "title=Intro\n"
        "creation_time=2001\n"
    )


def test_write_output_chapter_ffmetadata_drops_container_and_overlapping_duplicates(
    tmp_path: Path,
) -> None:
    chapters_tsv = tmp_path / "chapters.tsv"
    _write_overlapping_multi_chapter_tsv(chapters_tsv)
    header, rows = _read_chapters_tsv_rows(chapters_tsv)
    rows = _sort_rows_by_index(rows)
    _ffm, chapters = render_pipeline._load_chapters_from_tsv(chapters_tsv)
    out = tmp_path / "full-output.ffmetadata"

    count = render_pipeline.write_output_chapter_ffmetadata(
        header,
        rows,
        chapters,
        clip_start_frame=300,
        clip_end_frame=420,
        out_path=out,
    )

    assert count == 2
    text = out.read_text(encoding="utf-8")
    assert "title=Full Movie\n" not in text
    assert "title=Intro Duplicate\n" not in text
    assert "title=Intro\n" in text
    assert "title=Outro\n" in text


def test_write_output_chapter_ffmetadata_shortens_repeated_movie_prefix(
    tmp_path: Path,
) -> None:
    chapters_tsv = tmp_path / "chapters.tsv"
    _write_prefixed_multi_chapter_tsv(chapters_tsv)
    header, rows = _read_chapters_tsv_rows(chapters_tsv)
    rows = _sort_rows_by_index(rows)
    _ffm, chapters = render_pipeline._load_chapters_from_tsv(chapters_tsv)
    out = tmp_path / "prefixed-output.ffmetadata"

    count = render_pipeline.write_output_chapter_ffmetadata(
        header,
        rows,
        chapters,
        clip_start_frame=300,
        clip_end_frame=420,
        output_title="2001 - Dilbeck's Movie - 01 Full Movie",
        out_path=out,
    )

    assert count == 2
    text = out.read_text(encoding="utf-8")
    assert "title=Intro & Coyote Crossing At Agua Caliente\n" in text
    assert "title=Outro\n" in text
    assert "title=2001 - Dilbeck's Movie - 02 Intro" not in text
    assert "title=2001 - Dilbeck's Movie - 03 Outro" not in text


def test_make_encode_final_x264_maps_clip_chapters_after_subtitle_inputs(
    tmp_path: Path,
) -> None:
    qtgmc = tmp_path / "qtgmc.mkv"
    subtitle = tmp_path / "dialogue.ass"
    clip_ffmeta = tmp_path / "clip.ffmetadata"
    final_file = tmp_path / "out.mp4"

    cmd = [
        str(x)
        for x in render_pipeline.make_encode_final_x264(
            qtgmc,
            [{"path": subtitle, "title": "Dialogue", "forced": False}],
            final_file,
            "Unit Tester",
            "Unit Clip",
            "Unit Archive",
            "00:00:10",
            "00:00:20",
            "2001",
            "Pasadena, CA",
            chapter_metadata_path=clip_ffmeta,
            include_audio=True,
        )
    ]

    ffmeta_pos = cmd.index("ffmetadata")
    assert cmd[ffmeta_pos - 1] == "-f"
    assert cmd[ffmeta_pos + 1] == "-i"
    assert cmd[ffmeta_pos + 2] == str(clip_ffmeta)
    assert cmd[cmd.index("-map_metadata") + 1] == "2"
    assert cmd[cmd.index("-map_chapters") + 1] == "2"
    assert "1:s:0" in cmd


def test_run_pipeline_embeds_all_chapters_within_full_movie_range(monkeypatch, tmp_path: Path) -> None:
    archive_dir = tmp_path / "Archive"
    metadata_dir = tmp_path / "metadata"
    videos_dir = tmp_path / "Videos"
    clips_dir = tmp_path / "Clips"
    archive_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    archive_name = "demo_archive"
    chapter_title = "Full Movie"
    (archive_dir / f"{archive_name}.mkv").write_bytes(b"stub")

    archive_meta = metadata_dir / archive_name
    _write_multi_chapter_tsv(archive_meta / "chapters.tsv")
    (archive_meta / "filter.avs").write_text("c = last\nc\n", encoding="ascii")

    run_calls: list[list[str]] = []

    def _fake_run(cmd, cwd=None):
        _ = cwd
        run_calls.append([str(x) for x in cmd])
        return None

    monkeypatch.setattr(render_pipeline, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(render_pipeline, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(render_pipeline, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(render_pipeline, "CLIPS_DIR", clips_dir)
    monkeypatch.setattr(render_pipeline, "run", _fake_run)
    monkeypatch.setattr(render_pipeline, "assert_expected_frame_count", lambda *args, **kwargs: None)
    monkeypatch.setattr(render_pipeline, "chapter_done", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(render_pipeline, "transcript_mode", lambda *_args, **_kwargs: "off")
    monkeypatch.setenv("RENDER_KEEP_TEMP", "1")

    args = argparse.Namespace(
        archive=[archive_name],
        title=[chapter_title],
        title_exact=True,
        no_bob=False,
        subtitles_only=False,
        debug_extracted_frames=False,
    )

    render_pipeline._run_with_args(args)

    final_cmd = run_calls[-1]
    assert final_cmd[final_cmd.index("-map_metadata") + 1] == "1"
    assert final_cmd[final_cmd.index("-map_chapters") + 1] == "1"

    ffmeta_pos = final_cmd.index("ffmetadata")
    ffmeta_path = Path(final_cmd[ffmeta_pos + 2])
    ffmeta_text = ffmeta_path.read_text(encoding="utf-8")

    assert ffmeta_text.count("[CHAPTER]") == 2
    assert ("[CHAPTER]\nTIMEBASE=1001/30000\nSTART=0\nEND=40\ntitle=Intro\n") in ffmeta_text
    assert ("[CHAPTER]\nTIMEBASE=1001/30000\nSTART=40\nEND=120\ntitle=Outro\n") in ffmeta_text
    assert "title=Full Movie\n" not in ffmeta_text
