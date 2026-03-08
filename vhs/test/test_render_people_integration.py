from __future__ import annotations

import argparse
from pathlib import Path

from vhs_pipeline import render_pipeline


def _write_chapters(path: Path, title: str, start_frame: int, end_frame: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "__chapter_index\tTIMEBASE\tSTART\tEND\ttitle\n"
        f"1\t1001/30000\t{int(start_frame)}\t{int(end_frame)}\t{title}\n",
        encoding="utf-8",
    )


def _write_people_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "start_frame\tend_frame\tpeople\n"
        "295\t310\tLynda\n"
        "330\t350\tJim | Linda\n"
        "390\t410\tOutside\n",
        encoding="utf-8",
    )


def _write_subtitles_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "start_frame\tend_frame\ttext\tspeaker\tconfidence\tsource\n"
        "301\t330\tMetadata subtitle line\tNarrator\t0.92\tmanual\n",
        encoding="utf-8",
    )


def _configure_render_env(monkeypatch, tmp_path: Path, transcript_mode: str) -> tuple[Path, Path]:
    archive_dir = tmp_path / "Archive"
    metadata_dir = tmp_path / "metadata"
    videos_dir = tmp_path / "Videos"
    clips_dir = tmp_path / "Clips"
    archive_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    archive_name = "demo_archive"
    chapter_title = "Chapter A"
    (archive_dir / f"{archive_name}.mkv").write_bytes(b"stub")

    archive_meta = metadata_dir / archive_name
    _write_chapters(archive_meta / "chapters.tsv", chapter_title, 300, 360)
    (archive_meta / "filter.avs").write_text("c = last\nc\n", encoding="ascii")
    _write_people_tsv(archive_meta / "people.tsv")

    monkeypatch.setattr(render_pipeline, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(render_pipeline, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(render_pipeline, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(render_pipeline, "CLIPS_DIR", clips_dir)
    monkeypatch.setattr(render_pipeline, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(render_pipeline, "assert_expected_frame_count", lambda *args, **kwargs: None)
    monkeypatch.setattr(render_pipeline, "chapter_done", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(render_pipeline, "transcript_mode", lambda *_args, **_kwargs: str(transcript_mode))

    class _WhisperStub:
        @staticmethod
        def load_model(*_args, **_kwargs):
            return object()

    monkeypatch.setattr(render_pipeline, "whisper", _WhisperStub)

    def _fake_transcribe(_model, _audio, final_srt, final_vtt, _final_dir, prompt_text=None):
        _ = prompt_text
        Path(final_srt).write_text(
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Hello there\n\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "General Kenobi\n",
            encoding="utf-8",
        )
        Path(final_vtt).write_text(
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "Hello there\n\n"
            "2\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "General Kenobi\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(render_pipeline, "transcribe_audio", _fake_transcribe)
    return clips_dir, Path(chapter_title)


def test_run_pipeline_merges_people_into_transcribed_sidecars(monkeypatch, tmp_path: Path) -> None:
    clips_dir, chapter_title = _configure_render_env(monkeypatch, tmp_path, transcript_mode="on")
    args = argparse.Namespace(
        archive=["demo_archive"],
        title=[str(chapter_title)],
        title_exact=True,
        no_bob=False,
        debug_extracted_frames=False,
    )

    render_pipeline._run_with_args(args)

    out_srt = clips_dir / f"{chapter_title}.srt"
    out_ass = clips_dir / f"{chapter_title}.ass"
    assert out_srt.exists()
    assert out_ass.exists()

    srt_text = out_srt.read_text(encoding="utf-8")
    assert "Hello there\n[Lynda]" in srt_text
    assert "General Kenobi\n[Jim | Linda]" in srt_text

    ass_text = out_ass.read_text(encoding="utf-8")
    assert r"Hello there\N{\rPeople}Lynda{\rDefault}" in ass_text
    assert r"General Kenobi\N{\rPeople}Jim | Linda{\rDefault}" in ass_text
    assert "[Lynda]" not in ass_text


def test_run_pipeline_writes_people_only_sidecars_when_transcript_off(monkeypatch, tmp_path: Path) -> None:
    clips_dir, chapter_title = _configure_render_env(monkeypatch, tmp_path, transcript_mode="off")
    args = argparse.Namespace(
        archive=["demo_archive"],
        title=[str(chapter_title)],
        title_exact=True,
        no_bob=False,
        debug_extracted_frames=False,
    )

    render_pipeline._run_with_args(args)

    out_srt = clips_dir / f"{chapter_title}.srt"
    out_ass = clips_dir / f"{chapter_title}.ass"
    assert out_srt.exists()
    assert out_ass.exists()

    srt_text = out_srt.read_text(encoding="utf-8")
    assert "[Lynda]" in srt_text
    assert "[Jim | Linda]" in srt_text
    assert "Hello there" not in srt_text

    ass_text = out_ass.read_text(encoding="utf-8")
    assert r"{\rPeople}Lynda{\rDefault}" in ass_text
    assert r"{\rPeople}Jim | Linda{\rDefault}" in ass_text


def test_run_pipeline_prefers_metadata_subtitles_over_whisper_generation(monkeypatch, tmp_path: Path) -> None:
    clips_dir, chapter_title = _configure_render_env(monkeypatch, tmp_path, transcript_mode="on")
    subtitles_tsv = tmp_path / "metadata" / "demo_archive" / "subtitles.tsv"
    _write_subtitles_tsv(subtitles_tsv)

    def _should_not_transcribe(*_args, **_kwargs):
        raise AssertionError("transcribe_audio should not be called when metadata subtitles.tsv exists")

    monkeypatch.setattr(render_pipeline, "transcribe_audio", _should_not_transcribe)
    args = argparse.Namespace(
        archive=["demo_archive"],
        title=[str(chapter_title)],
        title_exact=True,
        no_bob=False,
        debug_extracted_frames=False,
    )

    render_pipeline._run_with_args(args)

    out_srt = clips_dir / f"{chapter_title}.srt"
    out_ass = clips_dir / f"{chapter_title}.ass"
    assert out_srt.exists()
    assert out_ass.exists()

    srt_text = out_srt.read_text(encoding="utf-8")
    assert "Metadata subtitle line\n[Lynda]" in srt_text
    assert "Hello there" not in srt_text

    ass_text = out_ass.read_text(encoding="utf-8")
    assert r"Metadata subtitle line\N{\rPeople}Lynda{\rDefault}" in ass_text
