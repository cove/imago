"""Tests for _rename_chapter_outputs in the VHS tuner server."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apps.plain_html_wizard.server import _rename_chapter_outputs
from common import safe

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_mp4(path: Path, size: int = 200_000) -> None:
    """Write a stub MP4 file large enough to pass the 100 KB guard."""
    path.write_bytes(b"\x00" * size)


def _make_sidecars(dir_: Path, stem: str) -> list[Path]:
    files = []
    for ext in (".srt", ".vtt", ".ass"):
        p = dir_ / f"{stem}{ext}"
        p.write_text("dummy subtitle")
        files.append(p)
    return files


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


TEST_ARCHIVE = "test_archive"


@pytest.fixture()
def videos_dir(tmp_path: Path, monkeypatch):
    d = tmp_path / "videos"
    d.mkdir()
    import apps.plain_html_wizard.server as wizard_server

    monkeypatch.setattr(wizard_server, "videos_dir_for", lambda _archive: d)
    monkeypatch.setattr(wizard_server, "clips_dir_for", lambda _archive: tmp_path / "clips_nonexistent")
    return d


# ---------------------------------------------------------------------------
# subtitle sidecar rename
# ---------------------------------------------------------------------------


def test_subtitle_sidecars_are_renamed(videos_dir: Path, monkeypatch) -> None:
    old_title = "Model A Chapter"
    new_title = "Model A Roadster"
    old_stem = safe(old_title)
    new_stem = safe(new_title)

    _make_mp4(videos_dir / f"{old_stem}.mp4")
    _make_sidecars(videos_dir, old_stem)

    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        # make ffmpeg write the tmp file so rename succeeds
        def fake_ffmpeg(*args, **kwargs):
            # args[0] is the command list; extract tmp path and create it
            cmd = args[0]
            tmp = Path(cmd[-1])
            tmp.write_bytes(b"\x00" * 200_000)
            return mock_result

        mock_run.side_effect = fake_ffmpeg
        _rename_chapter_outputs(old_title, new_title, TEST_ARCHIVE)

    for ext in (".srt", ".vtt", ".ass"):
        assert (videos_dir / f"{new_stem}{ext}").exists(), f"missing {ext}"
        assert not (videos_dir / f"{old_stem}{ext}").exists(), f"old {ext} not removed"


# ---------------------------------------------------------------------------
# MP4 remux path (ffmpeg succeeds)
# ---------------------------------------------------------------------------


def test_mp4_renamed_and_old_deleted_on_ffmpeg_success(videos_dir: Path) -> None:
    old_title = "Model A Chapter"
    new_title = "Model A Roadster"
    old_stem = safe(old_title)
    new_stem = safe(new_title)

    _make_mp4(videos_dir / f"{old_stem}.mp4")

    mock_result = MagicMock()
    mock_result.returncode = 0

    def fake_ffmpeg(*args, **kwargs):
        cmd = args[0]
        tmp = Path(cmd[-1])
        tmp.write_bytes(b"\x00" * 200_000)
        return mock_result

    with patch("subprocess.run", side_effect=fake_ffmpeg):
        result = _rename_chapter_outputs(old_title, new_title, TEST_ARCHIVE)

    assert (videos_dir / f"{new_stem}.mp4").exists()
    assert not (videos_dir / f"{old_stem}.mp4").exists()
    assert f"{new_stem}.mp4" in result


# ---------------------------------------------------------------------------
# MP4 fallback rename (ffmpeg fails)
# ---------------------------------------------------------------------------


def test_mp4_renamed_without_remux_on_ffmpeg_failure(videos_dir: Path) -> None:
    old_title = "Model A Chapter"
    new_title = "Model A Roadster"
    old_stem = safe(old_title)
    new_stem = safe(new_title)

    _make_mp4(videos_dir / f"{old_stem}.mp4")

    mock_result = MagicMock()
    mock_result.returncode = 1  # simulate ffmpeg failure

    with patch("subprocess.run", return_value=mock_result):
        result = _rename_chapter_outputs(old_title, new_title, TEST_ARCHIVE)

    assert (videos_dir / f"{new_stem}.mp4").exists()
    assert not (videos_dir / f"{old_stem}.mp4").exists()
    assert f"{new_stem}.mp4" in result


# ---------------------------------------------------------------------------
# ffmetadata title argument
# ---------------------------------------------------------------------------


def test_ffmpeg_called_with_correct_title_metadata(videos_dir: Path) -> None:
    old_title = "Model A Chapter"
    new_title = "Model A Roadster"
    old_stem = safe(old_title)

    _make_mp4(videos_dir / f"{old_stem}.mp4")

    mock_result = MagicMock()
    mock_result.returncode = 0
    captured = {}

    def fake_ffmpeg(*args, **kwargs):
        captured["cmd"] = args[0]
        tmp = Path(args[0][-1])
        tmp.write_bytes(b"\x00" * 200_000)
        return mock_result

    with patch("subprocess.run", side_effect=fake_ffmpeg):
        _rename_chapter_outputs(old_title, new_title, TEST_ARCHIVE)

    cmd = captured["cmd"]
    assert f"title={new_title}" in cmd
    assert "-c" in cmd and "copy" in cmd


# ---------------------------------------------------------------------------
# skip file below size guard
# ---------------------------------------------------------------------------


def test_small_mp4_is_skipped(videos_dir: Path) -> None:
    old_title = "Model A Chapter"
    new_title = "Model A Roadster"
    old_stem = safe(old_title)

    _make_mp4(videos_dir / f"{old_stem}.mp4", size=50_000)  # below 100 KB

    with patch("subprocess.run") as mock_run:
        result = _rename_chapter_outputs(old_title, new_title, TEST_ARCHIVE)

    mock_run.assert_not_called()
    assert result == []


# ---------------------------------------------------------------------------
# temp dir rename
# ---------------------------------------------------------------------------


def test_temp_dir_is_renamed(videos_dir: Path) -> None:
    old_title = "Model A Chapter"
    new_title = "Model A Roadster"
    old_stem = safe(old_title)
    new_stem = safe(new_title)

    _make_mp4(videos_dir / f"{old_stem}.mp4")
    old_temp = videos_dir / f"{old_stem}_temp"
    old_temp.mkdir()
    (old_temp / "frame_001.jpg").write_bytes(b"fake")

    mock_result = MagicMock()
    mock_result.returncode = 0

    def fake_ffmpeg(*args, **kwargs):
        Path(args[0][-1]).write_bytes(b"\x00" * 200_000)
        return mock_result

    with patch("subprocess.run", side_effect=fake_ffmpeg):
        _rename_chapter_outputs(old_title, new_title, TEST_ARCHIVE)

    assert (videos_dir / f"{new_stem}_temp").exists()
    assert not old_temp.exists()


# ---------------------------------------------------------------------------
# no-op when source file missing
# ---------------------------------------------------------------------------


def test_returns_empty_when_no_mp4_found(videos_dir: Path) -> None:
    result = _rename_chapter_outputs("Ghost Chapter", "Ghost Roadster", TEST_ARCHIVE)
    assert result == []
