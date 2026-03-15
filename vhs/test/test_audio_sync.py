"""Tests for audio sync functionality in common.py.

Covers:
- _canonicalize_audio_sync_offsets (pure logic)
- _migrate_v2_to_v3 (pure logic)
- load_render_settings auto-migration v2 → v3
- get_audio_sync_offset_for_chapter (range lookup)
- update_chapter_audio_sync_in_render_settings (write/overwrite/clear)
- make_frame_accurate_extract_chapter audio filter strings
"""

import shutil
from pathlib import Path

from common import (
    _canonicalize_audio_sync_offsets,
    _migrate_v2_to_v3,
    get_audio_sync_offset_for_chapter,
    load_render_settings,
    make_frame_accurate_extract_chapter,
    save_render_settings,
    update_chapter_audio_sync_in_render_settings,
)


ROOT = Path(__file__).resolve().parents[1]
_DUMMY_PATH = Path("/dev/null/src.mkv")
_DUMMY_DEST = Path("/dev/null/out.mkv")


# ---------------------------------------------------------------------------
# _canonicalize_audio_sync_offsets
# ---------------------------------------------------------------------------


def test_canonicalize_empty() -> None:
    assert _canonicalize_audio_sync_offsets([]) == []
    assert _canonicalize_audio_sync_offsets(None) == []


def test_canonicalize_invalid_entries_ignored() -> None:
    raw = [
        {"start_frame": "bad", "end_frame": 100, "offset_seconds": 0.5},
        {"start_frame": 10, "end_frame": 5, "offset_seconds": 0.1},  # end <= start
        {"start_frame": 20, "end_frame": 30},  # missing offset_seconds
        None,
        42,
    ]
    assert _canonicalize_audio_sync_offsets(raw) == []


def test_canonicalize_sorts_by_start_frame() -> None:
    raw = [
        {"start_frame": 200, "end_frame": 300, "offset_seconds": 0.2},
        {"start_frame": 50, "end_frame": 100, "offset_seconds": 0.1},
    ]
    result = _canonicalize_audio_sync_offsets(raw)
    assert [r["start_frame"] for r in result] == [50, 200]


def test_canonicalize_merges_adjacent_same_offset() -> None:
    raw = [
        {"start_frame": 0, "end_frame": 100, "offset_seconds": 0.5},
        {"start_frame": 100, "end_frame": 200, "offset_seconds": 0.5},
    ]
    result = _canonicalize_audio_sync_offsets(raw)
    assert len(result) == 1
    assert result[0] == {"start_frame": 0, "end_frame": 200, "offset_seconds": 0.5}


def test_canonicalize_does_not_merge_different_offset() -> None:
    raw = [
        {"start_frame": 0, "end_frame": 100, "offset_seconds": 0.5},
        {"start_frame": 100, "end_frame": 200, "offset_seconds": 0.6},
    ]
    result = _canonicalize_audio_sync_offsets(raw)
    assert len(result) == 2


def test_canonicalize_rounds_offset_to_4dp() -> None:
    raw = [{"start_frame": 0, "end_frame": 100, "offset_seconds": 0.123456789}]
    result = _canonicalize_audio_sync_offsets(raw)
    assert result[0]["offset_seconds"] == round(0.123456789, 4)


# ---------------------------------------------------------------------------
# _migrate_v2_to_v3
# ---------------------------------------------------------------------------


def test_migrate_v2_to_v3_adds_audio_sync_offsets() -> None:
    v2 = {"version": 2, "bad_frames": [10, 20], "archive_settings": {}}
    v3 = _migrate_v2_to_v3(v2)
    assert v3["version"] == 3
    assert v3["audio_sync_offsets"] == []
    # Original keys preserved
    assert v3["bad_frames"] == [10, 20]


def test_migrate_v2_to_v3_preserves_existing_audio_sync_offsets() -> None:
    existing_offsets = [{"start_frame": 0, "end_frame": 100, "offset_seconds": 0.3}]
    v2 = {"version": 2, "audio_sync_offsets": existing_offsets}
    v3 = _migrate_v2_to_v3(v2)
    assert v3["audio_sync_offsets"] == existing_offsets


def test_migrate_v2_to_v3_does_not_mutate_input() -> None:
    v2 = {"version": 2}
    _migrate_v2_to_v3(v2)
    assert "audio_sync_offsets" not in v2


# ---------------------------------------------------------------------------
# load_render_settings: auto-migration v2 → v3
# ---------------------------------------------------------------------------


def test_load_render_settings_migrates_v2_to_v3() -> None:
    import json

    archive = "__audio_sync_unit_migrate"
    meta_dir = ROOT / "metadata" / archive
    try:
        meta_dir.mkdir(parents=True, exist_ok=True)
        settings_path = meta_dir / "render_settings.json"
        # Write a minimal v2 file (no audio_sync_offsets key)
        v2_data = {
            "version": 2,
            "bad_frames": [],
            "archive_settings": {"gamma_correction_default": 1.0, "gamma_correction_ranges": []},
        }
        settings_path.write_text(json.dumps(v2_data), encoding="utf-8")

        _path, loaded = load_render_settings(archive)
        assert loaded["version"] == 3
        assert "audio_sync_offsets" in loaded
        assert loaded["audio_sync_offsets"] == []
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# get_audio_sync_offset_for_chapter
# ---------------------------------------------------------------------------


def test_get_audio_sync_offset_returns_zero_when_no_entries() -> None:
    archive = "__audio_sync_unit_get"
    meta_dir = ROOT / "metadata" / archive
    try:
        _path, settings = load_render_settings(archive, create=True)
        save_render_settings(archive, settings)
        result = get_audio_sync_offset_for_chapter(archive, ch_start=0, ch_end=1000)
        assert result == 0.0
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


def test_get_audio_sync_offset_matches_by_midpoint() -> None:
    archive = "__audio_sync_unit_get2"
    meta_dir = ROOT / "metadata" / archive
    try:
        _path, settings = load_render_settings(archive, create=True)
        settings["audio_sync_offsets"] = [
            {"start_frame": 0, "end_frame": 500, "offset_seconds": 0.25},
            {"start_frame": 500, "end_frame": 1000, "offset_seconds": 0.75},
        ]
        save_render_settings(archive, settings)

        # Chapter [0, 400) → midpoint 200 → in [0, 500) → 0.25
        assert get_audio_sync_offset_for_chapter(archive, ch_start=0, ch_end=400) == 0.25
        # Chapter [600, 900) → midpoint 750 → in [500, 1000) → 0.75
        assert get_audio_sync_offset_for_chapter(archive, ch_start=600, ch_end=900) == 0.75
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


def test_get_audio_sync_offset_returns_zero_when_midpoint_not_covered() -> None:
    archive = "__audio_sync_unit_get3"
    meta_dir = ROOT / "metadata" / archive
    try:
        _path, settings = load_render_settings(archive, create=True)
        settings["audio_sync_offsets"] = [
            {"start_frame": 200, "end_frame": 400, "offset_seconds": 0.5},
        ]
        save_render_settings(archive, settings)
        # Chapter [0, 100) → midpoint 50 → not in [200, 400)
        assert get_audio_sync_offset_for_chapter(archive, ch_start=0, ch_end=100) == 0.0
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# update_chapter_audio_sync_in_render_settings
# ---------------------------------------------------------------------------


def test_update_audio_sync_writes_new_entry() -> None:
    archive = "__audio_sync_unit_upd1"
    meta_dir = ROOT / "metadata" / archive
    try:
        update_chapter_audio_sync_in_render_settings(archive, ch_start=100, ch_end=500, offset_seconds=0.3)
        offset = get_audio_sync_offset_for_chapter(archive, ch_start=100, ch_end=500)
        assert abs(offset - 0.3) < 1e-6
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


def test_update_audio_sync_overwrites_existing_entry() -> None:
    archive = "__audio_sync_unit_upd2"
    meta_dir = ROOT / "metadata" / archive
    try:
        update_chapter_audio_sync_in_render_settings(archive, ch_start=100, ch_end=500, offset_seconds=0.3)
        update_chapter_audio_sync_in_render_settings(archive, ch_start=100, ch_end=500, offset_seconds=-0.2)
        offset = get_audio_sync_offset_for_chapter(archive, ch_start=100, ch_end=500)
        assert abs(offset - (-0.2)) < 1e-6
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


def test_update_audio_sync_clears_entry_when_zero() -> None:
    archive = "__audio_sync_unit_upd3"
    meta_dir = ROOT / "metadata" / archive
    try:
        update_chapter_audio_sync_in_render_settings(archive, ch_start=100, ch_end=500, offset_seconds=0.3)
        update_chapter_audio_sync_in_render_settings(archive, ch_start=100, ch_end=500, offset_seconds=0.0)
        offset = get_audio_sync_offset_for_chapter(archive, ch_start=100, ch_end=500)
        assert offset == 0.0
        _, settings = load_render_settings(archive)
        assert settings["audio_sync_offsets"] == []
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


def test_update_audio_sync_preserves_entries_outside_chapter_span() -> None:
    archive = "__audio_sync_unit_upd4"
    meta_dir = ROOT / "metadata" / archive
    try:
        # Set up two entries: [0,200) and [800,1000)
        update_chapter_audio_sync_in_render_settings(archive, ch_start=0, ch_end=200, offset_seconds=0.1)
        update_chapter_audio_sync_in_render_settings(archive, ch_start=800, ch_end=1000, offset_seconds=0.9)
        # Update only [300,600) — others should survive
        update_chapter_audio_sync_in_render_settings(archive, ch_start=300, ch_end=600, offset_seconds=0.5)

        assert abs(get_audio_sync_offset_for_chapter(archive, ch_start=0, ch_end=200) - 0.1) < 1e-6
        assert abs(get_audio_sync_offset_for_chapter(archive, ch_start=300, ch_end=600) - 0.5) < 1e-6
        assert abs(get_audio_sync_offset_for_chapter(archive, ch_start=800, ch_end=1000) - 0.9) < 1e-6
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


def test_update_audio_sync_splits_overlapping_entry() -> None:
    archive = "__audio_sync_unit_upd5"
    meta_dir = ROOT / "metadata" / archive
    try:
        # One wide entry [0,1000)
        update_chapter_audio_sync_in_render_settings(archive, ch_start=0, ch_end=1000, offset_seconds=0.4)
        # Overwrite [300,700) with a different offset — [0,300) and [700,1000) should be preserved
        update_chapter_audio_sync_in_render_settings(archive, ch_start=300, ch_end=700, offset_seconds=0.8)

        _, settings = load_render_settings(archive)
        offsets = settings["audio_sync_offsets"]
        starts = {e["start_frame"]: e for e in offsets}
        assert 0 in starts and starts[0]["end_frame"] == 300
        assert 300 in starts and starts[300]["end_frame"] == 700
        assert 700 in starts and starts[700]["end_frame"] == 1000
        assert abs(starts[0]["offset_seconds"] - 0.4) < 1e-4
        assert abs(starts[300]["offset_seconds"] - 0.8) < 1e-4
        assert abs(starts[700]["offset_seconds"] - 0.4) < 1e-4
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# make_frame_accurate_extract_chapter: audio filter strings
# ---------------------------------------------------------------------------


def _get_af(cmd: list) -> str:
    idx = cmd.index("-af")
    return cmd[idx + 1]


def test_make_extract_zero_offset_baseline() -> None:
    cmd = make_frame_accurate_extract_chapter(
        src=_DUMMY_PATH,
        start=100.0,
        end=200.0,
        dest=_DUMMY_DEST,
        start_frame=3000,
        end_frame=6000,
        audio_offset_seconds=0.0,
    )
    af = _get_af(cmd)
    assert af.startswith("atrim=start=100.000000:end=200.000000")
    assert "adelay" not in af
    assert "apad=whole_dur=100.000000" in af


def test_make_extract_positive_offset_shifts_window() -> None:
    cmd = make_frame_accurate_extract_chapter(
        src=_DUMMY_PATH,
        start=100.0,
        end=200.0,
        dest=_DUMMY_DEST,
        start_frame=3000,
        end_frame=6000,
        audio_offset_seconds=0.5,
    )
    af = _get_af(cmd)
    assert af.startswith("atrim=start=100.500000:end=200.500000")
    assert "adelay" not in af


def test_make_extract_negative_offset_stays_above_zero() -> None:
    # start=10, offset=-2 → audio_start_raw=8 ≥ 0 → no adelay
    cmd = make_frame_accurate_extract_chapter(
        src=_DUMMY_PATH,
        start=10.0,
        end=20.0,
        dest=_DUMMY_DEST,
        start_frame=300,
        end_frame=600,
        audio_offset_seconds=-2.0,
    )
    af = _get_af(cmd)
    assert af.startswith("atrim=start=8.000000:end=18.000000")
    assert "adelay" not in af


def test_make_extract_negative_offset_below_zero_adds_adelay() -> None:
    # start=3, offset=-5 → audio_start_raw=-2 → silence_prepend=2s → adelay=2000ms
    cmd = make_frame_accurate_extract_chapter(
        src=_DUMMY_PATH,
        start=3.0,
        end=13.0,
        dest=_DUMMY_DEST,
        start_frame=90,
        end_frame=390,
        audio_offset_seconds=-5.0,
    )
    af = _get_af(cmd)
    assert "atrim=start=0.000000" in af
    assert "adelay=2000.0:all=1" in af
    assert "apad=whole_dur=10.000000" in af


def test_make_extract_large_negative_offset_full_silence() -> None:
    # start=1, offset=-10 → audio_start_raw=-9 → audio_end_raw=-9+10=1-10+10=1-9=-9 → end=0
    # Wait: start=1, end=11, offset=-10 → audio_start_raw=-9, audio_end_raw=1
    # atrim=start=0:end=1, adelay=9000ms
    cmd = make_frame_accurate_extract_chapter(
        src=_DUMMY_PATH,
        start=1.0,
        end=11.0,
        dest=_DUMMY_DEST,
        start_frame=30,
        end_frame=330,
        audio_offset_seconds=-10.0,
    )
    af = _get_af(cmd)
    assert "atrim=start=0.000000" in af
    assert "adelay=9000.0:all=1" in af
