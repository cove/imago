from pathlib import Path
import shutil

from common import (
    get_gamma_profile_for_chapter,
    load_render_settings,
    save_render_settings,
    update_chapter_gamma_in_render_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def test_gamma_profile_loads_archive_ranges_clipped_to_chapter() -> None:
    archive = "__gamma_unit_archive"
    archive_meta_dir = ROOT / "metadata" / archive
    try:
        _path, settings = load_render_settings(archive, create=True)
        settings["archive_settings"]["gamma_correction_default"] = 1.0
        settings["archive_settings"]["gamma_correction_ranges"] = [
            {"start_frame": 100, "end_frame": 200, "gamma": 1.3},
            {"start_frame": 210, "end_frame": 260, "gamma": 1.6},
        ]
        save_render_settings(archive, settings)

        profile = get_gamma_profile_for_chapter(archive, ch_start=120, ch_end=230)
        assert profile["source"] == "archive"
        assert profile["ranges"] == [
            {"start_frame": 120, "end_frame": 200, "gamma": 1.3},
            {"start_frame": 210, "end_frame": 230, "gamma": 1.6},
        ]

        # Updating gamma for the chapter span replaces archive ranges within [120, 230)
        # and preserves those outside it.
        update_chapter_gamma_in_render_settings(
            archive,
            ch_start=120,
            ch_end=230,
            gamma_ranges=[
                {"start_frame": 150, "end_frame": 180, "gamma": 1.8},
                {"start_frame": 170, "end_frame": 190, "gamma": 1.5},
            ],
            default_gamma=1.0,
        )
        updated_profile = get_gamma_profile_for_chapter(
            archive, ch_start=120, ch_end=230
        )
        assert updated_profile["source"] == "archive"
        assert updated_profile["ranges"] == [
            {"start_frame": 150, "end_frame": 170, "gamma": 1.8},
            {"start_frame": 170, "end_frame": 190, "gamma": 1.5},
        ]
        # Archive ranges outside the chapter span are preserved
        full_profile = get_gamma_profile_for_chapter(archive)
        range_starts = {r["start_frame"] for r in full_profile["ranges"]}  # type: ignore[union-attr]
        assert 260 not in range_starts  # 230-260 portion of old [210-260] survives
        assert any(r["start_frame"] == 230 for r in full_profile["ranges"])  # type: ignore[union-attr]
    finally:
        shutil.rmtree(archive_meta_dir, ignore_errors=True)
