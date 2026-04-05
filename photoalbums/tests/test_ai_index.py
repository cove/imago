import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from cast.storage import TextFaceStore
from photoalbums.lib import (
    ai_index,
    ai_index_analysis,
    ai_index_args,
    ai_index_engine_cache,
    ai_index_runner,
    ai_index_scan,
)
from photoalbums.lib import xmp_sidecar


class TestAIIndex(unittest.TestCase):
    def _valid_sidecar_text(self) -> str:
        return "x" * (ai_index.MIN_EXISTING_SIDECAR_BYTES + 1)

    @contextmanager
    def _mock_layout(self, image: Path):
        yield SimpleNamespace(
            kind="single_image",
            page_like=False,
            split_mode="manual",
            content_bounds=SimpleNamespace(as_dict=lambda: {"x": 0, "y": 0, "width": 0, "height": 0}),
            split_applied=False,
            fallback_used=False,
            subphotos=[],
            content_path=image,
        )

    def test_discover_images_archive_and_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Family_Archive"
            view = base / "Family_View"
            archive.mkdir()
            view.mkdir()
            (archive / "a.jpg").write_bytes(b"a")
            (view / "b.png").write_bytes(b"b")
            (base / "other.jpg").write_bytes(b"c")

            files = ai_index.discover_images(
                base,
                include_archive=True,
                include_view=False,
                extensions={".jpg", ".png"},
            )
            self.assertEqual([p.name for p in files], ["a.jpg"])

            files = ai_index.discover_images(
                base,
                include_archive=False,
                include_view=True,
                extensions={".jpg", ".png"},
            )
            self.assertEqual([p.name for p in files], ["b.png"])

    def test_coalesce_archive_processing_files_keeps_s01_and_preserves_derived_images(self):
        files = [
            Path("China_1986_B02_Archive/China_1986_B02_P02_S02.tif"),
            Path("China_1986_B02_Archive/China_1986_B02_P02_S01.tif"),
            Path("China_1986_B02_Archive/China_1986_B02_P02_D01-01.tif"),
            Path("China_1986_B02_View/China_1986_B02_P02_V.jpg"),
        ]

        selected = ai_index._coalesce_archive_processing_files(files)

        self.assertEqual(
            selected,
            [
                Path("China_1986_B02_Archive/China_1986_B02_P02_S01.tif"),
                Path("China_1986_B02_Archive/China_1986_B02_P02_D01-01.tif"),
                Path("China_1986_B02_View/China_1986_B02_P02_V.jpg"),
            ],
        )

    def test_coalesce_archive_processing_files_fails_when_s01_missing(self):
        files = [
            Path("China_1986_B02_Archive/China_1986_B02_P02_S02.tif"),
            Path("China_1986_B02_Archive/China_1986_B02_P02_S03.tif"),
        ]

        with self.assertRaises(RuntimeError):
            ai_index._coalesce_archive_processing_files(files)

    def test_display_work_label_uses_page_name_for_archive_scan(self):
        self.assertEqual(
            ai_index._display_work_label(Path("China_1986_B02_Archive/China_1986_B02_P03_S01.tif")),
            "China_1986_B02_P03",
        )
        self.assertEqual(
            ai_index._display_work_label(Path("China_1986_B02_Archive/China_1986_B02_P03_D01-01.tif")),
            "China_1986_B02_P03_D01-01.tif",
        )

    def test_append_xmp_job_artifact_records_page_sidecar_paths_for_scan_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P03_S01.tif"
            scan2 = archive / "China_1986_B02_P03_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")

            with mock.patch.object(ai_index, "append_job_artifact") as append_mock:
                ai_index._append_xmp_job_artifact(scan1, scan1.with_suffix(".xmp"))

            payload = append_mock.call_args.args[0]
            self.assertEqual(payload["label"], "China_1986_B02_P03")
            self.assertEqual(
                payload["sidecar_paths"],
                [
                    str(scan1.with_suffix(".xmp")),
                    str(scan2.with_suffix(".xmp")),
                ],
            )

    def test_filter_files_by_tree_keeps_archive_only_when_view_not_requested(self):
        files = [
            Path("China_1986_B02_Archive/China_1986_B02_P01_S01.tif"),
            Path("China_1986_B02_View/China_1986_B02_P01_V.jpg"),
        ]

        filtered = ai_index._filter_files_by_tree(files, include_archive=True, include_view=False)

        self.assertEqual(filtered, [Path("China_1986_B02_Archive/China_1986_B02_P01_S01.tif")])

    def test_expand_album_title_dependencies_prepends_title_page_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "EasternEuropeSpainMorocco_1988_B00_Archive"
            archive.mkdir()
            for name in (
                "EasternEuropeSpainMorocco_1988_B00_P01_D01-01.tif",
                "EasternEuropeSpainMorocco_1988_B00_P01_D01-02.tif",
                "EasternEuropeSpainMorocco_1988_B00_P01_S01.tif",
                "EasternEuropeSpainMorocco_1988_B00_P02_S01.tif",
            ):
                (archive / name).write_bytes(b"x")

            files = ai_index._expand_album_title_dependencies(
                [
                    archive / "EasternEuropeSpainMorocco_1988_B00_P01_D01-01.tif",
                    archive / "EasternEuropeSpainMorocco_1988_B00_P01_D01-02.tif",
                    archive / "EasternEuropeSpainMorocco_1988_B00_P01_S01.tif",
                    archive / "EasternEuropeSpainMorocco_1988_B00_P02_S01.tif",
                ],
                {".jpg", ".tif"},
            )

            self.assertEqual(
                [p.name for p in files],
                [
                    "EasternEuropeSpainMorocco_1988_B00_P01_S01.tif",
                    "EasternEuropeSpainMorocco_1988_B00_P01_D01-01.tif",
                    "EasternEuropeSpainMorocco_1988_B00_P01_D01-02.tif",
                    "EasternEuropeSpainMorocco_1988_B00_P02_S01.tif",
                ],
            )

    def test_resolve_xmp_text_layers_does_not_fallback_from_ocr_text(self):
        layers = ai_index._resolve_xmp_text_layers(
            image_path=Path("China_1986_B02_P02_stitched.jpg"),
            ocr_text="TEMPLE OF HEAVEN",
            page_like=True,
        )
        self.assertEqual(layers["author_text"], "")
        self.assertEqual(layers["scene_text"], "")

    def test_looks_like_album_title_page_requires_album_style_name(self):
        self.assertFalse(ai_index._looks_like_album_title_page(Path("a.jpg")))
        self.assertFalse(ai_index._looks_like_album_title_page(Path("China_1986_B02_P00.jpg")))
        self.assertTrue(ai_index._looks_like_album_title_page(Path("China_1986_B02_P01_S01.tif")))

    def test_is_album_title_source_candidate_rejects_detail_views_and_second_scans(self):
        self.assertFalse(ai_index._is_album_title_source_candidate(Path("China_1986_B02_P01_D01-01_V.jpg")))
        self.assertFalse(ai_index._is_album_title_source_candidate(Path("China_1986_B02_P01_S02.tif")))
        self.assertTrue(ai_index._is_album_title_source_candidate(Path("China_1986_B02_P01_S01.tif")))

    def test_build_dc_source_uses_all_page_scans_for_raw_scan_sidecar(self):
        source = ai_index._build_dc_source(
            "Eastern Europe Spain and Morocco 1988",
            Path("EasternEuropeSpainMorocco_1988_B00_P34_S01.tif"),
            [
                "EasternEuropeSpainMorocco_1988_B00_P34_S01.tif",
                "EasternEuropeSpainMorocco_1988_B00_P34_S02.tif",
            ],
        )
        self.assertEqual(
            source,
            (
                "Eastern Europe Spain and Morocco 1988 Page 34 "
                "Scan(s) S01 S02; EasternEuropeSpainMorocco_1988_B00_P34_S01.tif; "
                "EasternEuropeSpainMorocco_1988_B00_P34_S02.tif"
            ),
        )

    def test_page_scan_filenames_uses_archive_scans_for_stitched_view_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            image = view / "Egypt_1975_B00_P39_stitched.jpg"
            image.write_bytes(b"x")
            (archive / "Egypt_1975_B00_P39_S01.tif").write_bytes(b"a")
            (archive / "Egypt_1975_B00_P39_S02.tif").write_bytes(b"b")

            self.assertEqual(
                ai_index._page_scan_filenames(image),
                ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
            )

    def test_page_scan_filenames_uses_archive_scan_for_title_view_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            image = view / "Egypt_1975_B00_P01.jpg"
            image.write_bytes(b"x")
            (archive / "Egypt_1975_B00_P01_S01.tif").write_bytes(b"a")

            self.assertEqual(
                ai_index._page_scan_filenames(image),
                ["Egypt_1975_B00_P01_S01.tif"],
            )

    def test_page_scan_filenames_uses_archive_scans_for_derived_view_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            image = view / "Egypt_1975_B00_P39_D01-01_V.jpg"
            image.write_bytes(b"x")
            (archive / "Egypt_1975_B00_P39_S01.tif").write_bytes(b"a")
            (archive / "Egypt_1975_B00_P39_S02.tif").write_bytes(b"b")

            self.assertEqual(
                ai_index._page_scan_filenames(image),
                ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
            )

    def test_page_scan_filenames_handles_crowded_archive_and_view_derived_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Family_1907-1946_B01_Archive"
            view = base / "Family_1907-1946_B01_View"
            archive.mkdir()
            view.mkdir()
            archive_image = archive / "Family_1907-1946_B01_P03_D02-03.tif"
            view_image = view / "Family_1907-1946_B01_P03_D02-03_V.jpg"
            for path in (
                archive_image,
                view_image,
                archive / "Family_1907-1946_B01_P03_D01-01.tif",
                archive / "Family_1907-1946_B01_P03_D01-02.tif",
                archive / "Family_1907-1946_B01_P03_D02-01.tif",
                archive / "Family_1907-1946_B01_P03_D02-02.tif",
                archive / "Family_1907-1946_B01_P03_D03-01.tif",
                view / "Family_1907-1946_B01_P03_D01-01_V.jpg",
                view / "Family_1907-1946_B01_P03_D01-02_V.jpg",
                view / "Family_1907-1946_B01_P03_D02-01_V.jpg",
                view / "Family_1907-1946_B01_P03_D02-02_V.jpg",
                view / "Family_1907-1946_B01_P03_D03-01_V.jpg",
            ):
                path.write_bytes(b"x")
            (archive / "Family_1907-1946_B01_P03_S01.tif").write_bytes(b"a")
            (archive / "Family_1907-1946_B01_P03_S02.tif").write_bytes(b"b")

            self.assertEqual(
                ai_index._page_scan_filenames(archive_image),
                ["Family_1907-1946_B01_P03_S01.tif", "Family_1907-1946_B01_P03_S02.tif"],
            )
            self.assertEqual(
                ai_index._page_scan_filenames(view_image),
                ["Family_1907-1946_B01_P03_S01.tif", "Family_1907-1946_B01_P03_S02.tif"],
            )

    def test_page_scan_filenames_handles_title_and_derived_pages_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            title_image = view / "Egypt_1975_B00_P01.jpg"
            derived_image = view / "Egypt_1975_B00_P39_D01-01_V.jpg"
            title_image.write_bytes(b"title")
            derived_image.write_bytes(b"derived")
            (archive / "Egypt_1975_B00_P01_S01.tif").write_bytes(b"a")
            (archive / "Egypt_1975_B00_P39_S01.tif").write_bytes(b"b")
            (archive / "Egypt_1975_B00_P39_S02.tif").write_bytes(b"c")

            self.assertEqual(
                ai_index._page_scan_filenames(title_image),
                ["Egypt_1975_B00_P01_S01.tif"],
            )
            self.assertEqual(
                ai_index._page_scan_filenames(derived_image),
                ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
            )

    def test_page_scan_filenames_uses_singleton_archive_scan_for_base_view_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            image = view / "Egypt_1975_B00_P05.jpg"
            image.write_bytes(b"x")
            (archive / "Egypt_1975_B00_P05_S01.tif").write_bytes(b"a")

            self.assertEqual(
                ai_index._page_scan_filenames(image),
                ["Egypt_1975_B00_P05_S01.tif"],
            )

    def test_page_scan_filenames_returns_sibling_scans_for_raw_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Egypt_1975_B00_Archive"
            archive.mkdir()
            scan1 = archive / "Egypt_1975_B00_P39_S01.tif"
            scan2 = archive / "Egypt_1975_B00_P39_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")

            self.assertEqual(
                ai_index._page_scan_filenames(scan1),
                ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
            )

    def test_page_scan_filenames_returns_empty_when_archive_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp) / "Egypt_1975_B00_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P39_stitched.jpg"
            image.write_bytes(b"x")

            self.assertEqual(ai_index._page_scan_filenames(image), [])

    def test_page_scan_filenames_uses_archive_scan_for_v_view_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "England_1983_B02_Archive"
            view = base / "England_1983_B02_View"
            archive.mkdir()
            view.mkdir()
            image = view / "England_1983_B02_P47_V.jpg"
            image.write_bytes(b"x")
            (archive / "England_1983_B02_P47_S01.tif").write_bytes(b"a")

            self.assertEqual(
                ai_index._page_scan_filenames(image),
                ["England_1983_B02_P47_S01.tif"],
            )

    def test_page_scan_filenames_uses_archive_scans_for_multi_scan_view_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            image = view / "Egypt_1975_B00_P39_V.jpg"
            image.write_bytes(b"x")
            (archive / "Egypt_1975_B00_P39_S01.tif").write_bytes(b"a")
            (archive / "Egypt_1975_B00_P39_S02.tif").write_bytes(b"b")

            self.assertEqual(
                ai_index._page_scan_filenames(image),
                ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
            )

    def test_page_scan_filenames_uses_archive_scans_for_legacy_vc_view_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            image = view / "Egypt_1975_B00_P39_VC.jpg"
            image.write_bytes(b"x")
            (archive / "Egypt_1975_B00_P39_S01.tif").write_bytes(b"a")
            (archive / "Egypt_1975_B00_P39_S02.tif").write_bytes(b"b")

            self.assertEqual(
                ai_index._page_scan_filenames(image),
                ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
            )

    def test_page_scan_filenames_returns_empty_for_unparseable_non_scan_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            image = view / "cover.jpg"
            image.write_bytes(b"x")
            (archive / "Egypt_1975_B00_P39_S01.tif").write_bytes(b"a")

            self.assertEqual(ai_index._page_scan_filenames(image), [])

    def test_build_dc_source_uses_archive_scans_for_stitched_view_page(self):
        source = ai_index._build_dc_source(
            "EUROPE 1973 EGYPT 1975",
            Path("Egypt_1975_B00_P39_stitched.jpg"),
            ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
        )
        self.assertEqual(
            source,
            ("EUROPE 1973 EGYPT 1975 Page 39 Scan(s) S01 S02; Egypt_1975_B00_P39_S01.tif; Egypt_1975_B00_P39_S02.tif"),
        )

    def test_build_dc_source_uses_archive_scans_for_multi_scan_view_page(self):
        source = ai_index._build_dc_source(
            "EUROPE 1973 EGYPT 1975",
            Path("Egypt_1975_B00_P39_V.jpg"),
            ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
        )
        self.assertEqual(
            source,
            ("EUROPE 1973 EGYPT 1975 Page 39 Scan(s) S01 S02; Egypt_1975_B00_P39_S01.tif; Egypt_1975_B00_P39_S02.tif"),
        )

    def test_build_dc_source_uses_archive_scan_for_v_view_page(self):
        source = ai_index._build_dc_source(
            "ENGLAND BOOK 11 1983",
            Path("England_1983_B02_P47_V.jpg"),
            ["England_1983_B02_P47_S01.tif"],
        )
        self.assertEqual(
            source,
            "ENGLAND BOOK 11 1983 Page 47 Scan(s) S01; England_1983_B02_P47_S01.tif",
        )

    def test_build_dc_source_with_no_scan_filenames_omits_file_reference(self):
        # No archive TIFs found — honest result includes title + page but no filename.
        source = ai_index._build_dc_source(
            "ENGLAND BOOK 11 1983",
            Path("England_1983_B02_P47_V.jpg"),
            [],
        )
        self.assertEqual(source, "ENGLAND BOOK 11 1983 Page 47")

    def test_build_dc_source_with_empty_album_title(self):
        source = ai_index._build_dc_source(
            "",
            Path("Egypt_1975_B00_P39_V.jpg"),
            ["Egypt_1975_B00_P39_S01.tif", "Egypt_1975_B00_P39_S02.tif"],
        )
        self.assertEqual(
            source,
            "Page 39 Scan(s) S01 S02; Egypt_1975_B00_P39_S01.tif; Egypt_1975_B00_P39_S02.tif",
        )

    def test_compute_xmp_title_ignores_scene_text_by_default(self):
        layers = ai_index._resolve_xmp_text_layers(
            image_path=Path("Family_1986_B01_P02.jpg"),
            ocr_text="NO SMOKING",
            page_like=False,
        )
        title, title_source = ai_index._compute_xmp_title(
            image_path=Path("Family_1986_B01_P02.jpg"),
            explicit_title="",
            author_text=layers["author_text"],
        )
        self.assertEqual(title, "")
        self.assertEqual(title_source, "")

    def test_compute_xmp_title_preserves_explicit_title(self):
        title, title_source = ai_index._compute_xmp_title(
            image_path=Path("China_1986_B02_P01.jpg"),
            explicit_title="MAINLAND CHINA 1986 BOOK 11",
            title_source="author_text",
            author_text="MAINLAND CHINA 1986 BOOK 11",
        )
        self.assertEqual(title, "MAINLAND CHINA 1986 BOOK 11")
        self.assertEqual(title_source, "author_text")

    def test_dc_date_needs_refresh_only_when_missing_and_unattempted(self):
        image = Path("Egypt_1975_B00_P02_S01.tif")
        state = {
            "dc_date": "",
            "date_time_original": "",
            "ocr_text": "March 15, 1975",
            "album_title": "Egypt 1975",
            "date_estimate_input_hash": "",
        }
        self.assertTrue(ai_index._dc_date_needs_refresh(image, state, enabled=True))

        state["date_estimate_input_hash"] = ai_index._date_estimate_input_hash(
            str(state["ocr_text"]),
            str(state["album_title"]),
        )
        self.assertFalse(ai_index._dc_date_needs_refresh(image, state, enabled=True))

        state["dc_date"] = "1975-03-15"
        self.assertTrue(ai_index._dc_date_needs_refresh(image, state, enabled=True))
        state["date_time_original"] = "1975-03-15T12:00:00"
        self.assertFalse(ai_index._dc_date_needs_refresh(image, state, enabled=True))
        self.assertFalse(ai_index._dc_date_needs_refresh(image, state, enabled=False))

    def test_resolve_dc_date_preserves_existing_multiple_values(self):
        self.assertEqual(
            ai_index._resolve_dc_date(
                existing_dc_date=["1934-09", "1934-10"],
                ocr_text="September 1934 and October 1934",
                album_title="Family 1934",
                image_path=Path("Family_1934_B01_P02_S01.tif"),
                date_engine=None,
                prompt_debug=None,
            ),
            ["1934-09", "1934-10"],
        )

    def test_effective_sidecar_ocr_text_uses_source_scan_sidecar_for_derived_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            scan1 = archive / "Egypt_1975_B00_P02_S01.tif"
            scan2 = archive / "Egypt_1975_B00_P02_S02.tif"
            derived = view / "Egypt_1975_B00_P02_D01-01_V.jpg"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            derived.write_bytes(b"view")
            xmp_sidecar.write_xmp_sidecar(
                scan1.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Scan sidecar",
                album_title="Egypt 1975",
                source_text=ai_index._build_dc_source("Egypt 1975", scan1, [scan1.name]),
                ocr_text="March 1975",
                detections_payload={"people": [], "objects": [], "ocr": {}, "caption": {}},
                subphotos=[],
            )

            text = ai_index._effective_sidecar_ocr_text(
                derived,
                {
                    "source_text": ai_index._build_dc_source("Egypt 1975", derived, [scan1.name, scan2.name]),
                    "ocr_text": "WRONG OCR",
                },
            )

            self.assertEqual(text, "March 1975")

    def test_effective_sidecar_location_payload_uses_source_scan_sidecar_for_derived_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            scan1 = archive / "Egypt_1975_B00_P02_S01.tif"
            scan2 = archive / "Egypt_1975_B00_P02_S02.tif"
            derived = view / "Egypt_1975_B00_P02_D01-01_V.jpg"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            derived.write_bytes(b"view")
            xmp_sidecar.write_xmp_sidecar(
                scan1.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Scan sidecar",
                album_title="Egypt 1975",
                gps_latitude="39.7875",
                gps_longitude="100.307222",
                location_city="Dunhuang",
                location_state="Gansu",
                location_country="China",
                source_text=ai_index._build_dc_source("Egypt 1975", scan1, [scan1.name]),
                ocr_text="March 1975",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                    "location": {
                        "gps_latitude": 39.7875,
                        "gps_longitude": 100.307222,
                        "city": "Dunhuang",
                        "state": "Gansu",
                        "country": "China",
                    },
                },
                subphotos=[],
            )

            location = ai_index._effective_sidecar_location_payload(
                derived,
                {
                    "source_text": ai_index._build_dc_source("Egypt 1975", derived, [scan1.name, scan2.name]),
                    "gps_latitude": "0,0N",
                    "gps_longitude": "0,0E",
                    "detections": {
                        "location": {
                            "gps_latitude": 0,
                            "gps_longitude": 0,
                            "city": "Wrong",
                        }
                    },
                },
            )

            self.assertEqual(location["gps_latitude"], 39.7875)
            self.assertEqual(location["gps_longitude"], 100.307222)
            self.assertEqual(location["city"], "Dunhuang")
            self.assertEqual(location["state"], "Gansu")
            self.assertEqual(location["country"], "China")

    def test_sidecar_has_people_to_refresh_ignores_caption_people_estimate_without_faces(self):
        state = {
            "people_detected": False,
            "people_identified": False,
            "detections": {
                "people": [],
                "caption": {
                    "people_present": True,
                    "estimated_people_count": 1,
                },
            },
        }
        self.assertFalse(ai_index._sidecar_has_people_to_refresh(state))

    def test_build_caption_metadata_preserves_engine_error_verbatim(self):
        payload = ai_index._build_caption_metadata(
            requested_engine="lmstudio",
            effective_engine="lmstudio",
            fallback=True,
            error="caption fallback",
            engine_error=(
                'LM Studio request failed: {"error":"request (4196 tokens) exceeds the available context size '
                '(4096 tokens), try increasing it"}'
            ),
            model="zai-org/glm-4.6v-flash",
        )
        self.assertEqual(payload["error"], "caption fallback")
        self.assertEqual(
            payload["engine_error"],
            'LM Studio request failed: {"error":"request (4196 tokens) exceeds the available context size '
            '(4096 tokens), try increasing it"}',
        )

    def test_needs_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")
            stat = image.stat()
            row = {
                "image_path": str(image),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sidecar_path": str(sidecar),
                "processor_signature": ai_index.PROCESSOR_SIGNATURE,
            }
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

            next_ns = max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            os.utime(image, ns=(next_ns, next_ns))
            self.assertTrue(ai_index.needs_processing(image, row, force=False))

    def test_run_refreshes_current_xmp_when_dc_date_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            archive.mkdir()
            image = archive / "Egypt_1975_B00_P02_S01.tif"
            image.write_bytes(b"scan")
            stat = image.stat()
            detections = {
                "people": [],
                "objects": [],
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "lmstudio",
                    "fallback": False,
                    "error": "",
                    "model": "test-model",
                },
                "processing": {
                    "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                    "settings_signature": "",
                    "cast_store_signature": "",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "analysis_mode": "single_image",
                },
            }
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Travel photo",
                album_title="Egypt 1975",
                source_text=ai_index._build_dc_source("Egypt 1975", image, [image.name]),
                ocr_text="March 15, 1975",
                detections_payload=detections,
                subphotos=[],
            )

            fake_date_engine = SimpleNamespace(
                estimate=mock.Mock(
                    return_value=SimpleNamespace(
                        date="1975-03-15",
                        error="",
                        fallback=False,
                    )
                )
            )

            with (
                mock.patch.object(ai_index_runner, "_init_date_engine", return_value=fake_date_engine),
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            write_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["dc_date"], "1975-03-15")
            self.assertEqual(write_mock.call_args.kwargs["date_time_original"], "1975-03-15T12:00:00")
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["processing"]["date_estimate_input_hash"],
                ai_index._date_estimate_input_hash("March 15, 1975", "Egypt 1975"),
            )

    def test_run_refreshes_current_xmp_when_timeline_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            archive.mkdir()
            image = archive / "Egypt_1975_B00_P02_S01.tif"
            image.write_bytes(b"scan")
            stat = image.stat()
            detections = {
                "people": [],
                "objects": [],
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "lmstudio",
                    "fallback": False,
                    "error": "",
                    "model": "test-model",
                },
                "processing": {
                    "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                    "settings_signature": "",
                    "cast_store_signature": "",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "date_estimate_input_hash": ai_index._date_estimate_input_hash("March 1975", "Egypt 1975"),
                    "analysis_mode": "single_image",
                },
            }
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Travel photo",
                album_title="Egypt 1975",
                source_text=ai_index._build_dc_source("Egypt 1975", image, [image.name]),
                ocr_text="March 1975",
                dc_date="1975-03",
                detections_payload=detections,
                subphotos=[],
            )

            sidecar_text = image.with_suffix(".xmp").read_text(encoding="utf-8")
            sidecar_text = sidecar_text.replace(
                "<exif:DateTimeOriginal>1975-03-15T12:00:00</exif:DateTimeOriginal>\n",
                "",
            )
            image.with_suffix(".xmp").write_text(sidecar_text, encoding="utf-8")

            with (
                mock.patch.object(ai_index_runner, "_init_date_engine") as date_engine_mock,
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            date_engine_mock.assert_not_called()
            analysis_mock.assert_not_called()
            write_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["dc_date"], "1975-03")
            self.assertEqual(write_mock.call_args.kwargs["date_time_original"], "1975-03-15T12:00:00")

    def test_run_refreshes_derived_view_xmp_using_source_scan_ocr_text_for_dc_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            view = base / "Egypt_1975_B00_View"
            archive.mkdir()
            view.mkdir()
            scan1 = archive / "Egypt_1975_B00_P02_S01.tif"
            scan2 = archive / "Egypt_1975_B00_P02_S02.tif"
            image = view / "Egypt_1975_B00_P02_D01-01_V.jpg"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            image.write_bytes(b"derived")
            scan1_stat = scan1.stat()
            xmp_sidecar.write_xmp_sidecar(
                scan1.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Source scan sidecar",
                album_title="Egypt 1975",
                gps_latitude="39.7875",
                gps_longitude="100.307222",
                location_city="Dunhuang",
                location_state="Gansu",
                location_country="China",
                source_text=ai_index._build_dc_source("Egypt 1975", scan1, [scan1.name]),
                ocr_text="March 1975",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "test-model",
                    },
                    "processing": {
                        "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                        "settings_signature": "",
                        "cast_store_signature": "",
                        "size": int(scan1_stat.st_size),
                        "mtime_ns": int(scan1_stat.st_mtime_ns),
                        "analysis_mode": "single_image",
                    },
                    "location": {
                        "gps_latitude": 39.7875,
                        "gps_longitude": 100.307222,
                        "city": "Dunhuang",
                        "state": "Gansu",
                        "country": "China",
                    },
                },
                subphotos=[],
            )
            stat = image.stat()
            detections = {
                "people": [],
                "objects": [],
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "lmstudio",
                    "fallback": False,
                    "error": "",
                    "model": "test-model",
                },
                "processing": {
                    "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                    "settings_signature": "",
                    "cast_store_signature": "",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "analysis_mode": "single_image",
                },
                "location": {
                    "gps_latitude": 0,
                    "gps_longitude": 0,
                    "city": "Wrong",
                    "state": "Wrong",
                    "country": "Wrong",
                },
            }
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt", "derived"],
                description="Derived image",
                album_title="Egypt 1975",
                gps_latitude="0",
                gps_longitude="0",
                location_city="Wrong",
                location_state="Wrong",
                location_country="Wrong",
                source_text=ai_index._build_dc_source("Egypt 1975", image, [scan1.name, scan2.name]),
                ocr_text="WRONG OCR",
                detections_payload=detections,
                subphotos=[],
            )

            fake_date_engine = SimpleNamespace(
                estimate=mock.Mock(
                    return_value=SimpleNamespace(
                        date="1975-03",
                        error="",
                        fallback=False,
                    )
                )
            )

            with (
                mock.patch.object(ai_index_runner, "_init_date_engine", return_value=fake_date_engine),
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            fake_date_engine.estimate.assert_called_once()
            self.assertEqual(fake_date_engine.estimate.call_args.kwargs["ocr_text"], "March 1975")
            write_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["ocr_text"], "March 1975")
            self.assertEqual(write_mock.call_args.kwargs["dc_date"], "1975-03")
            self.assertEqual(write_mock.call_args.kwargs["date_time_original"], "1975-03-15T12:00:00")
            self.assertEqual(write_mock.call_args.kwargs["gps_latitude"], "39.7875")
            self.assertEqual(write_mock.call_args.kwargs["gps_longitude"], "100.307222")
            self.assertEqual(write_mock.call_args.kwargs["location_city"], "Dunhuang")
            self.assertEqual(write_mock.call_args.kwargs["location_state"], "Gansu")
            self.assertEqual(write_mock.call_args.kwargs["location_country"], "China")
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["location"]["gps_latitude"],
                39.7875,
            )
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["location"]["gps_longitude"],
                100.307222,
            )
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["location"]["city"],
                "Dunhuang",
            )
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["processing"]["date_estimate_input_hash"],
                ai_index._date_estimate_input_hash("March 1975", "Egypt 1975"),
            )

    def test_run_skips_people_refresh_when_cast_changes_but_no_faces_were_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            archive.mkdir()
            image = archive / "Egypt_1975_B00_P02_S01.tif"
            image.write_bytes(b"scan")
            stat = image.stat()
            detections = {
                "people": [],
                "objects": [],
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "lmstudio",
                    "fallback": False,
                    "error": "",
                    "model": "test-model",
                    "people_present": True,
                    "estimated_people_count": 1,
                },
                "processing": {
                    "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                    "settings_signature": "",
                    "cast_store_signature": "old-sig",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "date_estimate_input_hash": ai_index._date_estimate_input_hash("March 1975", "Egypt 1975"),
                    "analysis_mode": "single_image",
                },
            }
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Travel photo",
                album_title="Egypt 1975",
                source_text=ai_index._build_dc_source("Egypt 1975", image, [image.name]),
                ocr_text="March 1975",
                dc_date="1975-03",
                detections_payload=detections,
                subphotos=[],
                people_detected=False,
                people_identified=False,
            )

            sidecar_text = image.with_suffix(".xmp").read_text(encoding="utf-8")
            sidecar_text = sidecar_text.replace(
                "<exif:DateTimeOriginal>1975-03-15T12:00:00</exif:DateTimeOriginal>\n",
                "",
            )
            image.with_suffix(".xmp").write_text(sidecar_text, encoding="utf-8")

            fake_matcher = mock.Mock()
            fake_matcher.store_signature.return_value = "new-sig"

            with (
                mock.patch.object(ai_index_runner, "_init_people_matcher", return_value=fake_matcher),
                mock.patch.object(ai_index_runner, "_init_caption_engine") as caption_engine_mock,
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            caption_engine_mock.assert_not_called()
            analysis_mock.assert_not_called()
            fake_matcher.match_image.assert_not_called()
            write_mock.assert_called_once()
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["processing"]["cast_store_signature"],
                "new-sig",
            )
            self.assertEqual(write_mock.call_args.kwargs["date_time_original"], "1975-03-15T12:00:00")

    def test_run_people_update_only_skips_caption_when_people_names_do_not_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Egypt_1975_B00_Archive"
            archive.mkdir()
            image = archive / "Egypt_1975_B00_P09_S01.tif"
            image.write_bytes(b"scan")
            stat = image.stat()
            detections = {
                "people": [],
                "objects": [],
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "lmstudio",
                    "fallback": False,
                    "error": "",
                    "model": "test-model",
                    "people_present": True,
                    "estimated_people_count": 9,
                },
                "processing": {
                    "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                    "settings_signature": "",
                    "cast_store_signature": "old-sig",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "date_estimate_input_hash": "",
                    "analysis_mode": "single_image",
                },
            }
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Travel photo",
                album_title="Egypt 1975",
                source_text=ai_index._build_dc_source("Egypt 1975", image, [image.name]),
                ocr_text="March 1975",
                dc_date="1975-03",
                detections_payload=detections,
                subphotos=[],
                people_detected=True,
                people_identified=False,
            )

            sidecar_text = image.with_suffix(".xmp").read_text(encoding="utf-8")
            sidecar_text = sidecar_text.replace(
                "<exif:DateTimeOriginal>1975-03-15T12:00:00</exif:DateTimeOriginal>\n",
                "",
            )
            image.with_suffix(".xmp").write_text(sidecar_text, encoding="utf-8")

            fake_matcher = mock.Mock()
            fake_matcher.store_signature.return_value = "new-sig"
            fake_matcher.match_image.return_value = []
            fake_matcher.last_faces_detected = 0

            with (
                mock.patch.object(ai_index_runner, "_init_people_matcher", return_value=fake_matcher),
                mock.patch.object(ai_index_runner, "_init_caption_engine") as caption_engine_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            fake_matcher.match_image.assert_called_once()
            caption_engine_mock.assert_not_called()
            write_mock.assert_called_once()
            self.assertFalse(write_mock.call_args.kwargs["people_detected"])
            self.assertFalse(write_mock.call_args.kwargs["people_identified"])
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["processing"]["cast_store_signature"],
                "new-sig",
            )
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["caption"]["estimated_people_count"],
                9,
            )

    def test_run_chains_location_shown_repair_after_people_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            view = base / "Egypt_1975_B00_View"
            view.mkdir()
            image = view / "Egypt_1975_B00_P09_V.jpg"
            image.write_bytes(b"scan")
            stat = image.stat()
            detections = {
                "people": [],
                "objects": [],
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "lmstudio",
                    "fallback": False,
                    "error": "",
                    "model": "test-model",
                    "people_present": True,
                    "estimated_people_count": 9,
                },
                "processing": {
                    "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                    "settings_signature": "",
                    "cast_store_signature": "old-sig",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "date_estimate_input_hash": "",
                    "analysis_mode": "single_image",
                },
                "location": {
                    "query": "Paris, France",
                    "gps_latitude": 48.8566,
                    "gps_longitude": 2.3522,
                    "city": "Paris",
                    "country": "France",
                },
            }
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["egypt"],
                description="Travel photo",
                album_title="Egypt 1975",
                gps_latitude="48.8566",
                gps_longitude="2.3522",
                location_city="Paris",
                location_country="France",
                source_text=ai_index._build_dc_source("Egypt 1975", image, []),
                ocr_text="March 1975",
                dc_date="1975-03",
                detections_payload=detections,
                subphotos=[],
                people_detected=True,
                people_identified=False,
            )

            fake_matcher = mock.Mock()
            fake_matcher.store_signature.return_value = "new-sig"
            fake_matcher.match_image.return_value = []
            fake_matcher.last_faces_detected = 0

            with (
                mock.patch.object(ai_index_runner, "_init_people_matcher", return_value=fake_matcher),
                mock.patch.object(
                    ai_index_runner,
                    "_init_caption_engine",
                    return_value=SimpleNamespace(effective_model_name="caption-current"),
                ) as caption_engine_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_location_metadata",
                    return_value=("48.8566", "2.3522", "Paris, France"),
                ) as location_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_locations_shown",
                    return_value=([{"city": "Paris", "country_name": "France"}], True),
                ) as shown_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_location_payload",
                    return_value={
                        "query": "Paris, France",
                        "gps_latitude": "48.8566",
                        "gps_longitude": "2.3522",
                        "city": "Paris",
                        "country": "France",
                    },
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "_write_sidecar_and_record") as write_mock,
                mock.patch.object(ai_index_runner, "_emit_prompt_debug_artifact"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            fake_matcher.match_image.assert_called_once()
            caption_engine_mock.assert_called_once()
            location_mock.assert_called_once()
            shown_mock.assert_called_once()
            self.assertEqual(write_mock.call_count, 2)
            self.assertEqual(write_mock.call_args_list[1].kwargs["detections_payload"]["location_shown_ran"], True)
            self.assertEqual(
                write_mock.call_args_list[1].kwargs["detections_payload"]["locations_shown"],
                [{"city": "Paris", "country_name": "France"}],
            )

    def test_needs_processing_requires_current_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            stat = image.stat()
            row = {
                "image_path": str(image),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sidecar_path": str(image.with_suffix(".xmp")),
                "processor_signature": ai_index.PROCESSOR_SIGNATURE,
            }
            self.assertTrue(ai_index.needs_processing(image, row, force=False))

            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

    def test_needs_processing_skips_current_sidecar_even_when_sidecar_state_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")
            row = {
                "image_path": str(image),
                "size": 999,
                "mtime_ns": 999,
                "sidecar_path": str(sidecar),
                "processor_signature": "old-signature",
            }

            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(
                ai_index.needs_processing(
                    image,
                    row,
                    force=False,
                    reprocess_required=True,
                )
            )

    def test_needs_processing_skips_when_sidecar_state_missing_but_valid_sidecar_exists(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")

            self.assertFalse(ai_index.needs_processing(image, None, force=False))
            self.assertTrue(ai_index.needs_processing(image, None, force=True))

    def test_needs_processing_ignores_tiny_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text("tiny", encoding="utf-8")

            self.assertTrue(ai_index.needs_processing(image, None, force=False))

    def test_sidecar_has_lmstudio_caption_error(self):
        self.assertTrue(
            ai_index._sidecar_has_lmstudio_caption_error(
                {
                    "detections": {
                        "caption": {
                            "requested_engine": "lmstudio",
                            "effective_engine": "lmstudio",
                            "error": "model offline",
                        }
                    }
                }
            )
        )
        self.assertFalse(
            ai_index._sidecar_has_lmstudio_caption_error(
                {
                    "detections": {
                        "caption": {
                            "requested_engine": "lmstudio",
                            "effective_engine": "lmstudio",
                            "error": "",
                        }
                    }
                }
            )
        )
        self.assertFalse(
            ai_index._sidecar_has_lmstudio_caption_error(
                {
                    "detections": {
                        "caption": {
                            "requested_engine": "local",
                            "effective_engine": "local",
                            "error": "model offline",
                        }
                    }
                }
            )
        )

    def test_prepare_ai_model_image_scales_when_threshold_exceeded(self):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"pillow unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.png"
            Image.new("RGB", (240, 180), color="white").save(image)
            with mock.patch.object(ai_index_analysis, "AI_MODEL_MAX_SOURCE_BYTES", 1):
                with ai_index._prepare_ai_model_image(image) as prepared:
                    self.assertNotEqual(prepared, image)
                    self.assertTrue(prepared.exists())
                    self.assertEqual(prepared.suffix.lower(), ".jpg")

    def test_prepare_ai_model_image_disables_pillow_pixel_guard(self):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"pillow unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "guard.jpg"
            Image.new("RGB", (8, 8), color="white").save(image)
            original_limit = Image.MAX_IMAGE_PIXELS
            try:
                Image.MAX_IMAGE_PIXELS = 1
                with mock.patch.object(ai_index_analysis, "AI_MODEL_MAX_SOURCE_BYTES", 1):
                    with ai_index._prepare_ai_model_image(image) as prepared:
                        self.assertTrue(prepared.exists())
            finally:
                Image.MAX_IMAGE_PIXELS = original_limit

    def test_resolve_archive_scan_authoritative_ocr_stitches_once_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            group_paths = [scan1, scan2]
            signature = ai_index._scan_group_signature(group_paths)
            cache: dict[str, ai_index.ArchiveScanOCRAuthority] = {}
            stitched = np.full((12, 20, 3), 255, dtype=np.uint8)

            with mock.patch(
                "photoalbums.stitch_oversized_pages.build_stitched_image",
                return_value=stitched,
            ) as build_mock:
                first = ai_index._resolve_archive_scan_authoritative_ocr(
                    image_path=scan1,
                    group_paths=group_paths,
                    group_signature=signature,
                    cache=cache,
                )
                second = ai_index._resolve_archive_scan_authoritative_ocr(
                    image_path=scan1,
                    group_paths=group_paths,
                    group_signature=signature,
                    cache=cache,
                )

            self.assertEqual(first.ocr_text, "")
            self.assertEqual(first.ocr_hash, "")
            self.assertEqual(tuple(first.group_paths), (scan1, scan2))
            self.assertEqual(first, second)
            build_mock.assert_called_once_with([str(scan1), str(scan2)])

    def test_resolve_archive_scan_authoritative_ocr_prefers_existing_view_jpg(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            view = base / "China_1986_B02_View"
            archive.mkdir()
            view.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            view_jpg = view / "China_1986_B02_P02_V.jpg"
            cv2.imwrite(str(view_jpg), np.full((24, 32, 3), 255, dtype=np.uint8))
            ocr_engine = mock.Mock()
            ocr_engine.engine = "local"
            ocr_engine.read_text.return_value = "MAINLAND CHINA 1986 BOOK 11"

            with mock.patch(
                "photoalbums.stitch_oversized_pages.build_stitched_image",
            ) as build_mock:
                authority = ai_index._resolve_archive_scan_authoritative_ocr(
                    image_path=scan1,
                    group_paths=[scan1, scan2],
                    group_signature=ai_index._scan_group_signature([scan1, scan2]),
                    cache={},
                    ocr_engine=ocr_engine,
                )

            build_mock.assert_not_called()
            self.assertEqual(authority.stitched_image_path, view_jpg)
            self.assertEqual(authority.ocr_text, "MAINLAND CHINA 1986 BOOK 11")
            self.assertEqual(authority.ocr_hash, ai_index._hash_text("MAINLAND CHINA 1986 BOOK 11"))

    def test_run_image_analysis_passes_people_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = [
                SimpleNamespace(
                    name="Alice",
                    score=0.92,
                    certainty=0.92,
                    reviewed_by_human=False,
                    face_id="face-1",
                )
            ]
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.engine = "none"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                ocr_text="TEMPLE OF HEAVEN\nNO SMOKING",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_image_path=Path(tmp) / "people.jpg",
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="template",
                ocr_engine_name="none",
                ocr_language="eng",
                people_hint_text="Page caption",
                people_source_path=Path(tmp) / "original.jpg",
                people_bbox_offset=(12, 34),
            )

            people_matcher.match_image.assert_called_once_with(
                Path(tmp) / "people.jpg",
                source_path=Path(tmp) / "original.jpg",
                bbox_offset=(12, 34),
                hint_text="Page caption",
            )
            ocr_engine.read_text.assert_not_called()
            self.assertEqual(analysis.people_names, ["Alice"])
            self.assertEqual(analysis.payload["people"][0]["face_id"], "face-1")
            self.assertFalse(analysis.payload["people"][0]["reviewed_by_human"])

    def test_match_people_with_cast_store_retry_retries_transient_permission_error(self):
        attempts = {"count": 0}

        class _FakeMatcher:
            def match_image(self, image_path, *, source_path, bbox_offset, hint_text):
                del image_path, source_path, bbox_offset, hint_text
                if attempts["count"] < 2:
                    attempts["count"] += 1
                    raise PermissionError(
                        13,
                        "Access is denied",
                        r"C:\Users\covec\Videos\imago\cast\data\faces.jsonl",
                    )
                attempts["count"] += 1
                return ["ok"]

        with mock.patch.object(ai_index.time, "sleep") as sleep_mock:
            result = ai_index._match_people_with_cast_store_retry(
                people_matcher=_FakeMatcher(),
                image_path=Path("page.jpg"),
                source_path=Path("page.jpg"),
                bbox_offset=(0, 0),
                hint_text="",
            )

        self.assertEqual(result, ["ok"])
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_run_image_analysis_skips_ocr_when_engine_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            scaled = Path(tmp) / "scaled.jpg"
            scaled.write_bytes(b"scaled")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.engine = "none"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                ocr_text="TEMPLE OF HEAVEN\nNO SMOKING",
                fallback=False,
                error="",
            )

            @contextmanager
            def fake_prepare(_path):
                yield scaled

            with mock.patch.object(ai_index_analysis, "_prepare_ai_model_image", side_effect=fake_prepare):
                analysis = ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="none",
                    ocr_language="eng",
                )

            ocr_engine.read_text.assert_not_called()
            people_matcher.match_image.assert_called_once_with(
                image,
                source_path=image,
                bbox_offset=(0, 0),
                hint_text="",
            )
            object_detector.detect_image.assert_called_once_with(scaled)
            caption_engine.generate.assert_called_once_with(
                image_path=scaled,
                people=[],
                objects=[],
                ocr_text="",
                source_path=image,
                album_title="",
                printed_album_title="",
                photo_count=1,
                people_positions={},
                context_ocr_text="",
                debug_recorder=None,
                debug_step="caption",
            )
            self.assertEqual(analysis.ocr_text, "TEMPLE OF HEAVEN\nNO SMOKING")

    def test_run_image_analysis_uses_scaled_image_for_explicit_ocr_objects_and_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            scaled = Path(tmp) / "scaled.jpg"
            scaled.write_bytes(b"scaled")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.engine = "lmstudio"
            ocr_engine.read_text.return_value = "hello"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                ocr_text="TEMPLE OF HEAVEN\nNO SMOKING",
                fallback=False,
                error="",
            )

            @contextmanager
            def fake_prepare(_path):
                yield scaled

            with mock.patch.object(ai_index_analysis, "_prepare_ai_model_image", side_effect=fake_prepare):
                analysis = ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="lmstudio",
                    ocr_language="eng",
                )

            ocr_engine.read_text.assert_called_once_with(scaled, debug_recorder=None, debug_step="ocr")
            people_matcher.match_image.assert_called_once_with(
                image,
                source_path=image,
                bbox_offset=(0, 0),
                hint_text="hello",
            )
            object_detector.detect_image.assert_called_once_with(scaled)
            caption_engine.generate.assert_called_once_with(
                image_path=scaled,
                people=[],
                objects=[],
                ocr_text="hello",
                source_path=image,
                album_title="",
                printed_album_title="",
                photo_count=1,
                people_positions={},
                context_ocr_text="",
                debug_recorder=None,
                debug_step="caption",
            )
            self.assertEqual(analysis.ocr_text, "hello")

    def test_run_image_analysis_reports_locations_shown_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            steps: list[str] = []
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.engine = "none"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                ocr_text="",
                fallback=False,
                error="",
            )

            with (
                mock.patch.object(ai_index_analysis, "_resolve_location_metadata", return_value=("", "", "")),
                mock.patch.object(ai_index_analysis, "_resolve_locations_shown", return_value=([], True)),
            ):
                ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="none",
                    ocr_language="eng",
                    step_fn=steps.append,
                )

            self.assertIn("locations_shown", steps)

    def test_run_image_analysis_preserves_empty_ocr_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            scaled = Path(tmp) / "scaled.jpg"
            scaled.write_bytes(b"scaled")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.engine = "lmstudio"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                ocr_text="SHOULD NOT BE USED",
                fallback=False,
                error="",
            )

            @contextmanager
            def fake_prepare(_path):
                yield scaled

            with mock.patch.object(ai_index_analysis, "_prepare_ai_model_image", side_effect=fake_prepare):
                analysis = ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="lmstudio",
                    ocr_language="eng",
                    ocr_text_override="",
                )

            ocr_engine.read_text.assert_not_called()
            self.assertEqual(analysis.ocr_text, "")

    def test_run_image_analysis_records_gps_location_from_ocr_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "Latitude: 39.7875\nLongitude: 100.307222"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Mogao Caves in Dunhuang, China (39°47′15″N 100°18′26″E).",
                engine="lmstudio",
                ocr_text="Latitude: 39.7875\nLongitude: 100.307222",
                gps_latitude="",
                gps_longitude="",
                location_name="",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="lmstudio",
                ocr_language="eng",
            )

            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.7875)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 100.307222)

    def test_resolve_location_metadata_merges_model_location_when_available(self):
        image = Path("cover.jpg")
        caption_engine = mock.Mock()
        caption_engine.estimate_location.return_value = SimpleNamespace(
            gps_latitude="19.1414769",
            gps_longitude="72.8323049",
            location_name="Mainland China",
            fallback=False,
        )

        gps_latitude, gps_longitude, location_name = ai_index._resolve_location_metadata(
            requested_caption_engine="lmstudio",
            caption_engine=caption_engine,
            model_image_path=image,
            people=[],
            objects=[],
            ocr_text="MAINLAND CHINA\n1986\nBOOK 11",
            source_path=image,
            album_title="",
            printed_album_title="",
            people_positions={},
            fallback_location_name="",
        )

        self.assertEqual((gps_latitude, gps_longitude, location_name), ("19.1414769", "72.8323049", "Mainland China"))
        caption_engine.estimate_location.assert_called_once()

    def test_resolve_locations_shown_prefers_geocoded_gps_over_ai_gps(self):
        caption_engine = mock.Mock()
        caption_engine.estimate_locations_shown.return_value = SimpleNamespace(
            fallback=False,
            locations_shown=[
                {
                    "name": "Hassan II Mosque",
                    "city": "Casablanca",
                    "country_name": "Morocco",
                }
            ],
        )
        geocoder = mock.Mock()
        geocoder.geocode.return_value = SimpleNamespace(
            latitude="33.6084",
            longitude="-7.6326",
        )

        locations, ran = ai_index._resolve_locations_shown(
            requested_caption_engine="lmstudio",
            caption_engine=caption_engine,
            model_image_path=Path("sample.jpg"),
            ocr_text="",
            source_path=Path("sample.jpg"),
            geocoder=geocoder,
        )

        self.assertEqual(ran, True)
        self.assertEqual(
            locations,
            [
                {
                    "name": "Hassan II Mosque",
                    "world_region": "",
                    "country_name": "Morocco",
                    "country_code": "",
                    "province_or_state": "",
                    "city": "Casablanca",
                    "sublocation": "",
                    "gps_latitude": "33.6084",
                    "gps_longitude": "-7.6326",
                    "gps_source": "nominatim",
                }
            ],
        )
        geocoder.geocode.assert_called_once_with("Hassan II Mosque, Casablanca, Morocco")

    def test_resolve_location_payload_records_geocode_artifact(self):
        geocoder = mock.Mock()
        geocoder.geocode.return_value = SimpleNamespace(
            query="Mogao Caves, Dunhuang, Gansu, China",
            latitude="39.9361",
            longitude="94.8076",
            display_name="Mogao Caves, Dunhuang, Jiuquan, Gansu, China",
            source="nominatim",
            city="Dunhuang",
            state="Gansu",
            country="China",
            sublocation="273 Mogao Road",
        )
        records: list[dict[str, object]] = []

        payload = ai_index._resolve_location_payload(
            geocoder=geocoder,
            gps_latitude="",
            gps_longitude="",
            location_name="Mogao Caves, Dunhuang, Gansu, China",
            artifact_recorder=records.append,
            artifact_step="location",
        )

        self.assertEqual(payload["query"], "Mogao Caves, Dunhuang, Gansu, China")
        self.assertEqual(payload["display_name"], "Mogao Caves, Dunhuang, Jiuquan, Gansu, China")
        self.assertEqual(payload["sublocation"], "273 Mogao Road")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["step"], "location")
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["query"], "Mogao Caves, Dunhuang, Gansu, China")
        self.assertEqual(records[0]["result"]["latitude"], "39.9361")
        self.assertEqual(records[0]["result"]["sublocation"], "273 Mogao Road")

    def test_resolve_location_payload_records_geocode_miss(self):
        geocoder = mock.Mock()
        geocoder.geocode.return_value = None
        records: list[dict[str, object]] = []

        payload = ai_index._resolve_location_payload(
            geocoder=geocoder,
            gps_latitude="",
            gps_longitude="",
            location_name="Unknown Place",
            artifact_recorder=records.append,
            artifact_step="location",
        )

        self.assertEqual(payload, {})
        self.assertEqual(
            records, [{"step": "location", "service": "nominatim", "query": "Unknown Place", "status": "miss"}]
        )

    def test_resolve_locations_shown_leaves_gps_empty_when_geocode_misses(self):
        caption_engine = mock.Mock()
        caption_engine.estimate_locations_shown.return_value = SimpleNamespace(
            fallback=False,
            locations_shown=[
                {
                    "name": "Hassan II Mosque",
                    "city": "Casablanca",
                    "country_name": "Morocco",
                }
            ],
        )
        geocoder = mock.Mock()
        geocoder.geocode.return_value = None

        locations, ran = ai_index._resolve_locations_shown(
            requested_caption_engine="lmstudio",
            caption_engine=caption_engine,
            model_image_path=Path("sample.jpg"),
            ocr_text="",
            source_path=Path("sample.jpg"),
            geocoder=geocoder,
        )

        self.assertEqual(ran, True)
        self.assertNotIn("gps_latitude", locations[0])
        self.assertNotIn("gps_longitude", locations[0])

    def test_resolve_locations_shown_records_geocode_queries(self):
        caption_engine = mock.Mock()
        caption_engine.estimate_locations_shown.return_value = SimpleNamespace(
            fallback=False,
            locations_shown=[
                {
                    "name": "Hassan II Mosque",
                    "city": "Casablanca",
                    "country_name": "Morocco",
                }
            ],
        )
        geocoder = mock.Mock()
        geocoder.geocode.return_value = SimpleNamespace(
            query="Hassan II Mosque, Casablanca, Morocco",
            latitude="33.6084",
            longitude="-7.6326",
            display_name="Hassan II Mosque, Casablanca, Morocco",
            source="nominatim",
            city="Casablanca",
            state="Casablanca-Settat",
            country="Morocco",
            sublocation="Boulevard de la Corniche",
        )
        records: list[dict[str, object]] = []

        locations, ran = ai_index._resolve_locations_shown(
            requested_caption_engine="lmstudio",
            caption_engine=caption_engine,
            model_image_path=Path("sample.jpg"),
            ocr_text="",
            source_path=Path("sample.jpg"),
            geocoder=geocoder,
            artifact_recorder=records.append,
            debug_step="locations_shown",
        )

        self.assertEqual(ran, True)
        self.assertEqual(locations[0]["gps_source"], "nominatim")
        self.assertEqual(locations[0]["sublocation"], "Boulevard de la Corniche")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["step"], "locations_shown")
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["result"]["display_name"], "Hassan II Mosque, Casablanca, Morocco")
        self.assertEqual(records[0]["result"]["sublocation"], "Boulevard de la Corniche")

    def test_resolve_locations_shown_dedupes_name_embedded_city_and_country(self):
        caption_engine = mock.Mock()
        caption_engine.estimate_locations_shown.return_value = SimpleNamespace(
            fallback=False,
            locations_shown=[
                {
                    "name": "Fes, Morocco",
                    "city": "Fes",
                    "country_name": "Morocco",
                }
            ],
        )
        geocoder = mock.Mock()
        geocoder.geocode.return_value = None

        locations, ran = ai_index._resolve_locations_shown(
            requested_caption_engine="lmstudio",
            caption_engine=caption_engine,
            model_image_path=Path("sample.jpg"),
            ocr_text="",
            source_path=Path("sample.jpg"),
            geocoder=geocoder,
        )

        self.assertEqual(ran, True)
        self.assertEqual(locations[0]["name"], "Fes, Morocco")
        geocoder.geocode.assert_called_once_with("Fes, Morocco")

    def test_has_legacy_ai_locations_shown_gps_detects_missing_source(self):
        self.assertEqual(
            ai_index._has_legacy_ai_locations_shown_gps(
                {
                    "detections": {
                        "locations_shown": [
                            {
                                "name": "Rock of Gibraltar",
                                "gps_latitude": "51.95101767",
                                "gps_longitude": "-2.72020567",
                            }
                        ]
                    }
                }
            ),
            True,
        )
        self.assertEqual(
            ai_index._has_legacy_ai_locations_shown_gps(
                {
                    "detections": {
                        "locations_shown": [
                            {
                                "name": "Rock of Gibraltar",
                                "gps_latitude": "36.1408",
                                "gps_longitude": "-5.3536",
                                "gps_source": "nominatim",
                            }
                        ]
                    }
                }
            ),
            False,
        )

    def test_run_image_analysis_geocodes_structured_location_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="The entrance to the Mogao Caves in Dunhuang.",
                engine="lmstudio",
                gps_latitude="",
                gps_longitude="",
                location_name="Mogao Caves, Dunhuang, Gansu, China",
                fallback=False,
                error="",
            )
            caption_engine.estimate_location.return_value = SimpleNamespace(
                fallback=False,
                gps_latitude="",
                gps_longitude="",
                location_name="Mogao Caves, Dunhuang, Gansu, China",
            )
            geocoder = mock.Mock()
            geocoder.geocode.return_value = SimpleNamespace(
                query="Mogao Caves, Dunhuang, Gansu, China",
                latitude="39.9361",
                longitude="94.8076",
                display_name="Mogao Caves, Dunhuang, Jiuquan, Gansu, China",
                source="nominatim",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="none",
                ocr_language="eng",
                geocoder=geocoder,
            )

            geocoder.geocode.assert_called_once_with("Mogao Caves, Dunhuang, Gansu, China")
            self.assertEqual(
                analysis.payload["location"]["query"],
                "Mogao Caves, Dunhuang, Gansu, China",
            )
            self.assertEqual(
                analysis.payload["location"]["display_name"],
                "Mogao Caves, Dunhuang, Jiuquan, Gansu, China",
            )
            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.9361)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 94.8076)
            self.assertEqual(analysis.payload["location"]["source"], "nominatim")

    def test_run_image_analysis_merges_lmstudio_location_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="A rocky desert landscape.",
                engine="lmstudio",
                ocr_text="",
                location_name="",
                fallback=False,
                error="",
            )
            caption_engine.estimate_location.return_value = SimpleNamespace(
                gps_latitude="39.9361",
                gps_longitude="94.8076",
                location_name="Mogao Caves, Dunhuang, Gansu, China",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="lmstudio",
                ocr_language="eng",
            )

            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.9361)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 94.8076)
            self.assertEqual(
                analysis.payload["location"]["query"],
                "Mogao Caves, Dunhuang, Gansu, China",
            )
            caption_engine.estimate_location.assert_called_once()

    def test_run_image_analysis_prefers_explicit_ocr_gps_over_model_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "Latitude: 39.7875\nLongitude: 100.307222"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="A scenic location.",
                engine="lmstudio",
                ocr_text="Latitude: 39.7875\nLongitude: 100.307222",
                location_name="",
                fallback=False,
                error="",
            )
            caption_engine.estimate_location.return_value = SimpleNamespace(
                gps_latitude="39.9361",
                gps_longitude="94.8076",
                location_name="Mogao Caves, Dunhuang, Gansu, China",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="lmstudio",
                ocr_language="eng",
            )

            self.assertEqual(analysis.payload["location"]["gps_latitude"], 39.7875)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], 100.307222)
            self.assertEqual(
                analysis.payload["location"]["query"],
                "Mogao Caves, Dunhuang, Gansu, China",
            )

    def test_run_image_analysis_uses_configured_title_page_location_for_p01(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "China_1986_B02_P01.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Album title page.",
                engine="lmstudio",
                ocr_text="",
                location_name="Los Angeles, California",
                album_title="Mainland China Book II",
                fallback=False,
                error="",
            )
            caption_engine.estimate_location.return_value = SimpleNamespace(
                gps_latitude="34.0522",
                gps_longitude="-118.2437",
                location_name="Los Angeles, California",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="lmstudio",
                ocr_language="eng",
                title_page_location={
                    "address": "2240 Lorain Rd, San Marino, CA 91108",
                    "gps_latitude": "34.11512",
                    "gps_longitude": "-118.10492",
                    "city": "San Marino",
                    "state": "CA",
                    "country": "United States",
                    "sublocation": "2240 Lorain Rd",
                },
            )

            self.assertEqual(analysis.payload["location"]["query"], "2240 Lorain Rd, San Marino, CA 91108")
            self.assertEqual(analysis.payload["location"]["gps_latitude"], 34.11512)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], -118.10492)
            self.assertEqual(analysis.payload["location"]["city"], "San Marino")
            self.assertEqual(analysis.payload["location"]["source"], ai_index.TITLE_PAGE_LOCATION_SOURCE)

    def test_run_image_analysis_does_not_use_configured_title_page_location_for_non_p01(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "China_1986_B02_P02.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Page description.",
                engine="lmstudio",
                ocr_text="",
                location_name="Los Angeles, California",
                album_title="Mainland China Book II",
                fallback=False,
                error="",
            )
            caption_engine.estimate_location.return_value = SimpleNamespace(
                gps_latitude="34.0522",
                gps_longitude="-118.2437",
                location_name="Los Angeles, California",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="lmstudio",
                ocr_language="eng",
                title_page_location={
                    "address": "2240 Lorain Rd, San Marino, CA 91108",
                    "gps_latitude": "34.11512",
                    "gps_longitude": "-118.10492",
                    "city": "San Marino",
                    "state": "CA",
                    "country": "United States",
                    "sublocation": "2240 Lorain Rd",
                },
            )

            self.assertEqual(analysis.payload["location"]["query"], "Los Angeles, California")
            self.assertEqual(analysis.payload["location"]["gps_latitude"], 34.0522)
            self.assertEqual(analysis.payload["location"]["gps_longitude"], -118.2437)
            self.assertEqual(analysis.payload["location"]["source"], "caption")

    def test_run_image_analysis_merges_lmstudio_people_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Four people stand together outdoors.",
                engine="lmstudio",
                location_name="",
                fallback=False,
                error="",
            )
            caption_engine.estimate_people.return_value = SimpleNamespace(
                people_present=True,
                estimated_people_count=4,
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="lmstudio",
                ocr_engine_name="none",
                ocr_language="eng",
            )

            self.assertTrue(analysis.payload["caption"]["people_present"])
            self.assertEqual(analysis.payload["caption"]["estimated_people_count"], 4)
            caption_engine.estimate_people.assert_called_once()

    def test_build_flat_page_description_uses_cover_caption_with_book_note_on_fallback(
        self,
    ):
        layout = SimpleNamespace(
            original_path=Path("China_1986_B02_P01.jpg"),
            content_path=Path("page.jpg"),
        )
        analysis = ai_index.ImageAnalysis(
            image_path=Path("page.jpg"),
            people_names=[],
            object_labels=[],
            ocr_text="MAINLAND CHINA 1986 BOOK 11",
            ocr_keywords=["mainland", "china", "1986", "book"],
            subjects=["mainland", "china", "1986", "book"],
            description='Visible text reads: "MAINLAND CHINA 1986 BOOK 11".',
            payload={
                "people": [],
                "objects": [],
                "ocr": {
                    "engine": "lmstudio",
                    "language": "eng",
                    "keywords": ["mainland"],
                    "chars": 27,
                },
                "caption": {
                    "requested_engine": "lmstudio",
                    "effective_engine": "template",
                    "fallback": True,
                    "error": "LM Studio returned invalid structured caption JSON: Expecting value",
                    "model": "",
                },
            },
        )

        description = ai_index._build_flat_page_description(analysis=analysis)

        self.assertIn("MAINLAND CHINA 1986 BOOK 11", description)

    def test_run_force_rewrites_existing_sidecar_and_merges_embedded_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            original = self._valid_sidecar_text()
            sidecar.write_text(original, encoding="utf-8")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(
                    ai_index_runner,
                    "_page_scan_filenames",
                    return_value=["Family_2020_B01_P01_S01.tif", "Family_2020_B01_P01_S02.tif"],
                ),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--force",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            self.assertEqual(
                write_mock.call_args.kwargs["source_text"],
                "Scan(s) S01 S02; Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif",
            )
            self.assertEqual(sidecar.read_text(encoding="utf-8"), original)

    def test_run_skips_existing_current_sidecar_without_ai_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool="https://github.com/cove/imago",
                person_names=[],
                subjects=[],
                description="",
                source_text="",
                ocr_text="",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": [],
                        "chars": 0,
                    },
                    "caption": {
                        "requested_engine": "template",
                        "effective_engine": "template",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
                subphotos=[],
            )

            with (
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_refreshes_current_view_sidecar_when_dc_source_text_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "England_1983_B02_Archive"
            view = base / "England_1983_B02_View"
            archive.mkdir()
            view.mkdir()
            image = view / "England_1983_B02_P47_V.jpg"
            image.write_bytes(b"abc")
            (archive / "England_1983_B02_P47_S01.tif").write_bytes(b"scan")
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool="https://github.com/cove/imago",
                person_names=[],
                subjects=["ENGLAND BOOK 11 1983"],
                description="",
                album_title="ENGLAND BOOK 11 1983",
                source_text="ENGLAND BOOK 11 1983 Page 47 Scans S01; England_1983_B02_P47_S01.tif",
                ocr_text="",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": [],
                        "chars": 0,
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
                subphotos=[],
            )

            with (
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "_get_image_dimensions", return_value=(1, 1)),
                mock.patch.object(ai_index_runner, "read_embedded_create_date", return_value=""),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--album",
                        "England_1983_B02_View",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            write_mock.assert_called_once()
            self.assertEqual(
                write_mock.call_args.kwargs["source_text"],
                "ENGLAND BOOK 11 1983 Page 47 Scan(s) S01; England_1983_B02_P47_S01.tif",
            )

    def test_run_skips_current_sidecar_when_manifest_settings_are_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="https://github.com/cove/imago",
                person_names=[],
                subjects=["hello"],
                description="Old description",
                source_text="",
                ocr_text="hello",
                dc_date="1975",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "lmstudio",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                        "model": "current-ocr-model",
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "current-caption-model",
                    },
                },
                subphotos=[],
                ocr_ran=True,
                people_detected=False,
                people_identified=False,
            )

            with (
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_reprocesses_current_sidecar_when_ai_fields_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <xmp:CreatorTool>https://github.com/cove/imago</xmp:CreatorTool>
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Old description</rdf:li>
        </rdf:Alt>
      </dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
                encoding="utf-8",
            )

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["hello"],
                description="Updated description",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                    },
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()
            write_mock.assert_called_once()

    def test_run_batch_repairs_missing_location_shown_without_full_reprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["hello"],
                description="Travel photo",
                album_title="Egypt 1975",
                gps_latitude="39.7875",
                gps_longitude="100.307222",
                location_city="Zhangye",
                location_country="China",
                source_text=ai_index._build_dc_source("Egypt 1975", image, ai_index._page_scan_filenames(image)),
                ocr_text="hello",
                dc_date="1975",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                        "model": "qwen-current-ocr",
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "caption-current",
                    },
                    "location": {
                        "query": "Zhangye, China",
                        "gps_latitude": 39.7875,
                        "gps_longitude": 100.307222,
                        "city": "Zhangye",
                        "country": "China",
                    },
                    "location_shown_ran": True,
                    "locations_shown": [
                        {
                            "city": "Zhangye",
                            "country_code": "CN",
                            "country_name": "China",
                            "province_or_state": "Gansu",
                            "sublocation": "Mati Temple",
                            "world_region": "Asia",
                        }
                    ],
                },
                subphotos=[],
                ocr_ran=True,
                people_detected=False,
                people_identified=False,
            )

            @contextmanager
            def fake_prepare(_path: Path):
                yield image

            with (
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_init_caption_engine",
                    return_value=SimpleNamespace(effective_model_name="caption-current"),
                ) as caption_engine_mock,
                mock.patch.object(ai_index_analysis, "_prepare_ai_model_image", side_effect=fake_prepare),
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_location_metadata",
                    return_value=("48.8566", "2.3522", "Paris, France"),
                ) as location_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_locations_shown",
                    return_value=([{"city": "Paris", "country_name": "France"}], True),
                ) as shown_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_location_payload",
                    return_value={
                        "query": "Paris, France",
                        "gps_latitude": "48.8566",
                        "gps_longitude": "2.3522",
                        "city": "Paris",
                        "country": "France",
                    },
                ),
                mock.patch.object(ai_index_runner, "_write_sidecar_and_record") as write_mock,
                mock.patch.object(ai_index_runner, "_emit_prompt_debug_artifact"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            caption_engine_mock.assert_called_once()
            location_mock.assert_called_once()
            shown_mock.assert_called_once()
            write_mock.assert_called_once()
            detections = write_mock.call_args.kwargs["detections_payload"]
            self.assertEqual(detections["caption"]["effective_engine"], "lmstudio")
            self.assertEqual(detections["location"]["query"], "Paris, France")
            self.assertEqual(detections["locations_shown"], [{"city": "Paris", "country_name": "France"}])
            self.assertEqual(detections["location_shown_ran"], True)

    def test_run_batch_repairs_missing_location_shown_when_flag_true_but_tag_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["hello"],
                description="Travel photo",
                album_title="Egypt 1975",
                source_text=ai_index._build_dc_source("Egypt 1975", image, ai_index._page_scan_filenames(image)),
                ocr_text="hello",
                dc_date="1975",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                        "model": "qwen-current-ocr",
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "caption-current",
                    },
                    "location_shown_ran": True,
                },
                subphotos=[],
                ocr_ran=True,
                people_detected=False,
                people_identified=False,
            )

            @contextmanager
            def fake_prepare(_path: Path):
                yield image

            with (
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_init_caption_engine",
                    return_value=SimpleNamespace(effective_model_name="caption-current"),
                ) as caption_engine_mock,
                mock.patch.object(ai_index_analysis, "_prepare_ai_model_image", side_effect=fake_prepare),
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_location_metadata",
                    return_value=("48.8566", "2.3522", "Paris, France"),
                ) as location_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_locations_shown",
                    return_value=([{"city": "Paris", "country_name": "France"}], True),
                ) as shown_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_location_payload",
                    return_value={
                        "query": "Paris, France",
                        "gps_latitude": "48.8566",
                        "gps_longitude": "2.3522",
                        "city": "Paris",
                        "country": "France",
                    },
                ),
                mock.patch.object(ai_index_runner, "_write_sidecar_and_record") as write_mock,
                mock.patch.object(ai_index_runner, "_emit_prompt_debug_artifact"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            caption_engine_mock.assert_called_once()
            location_mock.assert_called_once()
            shown_mock.assert_called_once()
            write_mock.assert_called_once()
            detections = write_mock.call_args.kwargs["detections_payload"]
            self.assertEqual(detections["locations_shown"], [{"city": "Paris", "country_name": "France"}])
            self.assertEqual(detections["location_shown_ran"], True)

    def test_run_batch_repairs_legacy_ai_location_shown_gps_without_full_reprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                person_names=[],
                subjects=["hello"],
                description="Travel photo",
                album_title="Spain 1988",
                source_text=ai_index._build_dc_source("Spain 1988", image, ai_index._page_scan_filenames(image)),
                ocr_text="ROCK OF GIBRALTAR",
                dc_date="1988",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["gibraltar"],
                        "chars": 18,
                        "model": "qwen-current-ocr",
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "caption-current",
                    },
                    "location_shown_ran": True,
                    "locations_shown": [
                        {
                            "name": "Rock of Gibraltar",
                            "world_region": "Europe",
                            "country_name": "United Kingdom",
                            "country_code": "GB",
                            "province_or_state": "Gibraltar",
                            "gps_latitude": "51.95101767",
                            "gps_longitude": "-2.72020567",
                        }
                    ],
                },
                locations_shown=[
                    {
                        "name": "Rock of Gibraltar",
                        "world_region": "Europe",
                        "country_name": "United Kingdom",
                        "country_code": "GB",
                        "province_or_state": "Gibraltar",
                        "gps_latitude": "51.95101767",
                        "gps_longitude": "-2.72020567",
                    }
                ],
                subphotos=[],
                ocr_ran=True,
                people_detected=False,
                people_identified=False,
            )

            @contextmanager
            def fake_prepare(_path: Path):
                yield image

            with (
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_init_caption_engine",
                    return_value=SimpleNamespace(effective_model_name="caption-current"),
                ) as caption_engine_mock,
                mock.patch.object(ai_index_analysis, "_prepare_ai_model_image", side_effect=fake_prepare),
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_location_metadata",
                    return_value=("", "", ""),
                ) as location_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_locations_shown",
                    return_value=(
                        [
                            {
                                "name": "Rock of Gibraltar",
                                "world_region": "Europe",
                                "country_name": "United Kingdom",
                                "country_code": "GB",
                                "province_or_state": "Gibraltar",
                                "gps_latitude": "36.1408",
                                "gps_longitude": "-5.3536",
                                "gps_source": "nominatim",
                            }
                        ],
                        True,
                    ),
                ) as shown_mock,
                mock.patch.object(ai_index_runner, "_resolve_location_payload", return_value={}),
                mock.patch.object(ai_index_runner, "_write_sidecar_and_record") as write_mock,
                mock.patch.object(ai_index_runner, "_emit_prompt_debug_artifact"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            caption_engine_mock.assert_called_once()
            location_mock.assert_called_once()
            shown_mock.assert_called_once()
            write_mock.assert_called_once()
            detections = write_mock.call_args.kwargs["detections_payload"]
            self.assertEqual(
                detections["locations_shown"],
                [
                    {
                        "name": "Rock of Gibraltar",
                        "world_region": "Europe",
                        "country_name": "United Kingdom",
                        "country_code": "GB",
                        "province_or_state": "Gibraltar",
                        "gps_latitude": "36.1408",
                        "gps_longitude": "-5.3536",
                        "gps_source": "nominatim",
                    }
                ],
            )
            self.assertEqual(detections["location_shown_ran"], True)

    def test_run_reprocesses_current_sidecar_when_lmstudio_caption_error_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                image.with_suffix(".xmp"),
                creator_tool="https://github.com/cove/imago",
                person_names=[],
                subjects=[],
                description="Old description",
                source_text="",
                ocr_text="",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": [],
                        "chars": 0,
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": True,
                        "error": "model offline",
                        "model": "qwen2.5-vl",
                    },
                },
                subphotos=[],
            )

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Updated description",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "none",
                        "language": "eng",
                        "keywords": [],
                        "chars": 0,
                    },
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "lmstudio",
                        "fallback": False,
                        "error": "",
                        "model": "qwen2.5-vl",
                    },
                },
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with (
                    mock.patch.object(
                        ai_index_runner,
                        "prepare_image_layout",
                        side_effect=lambda *args, **kwargs: self._mock_layout(image),
                    ),
                    mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis) as analysis_mock,
                    mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                    mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
                ):
                    result = ai_index.run(
                        [
                            "--photos-root",
                            str(base),
                            "--include-view",
                            "--disable-people",
                            "--disable-objects",
                            "--ocr-engine",
                            "none",
                            "--caption-engine",
                            "lmstudio",
                        ]
                    )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()
            write_mock.assert_called_once()
            self.assertIn("reprocess: lmstudio_caption_error", stdout.getvalue())

    def test_run_rewrites_sidecar_when_image_is_newer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")

            next_ns = max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            os.utime(image, ns=(next_ns, next_ns))

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()

    def test_run_records_final_cast_store_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=[],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["hello"],
                description="Alice",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )
            fake_matcher = mock.Mock()
            fake_matcher.store_signature.side_effect = [
                "sig-before",
                "sig-after",
            ]

            with (
                mock.patch.object(ai_index_runner, "_init_people_matcher", return_value=fake_matcher),
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            det = write_mock.call_args.kwargs["detections_payload"]
            self.assertEqual(det["processing"]["cast_store_signature"], "sig-after")

    def test_run_people_update_only_refreshes_detection_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            xmp_sidecar.write_xmp_sidecar(
                sidecar,
                creator_tool="https://github.com/cove/imago",
                person_names=["Alice"],
                subjects=["hello"],
                description="Old description",
                source_text="",
                ocr_text="hello",
                dc_date="1975",
                ocr_authority_source="archive_stitched",
                detections_payload={
                    "people": [
                        {
                            "name": "Alice",
                            "score": 0.95,
                            "bbox": [10, 10, 20, 20],
                        }
                    ],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["hello"],
                        "chars": 5,
                        "model": "qwen-old-ocr",
                    },
                    "caption": {
                        "requested_engine": "local",
                        "effective_engine": "local",
                        "fallback": False,
                        "error": "",
                        "model": "caption-old",
                    },
                },
                subphotos=[],
                ocr_ran=True,
                people_detected=True,
                people_identified=True,
            )

            fake_matcher = mock.Mock()
            fake_matcher.store_signature.return_value = "new-sig"
            fake_matcher.match_image.return_value = []
            fake_caption_engine = mock.Mock()
            fake_caption_engine.effective_model_name = "caption-new"
            fake_caption_engine.generate.return_value = SimpleNamespace(
                text="Updated description",
                engine="lmstudio",
                fallback=False,
                error="",
            )

            with (
                mock.patch.object(ai_index_runner, "_settings_signature", return_value="sig"),
                mock.patch.object(
                    ai_index_runner,
                    "_init_people_matcher",
                    return_value=fake_matcher,
                ) as people_matcher_mock,
                mock.patch.object(
                    ai_index_runner,
                    "_init_caption_engine",
                    return_value=fake_caption_engine,
                ) as caption_engine_mock,
                mock.patch.object(ai_index_runner, "_page_scan_filenames", return_value=[]),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--ocr-model",
                        "qwen-current-ocr",
                        "--caption-engine",
                        "lmstudio",
                        "--caption-model",
                        "caption-new",
                    ]
                )

            self.assertEqual(result, 0)
            people_matcher_mock.assert_not_called()
            caption_engine_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_enqueues_unmatched_face_in_cast_queue(self):
        if not hasattr(cv2, "rectangle") or not hasattr(cv2, "imwrite"):
            self.skipTest("opencv drawing helpers unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            frame = np.zeros((120, 120, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10, 10), (70, 70), (220, 220, 220), -1)
            cv2.imwrite(str(image), frame)
            cast_store = base / "cast_data"
            store = TextFaceStore(cast_store)
            store.ensure_files()

            class _FakeMatcher:
                def __init__(self, face_store: TextFaceStore, image_path: Path):
                    self._store = face_store
                    self._image_path = image_path
                    self.last_faces_detected = 0

                def store_signature(self) -> str:
                    return self._store.store_signature()

                def match_image(self, *_args, **_kwargs):
                    self.last_faces_detected = 1
                    face = self._store.add_face(
                        embedding=[1.0, 0.0, 0.0],
                        source_type="photo",
                        source_path=str(self._image_path),
                        bbox=[10, 10, 40, 40],
                        quality=0.93,
                        crop_path="crops/fake-face.jpg",
                        metadata={
                            "ingest": "photoalbums_ai",
                            "embedding_model": "test.embedding",
                            "detector_model": "test.detector",
                            "analysis_variant": "original",
                        },
                    )
                    self._store.add_review_item(
                        face_id=str(face["face_id"]),
                        candidates=[],
                        suggested_person_id=None,
                        suggested_score=None,
                        status="pending",
                    )
                    return []

                def match_image_recovery(self, *_args, **_kwargs):
                    self.last_faces_detected = 1
                    return []

            matcher = _FakeMatcher(store, image)

            with (
                mock.patch.object(ai_index_runner, "_init_people_matcher", return_value=matcher),
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_page_scan_filenames", return_value=[]),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--cast-store",
                        str(cast_store),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()

            faces = store.list_faces()
            reviews = store.list_review_items()

            self.assertEqual(len(faces), 1)
            self.assertEqual(str(faces[0]["source_path"]), str(image))
            self.assertEqual(len(reviews), 1)
            self.assertEqual(str(reviews[0]["face_id"]), str(faces[0]["face_id"]))
            self.assertEqual(str(reviews[0]["status"]), "pending")

    def test_run_stdout_prints_caption_only_and_skips_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "local"},
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_not_called()
            print_mock.assert_called_once_with("a.jpg: Alice with a dog")

    def test_run_stdout_emits_caption_fallback_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Fallback caption text",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {
                        "requested_engine": "local",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "model offline",
                        "model": "qwen/qwen3.5-9b",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            print_mock.assert_has_calls(
                [
                    mock.call(
                        "[1/1] warn  a.jpg: caption fallback: model offline",
                        file=sys.stderr,
                        flush=True,
                    ),
                    mock.call("a.jpg: Fallback caption text"),
                ]
            )
            self.assertEqual(print_mock.call_count, 2)

    def test_run_stdout_prints_filename_only_for_empty_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "model offline",
                        "model": "qwen2.5-vl-instruct",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            print_mock.assert_has_calls(
                [
                    mock.call(
                        "[1/1] warn  a.jpg: caption fallback: model offline",
                        file=sys.stderr,
                        flush=True,
                    ),
                    mock.call("a.jpg"),
                ]
            )
            self.assertEqual(print_mock.call_count, 2)

    def test_resolve_album_title_hint_prefers_existing_p01_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P01.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="This is the cover or title page of Mainland China Book II, a Photo Essay.",
                album_title="Mainland China Book II",
                source_text="",
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                subphotos=[],
            )

            title = ai_index._resolve_album_title_hint(image)
            self.assertEqual(title, "Mainland China Book II")

    def test_iter_album_cover_sidecars_excludes_detail_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            for name in (
                "China_1986_B02_P01.xmp",
                "China_1986_B02_P01_S01.xmp",
                "China_1986_B02_P01_D01-01.xmp",
            ):
                (album_dir / name).write_text("x", encoding="utf-8")

            sidecars = [path.name for path in ai_index._iter_album_cover_sidecars(image)]
            self.assertEqual(
                sidecars,
                [
                    "China_1986_B02_P01.xmp",
                    "China_1986_B02_P01_S01.xmp",
                ],
            )

    def test_resolve_album_title_hint_ignores_p00_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P00.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="This is the cover or title page of Mainland China Book II, a Photo Essay.",
                album_title="Mainland China Book II",
                source_text="",
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                subphotos=[],
            )

            title = ai_index._resolve_album_title_hint(image)
            self.assertEqual(title, "")

    def test_resolve_album_title_hint_ignores_p01_detail_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P01_D01-01.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=["derived", "listing"],
                description="Derived image sidecar that should not drive album title lookup.",
                album_title="Derived Image Listing",
                source_text="",
                ocr_text="DERIVED IMAGE LISTING",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                subphotos=[],
            )

            title = ai_index._resolve_album_title_hint(image)
            self.assertEqual(title, "")

    def test_resolve_album_title_hint_requires_existing_p01_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")

            title = ai_index._resolve_album_title_hint(image)
            self.assertEqual(title, "")

    def test_run_image_analysis_uses_explicit_album_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "China_1986_B02_P01_S01.tif"
            image.write_bytes(b"abc")

            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = SimpleNamespace(
                generate=mock.Mock(
                    return_value=SimpleNamespace(
                        text="敦煌 历史文物展览",
                        engine="template",
                        fallback=False,
                        error="",
                        ocr_text="敦煌 历史文物展览",
                        author_text="敦煌 历史文物展览",
                        scene_text="",
                        album_title="敦煌 历史文物展览",
                        title="",
                        location_name="",
                        people_present=False,
                        estimated_people_count=0,
                        image_regions=[],
                        ocr_lang="zh",
                    )
                ),
                effective_model_name="template-model",
                estimate_location=None,
            )

            @contextmanager
            def fake_prepare(path: Path):
                yield path

            with mock.patch.object(
                ai_index_analysis,
                "_prepare_ai_model_image",
                side_effect=fake_prepare,
            ):
                analysis = ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="none",
                    ocr_language="eng",
                )

            self.assertEqual(analysis.album_title, "敦煌 历史文物展览")

    def test_run_image_analysis_uses_title_page_ocr_text_when_album_title_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "China_1986_B02_P01_S01.tif"
            image.write_bytes(b"abc")

            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "MAINLAND CHINA 1986 BOOK 11"
            caption_engine = SimpleNamespace(
                generate=mock.Mock(
                    return_value=SimpleNamespace(
                        text="",
                        engine="template",
                        fallback=False,
                        error="",
                        ocr_text="MAINLAND CHINA 1986 BOOK 11",
                        author_text="MAINLAND CHINA 1986 BOOK 11",
                        scene_text="",
                        album_title="",
                        title="",
                        location_name="",
                        people_present=False,
                        estimated_people_count=0,
                        image_regions=[],
                        ocr_lang="zh",
                    )
                ),
                effective_model_name="template-model",
                estimate_location=None,
            )

            @contextmanager
            def fake_prepare(path: Path):
                yield path

            with mock.patch.object(
                ai_index_analysis,
                "_prepare_ai_model_image",
                side_effect=fake_prepare,
            ):
                analysis = ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="none",
                    ocr_language="eng",
                )

            self.assertEqual(analysis.album_title, "MAINLAND CHINA 1986 BOOK 11")

    def test_run_image_analysis_fails_when_title_page_has_no_album_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "EasternEuropeSpainMorocco_1988_B00_P01_S01.tif"
            image.write_bytes(b"abc")

            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = []
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = ""
            caption_engine = SimpleNamespace(
                generate=mock.Mock(
                    return_value=SimpleNamespace(
                        text="",
                        engine="template",
                        fallback=True,
                        error="caption empty",
                        ocr_text="",
                        author_text="",
                        scene_text="",
                        album_title="",
                        title="",
                        location_name="",
                        people_present=False,
                        estimated_people_count=0,
                        image_regions=[],
                        ocr_lang="eng",
                    )
                ),
                effective_model_name="template-model",
                estimate_location=None,
            )

            @contextmanager
            def fake_prepare(path: Path):
                yield path

            with (
                mock.patch.object(
                    ai_index_analysis,
                    "_prepare_ai_model_image",
                    side_effect=fake_prepare,
                ),
                self.assertRaises(RuntimeError) as exc,
            ):
                ai_index._run_image_analysis(
                    image_path=image,
                    people_matcher=people_matcher,
                    object_detector=object_detector,
                    ocr_engine=ocr_engine,
                    caption_engine=caption_engine,
                    requested_caption_engine="template",
                    ocr_engine_name="none",
                    ocr_language="eng",
                )

            self.assertIn("Missing album title for title page during analysis", str(exc.exception))

    def test_run_writes_title_page_album_title_from_current_ocr_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            image = archive / "China_1986_B02_P01_S01.tif"
            image.write_bytes(b"abc")
            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=["mainland", "china", "1986", "book"],
                subjects=["mainland", "china", "1986", "book"],
                description='Visible text reads: "MAINLAND CHINA 1986 BOOK 11".',
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["mainland", "china", "1986", "book"],
                        "chars": len("MAINLAND CHINA 1986 BOOK 11"),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
                album_title="",
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_page_scan_filenames", return_value=[image.name]),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["album_title"], "MAINLAND CHINA 1986 BOOK 11")

    def test_write_sidecar_and_record_uses_configured_title_page_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "China_1986_B02_P01_S01.tif"
            sidecar = image.with_suffix(".xmp")
            image.write_bytes(b"abc")

            with mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock:
                ai_index._write_sidecar_and_record(
                    sidecar,
                    image,
                    creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                    person_names=[],
                    subjects=[],
                    description="Title page",
                    album_title="Mainland China",
                    location_payload={
                        "query": "Wrong Place",
                        "gps_latitude": 0,
                        "gps_longitude": 0,
                        "city": "Wrong",
                    },
                    source_text="",
                    ocr_text="MAINLAND CHINA 1986 BOOK 11",
                    detections_payload={"location": {"query": "Wrong Place"}},
                    title_page_location={
                        "address": "2240 Lorain Rd, San Marino, CA 91108",
                        "gps_latitude": "34.11512",
                        "gps_longitude": "-118.10492",
                        "city": "San Marino",
                        "state": "CA",
                        "country": "United States",
                        "sublocation": "2240 Lorain Rd",
                    },
                )

            write_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["gps_latitude"], "34.11512")
            self.assertEqual(write_mock.call_args.kwargs["gps_longitude"], "-118.10492")
            self.assertEqual(write_mock.call_args.kwargs["location_city"], "San Marino")
            self.assertEqual(write_mock.call_args.kwargs["location_state"], "CA")
            self.assertEqual(write_mock.call_args.kwargs["location_country"], "United States")
            self.assertEqual(write_mock.call_args.kwargs["location_sublocation"], "2240 Lorain Rd")
            self.assertEqual(
                write_mock.call_args.kwargs["detections_payload"]["location"]["query"],
                "2240 Lorain Rd, San Marino, CA 91108",
            )

    def test_write_sidecar_and_record_clears_stale_title_page_location_for_non_p01(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "China_1986_B02_P02_S01.tif"
            sidecar = image.with_suffix(".xmp")
            image.write_bytes(b"abc")

            with mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock:
                ai_index._write_sidecar_and_record(
                    sidecar,
                    image,
                    creator_tool=ai_index.DEFAULT_CREATOR_TOOL,
                    person_names=[],
                    subjects=[],
                    description="Page",
                    album_title="Mainland China",
                    location_payload={
                        "query": "2240 Lorain Rd, San Marino, CA 91108",
                        "gps_latitude": 34.11512,
                        "gps_longitude": -118.10492,
                        "city": "San Marino",
                        "state": "CA",
                        "country": "United States",
                        "sublocation": "2240 Lorain Rd",
                        "source": ai_index.TITLE_PAGE_LOCATION_SOURCE,
                    },
                    source_text="",
                    ocr_text="PAGE 2",
                    detections_payload={
                        "location": {
                            "query": "2240 Lorain Rd, San Marino, CA 91108",
                            "source": ai_index.TITLE_PAGE_LOCATION_SOURCE,
                        }
                    },
                    title_page_location={
                        "address": "2240 Lorain Rd, San Marino, CA 91108",
                        "gps_latitude": "34.11512",
                        "gps_longitude": "-118.10492",
                        "city": "San Marino",
                        "state": "CA",
                        "country": "United States",
                        "sublocation": "2240 Lorain Rd",
                    },
                )

            write_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["gps_latitude"], "")
            self.assertEqual(write_mock.call_args.kwargs["gps_longitude"], "")
            self.assertEqual(write_mock.call_args.kwargs["location_city"], "")
            self.assertEqual(write_mock.call_args.kwargs["location_state"], "")
            self.assertEqual(write_mock.call_args.kwargs["location_country"], "")
            self.assertEqual(write_mock.call_args.kwargs["location_sublocation"], "")
            self.assertNotIn("location", write_mock.call_args.kwargs["detections_payload"])

    def test_resolve_album_printed_title_hint_prefers_existing_p01_cover_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P01.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="This is the cover or title page of Mainland China Book II, a Photo Essay.",
                album_title="Mainland China Book II",
                source_text="",
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                subphotos=[],
            )

            title = ai_index._resolve_album_printed_title_hint(image, {})
            self.assertEqual(title, "Mainland China Book II")

    def test_resolve_album_printed_title_hint_ignores_p00_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P00.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=[],
                description="Legacy P00 sidecar should no longer be used.",
                album_title="Mainland China Book II",
                source_text="",
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                subphotos=[],
            )

            title = ai_index._resolve_album_printed_title_hint(image, {})
            self.assertEqual(title, "")

    def test_resolve_album_printed_title_hint_ignores_p01_detail_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album_dir = base / "China_1986_B02_View"
            album_dir.mkdir()
            image = album_dir / "China_1986_B02_P02_stitched.jpg"
            image.write_bytes(b"abc")
            xmp_sidecar.write_xmp_sidecar(
                album_dir / "China_1986_B02_P01_D01-01.xmp",
                creator_tool="imago-test",
                person_names=[],
                subjects=["derived", "listing"],
                description="Derived image sidecar that should not drive printed title lookup.",
                album_title="Derived Image Listing",
                source_text="",
                ocr_text="DERIVED IMAGE LISTING",
                detections_payload={
                    "people": [],
                    "objects": [],
                    "ocr": {},
                    "caption": {},
                },
                subphotos=[],
            )

            title = ai_index._resolve_album_printed_title_hint(image, {})
            self.assertEqual(title, "")

    def test_run_stdout_forces_processing_even_when_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Caption from stdout mode",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "local"},
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(image),
                ),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()

    def test_run_archive_multi_scan_passes_stitched_ocr_override_and_records_authority_metadata(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            stitched_path = archive / "China_1986_B02_P02_stitched.jpg"
            authority = ai_index.ArchiveScanOCRAuthority(
                page_key=ai_index._scan_page_key(scan1) or "",
                group_paths=(scan1, scan2),
                signature=ai_index._scan_group_signature([scan1, scan2]),
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=("mainland", "china", "book"),
                ocr_hash=ai_index._hash_text("MAINLAND CHINA 1986 BOOK 11"),
                stitched_image_path=stitched_path,
            )
            analysis = ai_index.ImageAnalysis(
                image_path=stitched_path,
                people_names=[],
                object_labels=[],
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=["mainland", "china", "book"],
                subjects=["mainland", "china", "book"],
                description="Archive page caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["mainland", "china", "book"],
                        "chars": len("MAINLAND CHINA 1986 BOOK 11"),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(scan1),
                ),
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_archive_scan_authoritative_ocr",
                    return_value=authority,
                ) as authority_mock,
                mock.patch.object(ai_index_runner, "_page_scan_filenames", return_value=[]),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
                mock.patch.object(ai_index_runner, "_mirror_page_sidecars"),
                mock.patch.object(ai_index, "append_job_artifact") as append_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            authority_mock.assert_called_once()
            self.assertEqual(analysis_mock.call_args.kwargs["image_path"], stitched_path)
            self.assertEqual(analysis_mock.call_args.kwargs["people_image_path"], stitched_path)
            self.assertEqual(
                analysis_mock.call_args.kwargs["ocr_text_override"],
                "MAINLAND CHINA 1986 BOOK 11",
            )
            self.assertEqual(write_mock.call_args.kwargs["ocr_text"], analysis.ocr_text)
            self.assertEqual(
                write_mock.call_args.kwargs["ocr_authority_source"],
                "archive_stitched",
            )
            artifact_payload = append_mock.call_args.args[0]
            self.assertEqual(artifact_payload["label"], "China_1986_B02_P02")
            self.assertEqual(
                artifact_payload["sidecar_paths"],
                [
                    str(scan1.with_suffix(".xmp")),
                    str(scan2.with_suffix(".xmp")),
                ],
            )
            det = write_mock.call_args.kwargs["detections_payload"]
            self.assertEqual(det["processing"]["ocr_authority_signature"], authority.signature)
            self.assertEqual(det["processing"]["ocr_authority_hash"], authority.ocr_hash)

    def test_run_archive_multi_scan_preserves_caption_author_text_when_stitching(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            stitched_path = archive / "China_1986_B02_P02_stitched.jpg"
            authority = ai_index.ArchiveScanOCRAuthority(
                page_key=ai_index._scan_page_key(scan1) or "",
                group_paths=(scan1, scan2),
                signature=ai_index._scan_group_signature([scan1, scan2]),
                ocr_text="",
                ocr_keywords=(),
                ocr_hash="",
                stitched_image_path=stitched_path,
            )
            analysis = ai_index.ImageAnalysis(
                image_path=stitched_path,
                people_names=[],
                object_labels=[],
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=["mainland", "china", "book"],
                subjects=["mainland", "china", "book"],
                description="Archive page caption",
                author_text="MAINLAND CHINA 1986 BOOK 11",
                scene_text="NO SMOKING",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["mainland", "china", "book"],
                        "chars": len("MAINLAND CHINA 1986 BOOK 11"),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
            )
            caption_output = SimpleNamespace(
                text="Archive page caption",
                location_name="Archive page caption",
                author_text="MAINLAND CHINA 1986 BOOK 11",
                scene_text="NO SMOKING",
                gps_latitude="",
                gps_longitude="",
                fallback=False,
            )
            fake_caption_engine = SimpleNamespace(
                effective_model_name="fake-caption-model",
                generate=mock.Mock(return_value=caption_output),
                estimate_location=mock.Mock(
                    return_value=SimpleNamespace(
                        fallback=False,
                        gps_latitude="",
                        gps_longitude="",
                        location_name="Archive page caption",
                    )
                ),
            )
            recorded_artifacts: list[dict[str, object]] = []

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(scan1),
                ),
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_archive_scan_authoritative_ocr",
                    return_value=authority,
                ) as authority_mock,
                mock.patch.object(ai_index_runner, "_page_scan_filenames", return_value=[]),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index_runner, "_resolve_dc_date", return_value=""),
                mock.patch.object(ai_index_engine_cache, "CaptionEngine", return_value=fake_caption_engine),
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
                mock.patch.object(ai_index_runner, "_mirror_page_sidecars"),
                mock.patch.object(
                    ai_index_runner,
                    "_emit_prompt_debug_artifact",
                    side_effect=lambda pd, dry_run=False: (
                        recorded_artifacts.append(
                            {
                                "kind": "photoalbums_prompts",
                                "image_path": str(getattr(pd, "image_path", "")),
                                "label": str(getattr(pd, "label", "")),
                            }
                        )
                        if pd is not None
                        else None
                    ),
                ),
                mock.patch.object(
                    ai_index,
                    "append_job_artifact",
                    side_effect=lambda record: recorded_artifacts.append(record),
                ),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "lmstudio",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            authority_mock.assert_called_once()
            analysis_mock.assert_called_once()
            self.assertEqual(write_mock.call_args.kwargs["author_text"], "MAINLAND CHINA 1986 BOOK 11")
            self.assertEqual(write_mock.call_args.kwargs["scene_text"], "NO SMOKING")
            prompt_artifact = next(
                row for row in recorded_artifacts if str(row.get("kind") or "") == "photoalbums_prompts"
            )
            self.assertEqual(prompt_artifact["image_path"], str(scan1))
            self.assertEqual(prompt_artifact["label"], "China_1986_B02_P02")

    def test_run_archive_multi_scan_skips_when_authority_manifest_and_sidecar_match(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            stitched_path = archive / "China_1986_B02_P02_stitched.jpg"
            authority = ai_index.ArchiveScanOCRAuthority(
                page_key=ai_index._scan_page_key(scan1) or "",
                group_paths=(scan1, scan2),
                signature=ai_index._scan_group_signature([scan1, scan2]),
                ocr_text="",
                ocr_keywords=(),
                ocr_hash="",
                stitched_image_path=stitched_path,
            )
            analysis = ai_index.ImageAnalysis(
                image_path=stitched_path,
                people_names=[],
                object_labels=[],
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=["mainland", "china", "book"],
                subjects=["mainland", "china", "book"],
                description="Archive page caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["mainland", "china", "book"],
                        "chars": len("MAINLAND CHINA 1986 BOOK 11"),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(scan1),
                ),
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_archive_scan_authoritative_ocr",
                    return_value=authority,
                ),
                mock.patch.object(ai_index_runner, "_page_scan_filenames", return_value=[]),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
            ):
                first_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                        "--force",
                    ]
                )

            self.assertEqual(first_result, 0)

            with (
                mock.patch.object(ai_index_runner, "_resolve_archive_scan_authoritative_ocr") as authority_mock,
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                second_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(second_result, 0)
            authority_mock.assert_not_called()
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_archive_multi_scan_skips_without_authority_when_sidecar_has_authority(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "China_1986_B02_Archive"
            archive.mkdir()
            scan1 = archive / "China_1986_B02_P02_S01.tif"
            scan2 = archive / "China_1986_B02_P02_S02.tif"
            scan1.write_bytes(b"a")
            scan2.write_bytes(b"b")
            stitched_path = archive / "China_1986_B02_P02_stitched.jpg"
            authority = ai_index.ArchiveScanOCRAuthority(
                page_key=ai_index._scan_page_key(scan1) or "",
                group_paths=(scan1, scan2),
                signature=ai_index._scan_group_signature([scan1, scan2]),
                ocr_text="",
                ocr_keywords=(),
                ocr_hash="",
                stitched_image_path=stitched_path,
            )
            analysis = ai_index.ImageAnalysis(
                image_path=stitched_path,
                people_names=[],
                object_labels=[],
                ocr_text="MAINLAND CHINA 1986 BOOK 11",
                ocr_keywords=["mainland", "china", "book"],
                subjects=["mainland", "china", "book"],
                description="Archive page caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {
                        "engine": "local",
                        "language": "eng",
                        "keywords": ["mainland", "china", "book"],
                        "chars": len("MAINLAND CHINA 1986 BOOK 11"),
                    },
                    "caption": {
                        "requested_engine": "none",
                        "effective_engine": "none",
                        "fallback": False,
                        "error": "",
                        "model": "",
                    },
                },
            )

            with (
                mock.patch.object(
                    ai_index_runner,
                    "prepare_image_layout",
                    side_effect=lambda *args, **kwargs: self._mock_layout(scan1),
                ),
                mock.patch.object(
                    ai_index_runner,
                    "_resolve_archive_scan_authoritative_ocr",
                    return_value=authority,
                ),
                mock.patch.object(ai_index_runner, "_page_scan_filenames", return_value=[]),
                mock.patch.object(ai_index_runner, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index_runner, "_build_flat_payload", return_value=analysis.payload),
            ):
                first_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                        "--force",
                    ]
                )

            self.assertEqual(first_result, 0)

            with (
                mock.patch.object(ai_index_runner, "_resolve_archive_scan_authoritative_ocr") as authority_mock,
                mock.patch.object(ai_index_runner, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index_runner, "write_xmp_sidecar") as write_mock,
            ):
                second_result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--include-archive",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "local",
                        "--caption-engine",
                        "none",
                        "--max-images",
                        "1",
                    ]
                )

            self.assertEqual(second_result, 0)
            authority_mock.assert_not_called()
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_resolve_caption_prompt_reads_file_and_overrides_inline_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("Describe this image from file.\n", encoding="utf-8")
            text = ai_index._resolve_caption_prompt("Inline prompt", str(prompt_file))
        self.assertEqual(text, "Describe this image from file.")

    def test_resolve_caption_prompt_exits_for_missing_file(self):
        with self.assertRaises(SystemExit) as exc:
            ai_index._resolve_caption_prompt("", "/tmp/definitely-missing-caption-prompt.txt")
        self.assertIn("Caption prompt file does not exist", str(exc.exception))

    def test_parse_args_caption_flags(self):
        args = ai_index.parse_args(
            [
                "--ocr-model",
                "qwen2.5-vl-instruct",
                "--caption-engine",
                "lmstudio",
                "--caption-model",
                "qwen2.5-vl-instruct",
                "--caption-prompt",
                "Describe this exact image",
                "--caption-prompt-file",
                "/tmp/prompt.txt",
                "--lmstudio-base-url",
                "http://localhost:1234",
                "--caption-max-tokens",
                "64",
                "--caption-temperature",
                "0.1",
                "--caption-max-edge",
                "1024",
            ]
        )
        self.assertEqual(args.ocr_model, "qwen2.5-vl-instruct")
        self.assertEqual(args.caption_engine, "lmstudio")
        self.assertEqual(args.caption_model, "qwen2.5-vl-instruct")
        self.assertEqual(args.caption_prompt, "Describe this exact image")
        self.assertEqual(args.caption_prompt_file, "/tmp/prompt.txt")
        self.assertEqual(args.lmstudio_base_url, "http://localhost:1234")
        self.assertEqual(args.caption_max_tokens, 64)
        self.assertAlmostEqual(args.caption_temperature, 0.1)
        self.assertEqual(args.caption_max_edge, 1024)

    def test_parse_args_shard_flags(self):
        args = ai_index.parse_args(
            [
                "--shard-count",
                "3",
                "--shard-index",
                "1",
            ]
        )
        self.assertEqual(args.shard_count, 3)
        self.assertEqual(args.shard_index, 1)

    def test_apply_shard_keeps_album_on_single_worker(self):
        files = [
            Path("China_1986_B02_Archive/China_1986_B02_P01_S01.tif"),
            Path("China_1986_B02_Archive/China_1986_B02_P02_S01.tif"),
            Path("Morocco_1988_B01_Archive/Morocco_1988_B01_P01_S01.tif"),
            Path("Morocco_1988_B01_Archive/Morocco_1988_B01_P02_S01.tif"),
        ]

        shard_zero = ai_index._apply_shard(files, 2, 0)
        shard_one = ai_index._apply_shard(files, 2, 1)

        self.assertEqual(sorted(str(path) for path in shard_zero + shard_one), sorted(str(path) for path in files))
        for album_key in {ai_index._album_identity_key(path) for path in files}:
            in_zero = [path for path in shard_zero if ai_index._album_identity_key(path) == album_key]
            in_one = [path for path in shard_one if ai_index._album_identity_key(path) == album_key]
            self.assertTrue(bool(in_zero) ^ bool(in_one))

    def test_parse_args_defaults_use_lmstudio_for_caption_and_disable_ocr(self):
        with (
            mock.patch.object(ai_index_args, "default_ocr_model", return_value="qwen/qwen3-vl-30b"),
            mock.patch.object(ai_index_args, "default_lmstudio_base_url", return_value="http://lmstudio.local:1234/v1"),
        ):
            args = ai_index.parse_args([])
        self.assertEqual(args.caption_engine, "lmstudio")
        self.assertEqual(args.caption_model, "")
        self.assertEqual(args.caption_prompt, "")
        self.assertEqual(args.caption_prompt_file, "")
        self.assertEqual(args.lmstudio_base_url, "http://lmstudio.local:1234/v1")
        self.assertEqual(args.ocr_engine, "none")
        self.assertEqual(args.ocr_model, "qwen/qwen3-vl-30b")
        self.assertFalse(args.stdout)
        self.assertEqual(args.caption_max_edge, 0)

    def test_init_caption_engine_forwards_caption_prompt(self):
        with mock.patch.object(ai_index_engine_cache, "CaptionEngine") as engine_ctor:
            ai_index._init_caption_engine(
                engine="lmstudio",
                model_name="qwen2.5-vl-instruct",
                caption_prompt="Describe this exact image",
                max_tokens=64,
                temperature=0.1,
                lmstudio_base_url="http://localhost:1234",
                max_image_edge=1024,
            )

        engine_ctor.assert_called_once_with(
            engine="lmstudio",
            model_name="qwen2.5-vl-instruct",
            caption_prompt="Describe this exact image",
            max_tokens=64,
            temperature=0.1,
            lmstudio_base_url="http://localhost:1234",
            max_image_edge=1024,
            stream=False,
        )


if __name__ == "__main__":
    unittest.main()
