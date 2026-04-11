from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums import commands
from photoalbums.lib.ai_ctm_restoration import CTMResult


class TestRunCTM(unittest.TestCase):
    def test_generate_uses_stitched_view_image_and_archive_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "Family_2020_B01_Archive"
            view_dir = root / "Family_2020_B01_View"
            archive_dir.mkdir()
            view_dir.mkdir()
            scan = archive_dir / "Family_2020_B01_P01_S01.tif"
            view = view_dir / "Family_2020_B01_P01_V.jpg"
            scan.write_bytes(b"scan")
            view.write_bytes(b"view")

            result = CTMResult(
                matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                confidence=0.8,
                warnings=[],
                reasoning_summary="ok",
                model_name="gemma",
                source_path=str(view),
            )

            with (
                mock.patch("photoalbums.stitch_oversized_pages.list_archive_dirs", return_value=[str(archive_dir)]),
                mock.patch("photoalbums.stitch_oversized_pages.list_page_scans", return_value=[[str(scan)]]),
                mock.patch("photoalbums.stitch_oversized_pages._require_primary_scan", return_value=str(scan)),
                mock.patch(
                    "photoalbums.lib.ai_ctm_restoration.generate_and_store_ctm",
                    return_value=(scan.with_suffix(".xmp"), result),
                ) as generate_mock,
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = commands.run_ctm(["generate", "--photos-root", str(root)])

            self.assertEqual(exit_code, 0)
            generate_mock.assert_called_once_with(
                view,
                archive_sidecar_path=scan.with_suffix(".xmp"),
                force=False,
            )
            payload = json.loads(stdout.getvalue().strip())
            self.assertEqual(payload["image"], scan.name)
            self.assertEqual(payload["source_image"], str(view))
            self.assertEqual(payload["archive_xmp"], str(scan.with_suffix(".xmp")))


if __name__ == "__main__":
    unittest.main()
