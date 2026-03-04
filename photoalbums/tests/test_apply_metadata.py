import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import apply_metadata


class TestApplyMetadata(unittest.TestCase):
    def test_build_header(self):
        header = apply_metadata.build_header(
            "EU",
            "1973",
            "02",
            5,
            20,
            2,
            3,
        )
        self.assertEqual(
            header,
            "EU (1973) - Book 02, Page 05 of 20, Scan S02 of 3 total",
        )

    def test_update_tif_metadata_no_change(self):
        header = "EU (1973) - Book 02, Page 05 of 20, Scan S02 of 3 total"
        with mock.patch(
            "apply_metadata.get_tif_tag",
            side_effect=[header, apply_metadata.CREATOR],
        ), mock.patch("apply_metadata.subprocess.run") as run_mock:
            result = apply_metadata.update_tif_metadata(Path("test.tif"), header)

        self.assertFalse(result)
        run_mock.assert_not_called()

    def test_update_tif_metadata_duplicate_creator(self):
        header = "EU (1973) - Book 02, Page 05 of 20, Scan S02 of 3 total"
        dup_creator = f"{apply_metadata.CREATOR}; {apply_metadata.CREATOR}"
        with mock.patch(
            "apply_metadata.get_tif_tag",
            side_effect=[header, dup_creator],
        ), mock.patch("apply_metadata.subprocess.run") as run_mock:
            result = apply_metadata.update_tif_metadata(Path("test.tif"), header)

        self.assertTrue(result)
        self.assertEqual(run_mock.call_count, 2)
        first_args = run_mock.call_args_list[0][0][0]
        second_args = run_mock.call_args_list[1][0][0]
        self.assertIn("-XMP-dc:Creator=", first_args)
        self.assertIn(f"-XMP-dc:Creator={apply_metadata.CREATOR}", second_args)
        self.assertIn(f"-XMP-dc:Description={header}", second_args)


if __name__ == "__main__":
    unittest.main()
