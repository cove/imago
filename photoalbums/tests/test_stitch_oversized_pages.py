import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stitch_oversized_pages as sop


class TestStitchOversizedPages(unittest.TestCase):
    def test_build_scans_text(self):
        self.assertEqual(sop.build_scans_text([1, 2, 10]), "S01 S02 S10")

    def test_build_scan_header(self):
        header = sop.build_scan_header("EU", "1973", "02", 5, [1, 2])
        self.assertEqual(
            header,
            "EU (1973) - Book 02, Page 05, Scans S01 S02",
        )

    def test_extract_scan_numbers(self):
        files = [
            "EU_1973_B02_P05_S01.tif",
            "EU_1973_B02_P05_S02.tif",
            "no_scan_here.tif",
        ]
        self.assertEqual(sop.extract_scan_numbers(files), [1, 2])

    def test_build_derived_output_name_known(self):
        name = "EU_1973_B02_P05_D01_02.tif"
        self.assertEqual(
            sop.build_derived_output_name(name),
            "EU_1973_B02_P05_D01_02.jpg",
        )

    def test_build_derived_output_name_unknown(self):
        name = "EU_1973_Custom_D01_02.tif"
        self.assertEqual(
            sop.build_derived_output_name(name),
            "EU_1973_Custom_D01_02_D01_02.jpg",
        )

    def test_get_view_dirname(self):
        base = Path("C:/Photos/EU_1973_B02_Archive")
        view = sop.get_view_dirname(base)
        self.assertEqual(Path(view), Path("C:/Photos/EU_1973_B02_View"))


if __name__ == "__main__":
    unittest.main()
