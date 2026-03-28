import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stitch_oversized_pages_validate as sopv


class DummyStitcher:
    def __init__(self, **_cfg):
        pass

    def stitch(self, _files):
        class Result:
            size = 1

        return Result()


class TestStitchOversizedPagesValidate(unittest.TestCase):
    def test_parse_album_filename(self):
        collection, year, book, page = sopv.parse_album_filename("EU_1973_B02_P05_S01.tif")
        self.assertEqual((collection, year, book, page), ("EU", "1973", "02", "05"))

    def test_validate_stitch_with_stub(self):
        sopv.validate_stitch(["a.tif", "b.tif"], stitcher_factory=DummyStitcher)


if __name__ == "__main__":
    unittest.main()
