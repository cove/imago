import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import album_sets


class TestAlbumSets(unittest.TestCase):
    def test_cordell_set_exposes_title_page_location(self):
        album_set = album_sets.resolve_archive_set("cordell")
        self.assertIsNotNone(album_set.title_page_location)
        self.assertEqual(album_set.title_page_location["address"], "2240 Lorain Rd, San Marino, CA 91108")
        self.assertEqual(album_set.title_page_location["gps_latitude"], "34.11512")
        self.assertEqual(album_set.title_page_location["gps_longitude"], "-118.10492")


if __name__ == "__main__":
    unittest.main()
