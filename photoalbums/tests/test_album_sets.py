import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_read_people_roster_filters_empty_values(self):
        self.assertEqual(
            album_sets.read_people_roster("cordell"),
            {
                "audrey": "Audrey Cordell",
                "leslie": "Leslie Cordell",
            },
        )

    def test_read_people_roster_returns_empty_dict_when_table_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "album_sets.toml"
            config_path.write_text(
                "\n".join(
                    [
                        'default_archive_set = "archive"',
                        'default_scan_set = "scan"',
                        "",
                        "[sets.archive]",
                        'kind = "archive"',
                        'photos_root = "C:/Archive"',
                        "",
                        "[sets.scan]",
                        'kind = "scanwatch"',
                        'photos_root = "C:/Scan"',
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.object(album_sets, "ALBUM_SETS_PATH", config_path):
                self.assertEqual(album_sets.read_people_roster("archive"), {})


if __name__ == "__main__":
    unittest.main()
