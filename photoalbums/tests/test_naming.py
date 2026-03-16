import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import naming


class TestNaming(unittest.TestCase):
    def test_parse_album_filename_scan(self):
        value = naming.parse_album_filename("EU_1973_B02_P05_S01.tif")
        self.assertEqual(value, ("EU", "1973", "02", "05"))

    def test_parse_album_filename_derived(self):
        value = naming.parse_album_filename("EU_1973_B02_P05_D01_02.tif")
        self.assertEqual(value, ("EU", "1973", "02", "05"))

    def test_parse_album_filename_fallback(self):
        value = naming.parse_album_filename("unknown_name.jpg")
        self.assertEqual(value, ("Unknown", "Unknown", "00", "00"))

    def test_format_book_display_numeric(self):
        self.assertEqual(naming.format_book_display("2"), "02")

    def test_format_book_display_unknown_tokens(self):
        self.assertEqual(
            naming.format_book_display(naming.ELLIPSIS_BOOK), naming.ELLIPSIS_BOOK
        )
        self.assertEqual(
            naming.format_book_display(naming.LEGACY_ELLIPSIS_BOOK),
            naming.ELLIPSIS_BOOK,
        )


if __name__ == "__main__":
    unittest.main()
