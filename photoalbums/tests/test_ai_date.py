import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_date


class TestAIDate(unittest.TestCase):
    def test_parse_date_estimate_coerces_zero_day_to_month_precision(self):
        self.assertEqual(
            ai_date._parse_date_estimate('{"date":"1988-01-00"}'),
            "1988-01",
        )

    def test_parse_date_estimate_coerces_zero_month_and_day_to_year_precision(self):
        self.assertEqual(
            ai_date._parse_date_estimate('{"date":"1988-00-00"}'),
            "1988",
        )

    def test_parse_date_estimate_rejects_invalid_nonzero_month(self):
        with self.assertRaisesRegex(RuntimeError, "invalid dc:date value"):
            ai_date._parse_date_estimate('{"date":"1988-13-00"}')


if __name__ == "__main__":
    unittest.main()
