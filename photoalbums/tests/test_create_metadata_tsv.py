import sys
import unittest
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import create_metadata_tsv


class TestCreateMetadataTSV(unittest.TestCase):
    def test_main_returns_error_for_deprecated_command(self):
        stderr = StringIO()
        original_stderr = sys.stderr
        try:
            sys.stderr = stderr
            result = create_metadata_tsv.main()
        finally:
            sys.stderr = original_stderr

        self.assertEqual(result, 1)
        self.assertIn("deprecated", stderr.getvalue())
        self.assertIn("metadata.tsv", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
