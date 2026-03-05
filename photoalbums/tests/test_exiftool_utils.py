import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exiftool_utils


class TestExiftoolUtils(unittest.TestCase):
    def test_read_tag_success(self):
        with mock.patch("exiftool_utils.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(stdout="value\n")
            result = exiftool_utils.read_tag("a.jpg", "XMP-dc:Creator")
        self.assertEqual(result, "value")

    def test_read_tag_failure(self):
        with mock.patch(
            "exiftool_utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "exiftool"),
        ):
            result = exiftool_utils.read_tag("a.jpg", "XMP-dc:Creator")
        self.assertIsNone(result)

    def test_write_tags_builds_expected_command(self):
        with mock.patch("exiftool_utils.subprocess.run") as run_mock:
            exiftool_utils.write_tags(
                "a.jpg",
                set_tags={"XMP-dc:Creator": "A", "XMP-dc:Description": "B"},
                clear_tags=["XMP-dc:Subject"],
            )
        command = run_mock.call_args[0][0]
        self.assertEqual(command[0], "exiftool")
        self.assertIn("-overwrite_original", command)
        self.assertIn("-XMP-dc:Subject=", command)
        self.assertIn("-XMP-dc:Creator=A", command)
        self.assertIn("-XMP-dc:Description=B", command)
        self.assertEqual(command[-1], "a.jpg")

    def test_read_json_tags_returns_first_object(self):
        payload = [{"XMP-dc:Creator": "Audrey"}]
        with mock.patch("exiftool_utils.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(stdout=json.dumps(payload))
            result = exiftool_utils.read_json_tags(Path("a.jpg"), ["XMP-dc:Creator"])
        self.assertEqual(result, payload[0])


if __name__ == "__main__":
    unittest.main()
