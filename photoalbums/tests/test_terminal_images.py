import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import terminal_images  # noqa: E402


class TestTerminalImages(unittest.TestCase):
    def test_display_inline_image_returns_false_without_chafa(self):
        with mock.patch("terminal_images.shutil.which", return_value=None):
            self.assertFalse(terminal_images.display_inline_image("scan.tif"))

    def test_display_inline_image_uses_sixels_in_windows_terminal(self):
        with (
            mock.patch("terminal_images.shutil.which", return_value="chafa"),
            mock.patch("terminal_images.shutil.get_terminal_size", return_value=os.terminal_size((160, 50))),
            mock.patch.dict("terminal_images.os.environ", {"WT_SESSION": "1"}, clear=True),
            mock.patch("terminal_images.sys.platform", "win32"),
            mock.patch("terminal_images.subprocess.run") as run_mock,
        ):
            self.assertTrue(terminal_images.display_inline_image("scan.tif", title="Scan"))

        run_mock.assert_called_once_with(
            ["chafa", "--animate=off", "--size=120x36", "--format=sixels", "scan.tif"],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
