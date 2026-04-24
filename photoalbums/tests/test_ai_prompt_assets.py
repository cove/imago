import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_prompt_assets


class TestAIPromptAssets(unittest.TestCase):
    def test_load_prompt_renders_variables_and_hashes_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "step" / "user.md"
            prompt.parent.mkdir()
            prompt.write_text("Album: {album_title}\nOCR: {ocr_text}\nStatic line\n", encoding="utf-8")

            with mock.patch.object(ai_prompt_assets, "PROMPT_ROOT", root):
                asset = ai_prompt_assets.load_prompt(
                    "step/user.md",
                    {"album_title": "Egypt 1975", "ocr_text": ""},
                )

        self.assertEqual(asset.rendered, "Album: Egypt 1975\nStatic line")
        self.assertEqual(asset.hash, ai_prompt_assets.content_hash("Album: {album_title}\nOCR: {ocr_text}\nStatic line\n"))
        self.assertEqual(asset.path, prompt)

    def test_missing_prompt_file_includes_path_and_os_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(ai_prompt_assets, "PROMPT_ROOT", root):
                with self.assertRaises(ai_prompt_assets.PromptAssetError) as exc:
                    ai_prompt_assets.load_prompt("missing/user.md")

        self.assertIn("missing", str(exc.exception))
        self.assertIn("Could not stat prompt asset", str(exc.exception))
        self.assertRegex(str(exc.exception), r"WinError|No such file|cannot find")

    def test_invalid_toml_includes_parse_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = root / "step" / "params.toml"
            params.parent.mkdir()
            params.write_text("max_tokens = [", encoding="utf-8")

            with mock.patch.object(ai_prompt_assets, "PROMPT_ROOT", root):
                with self.assertRaises(ai_prompt_assets.PromptAssetError) as exc:
                    ai_prompt_assets.load_params("step/params.toml")

        self.assertIn(str(params), str(exc.exception))
        self.assertIn("Invalid value", str(exc.exception))

    def test_hash_changes_when_file_changes_and_cache_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "step" / "user.md"
            prompt.parent.mkdir()
            prompt.write_text("first", encoding="utf-8")

            with mock.patch.object(ai_prompt_assets, "PROMPT_ROOT", root):
                first = ai_prompt_assets.load_prompt("step/user.md")
                time.sleep(0.01)
                prompt.write_text("second", encoding="utf-8")
                second = ai_prompt_assets.load_prompt("step/user.md")

        self.assertEqual(first.rendered, "first")
        self.assertEqual(second.rendered, "second")
        self.assertNotEqual(first.hash, second.hash)


if __name__ == "__main__":
    unittest.main()
