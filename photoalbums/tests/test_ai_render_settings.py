import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_render_settings as ars


class TestAIRenderSettings(unittest.TestCase):
    def test_find_archive_dir_for_archive_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Family_Archive"
            archive.mkdir()
            image = archive / "a.jpg"
            image.write_bytes(b"x")
            self.assertEqual(ars.find_archive_dir_for_image(image), archive)

    def test_find_archive_dir_for_view_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Family_Archive"
            view = Path(tmp) / "Family_View"
            archive.mkdir()
            view.mkdir()
            image = view / "b.jpg"
            image.write_bytes(b"x")
            self.assertEqual(ars.find_archive_dir_for_image(image), archive)

    def test_load_and_resolve_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Family_Archive"
            archive.mkdir()
            image = archive / "img1.jpg"
            image.write_bytes(b"x")

            defaults = {
                "skip": False,
                "enable_people": True,
                "enable_objects": True,
                "ocr_engine": "qwen",
                "ocr_lang": "eng",
                "page_split_mode": "auto",
                "caption_engine": "qwen",
                "caption_model": "",
                "caption_prompt": "",
                "caption_max_tokens": 96,
                "caption_temperature": 0.2,
                "caption_max_edge": 0,
                "lmstudio_base_url": "http://127.0.0.1:1234/v1",
                "qwen_attn_implementation": "auto",
                "qwen_min_pixels": 0,
                "qwen_max_pixels": 0,
                "people_threshold": 0.72,
                "object_threshold": 0.30,
                "min_face_size": 40,
                "model": "models/yolo11n.pt",
                "creator_tool": "tool-default",
            }

            payload = {
                "archive_settings": {
                    "enable_people": False,
                    "ocr_engine": "none",
                    "model": "yolo11x.pt",
                    "page_split_mode": "off",
                    "caption_engine": "blip",
                    "qwen_attn_implementation": "sdpa",
                    "qwen_max_pixels": 262144,
                    "qwen_prompt": "Legacy prompt alias",
                },
                "image_settings": {
                    "img1.jpg": {
                        "enable_people": True,
                        "people_threshold": 0.88,
                        "creator_tool": "tool-image",
                        "caption_max_edge": 960,
                        "lmstudio_base_url": "http://localhost:1234",
                    }
                },
            }
            settings_path = archive / "render_settings.json"
            settings_path.write_text(json.dumps(payload), encoding="utf-8")

            _path, loaded = ars.load_render_settings(
                archive, defaults=defaults, create=False
            )
            effective = ars.resolve_effective_settings(
                image, defaults=defaults, loaded=loaded
            )
            self.assertTrue(effective["enable_people"])
            self.assertTrue(effective["enable_objects"])
            self.assertEqual(effective["ocr_engine"], "none")
            self.assertEqual(effective["page_split_mode"], "off")
            self.assertEqual(effective["caption_engine"], "qwen")
            self.assertEqual(effective["qwen_attn_implementation"], "sdpa")
            self.assertEqual(effective["qwen_max_pixels"], 262144)
            self.assertEqual(effective["caption_prompt"], "Legacy prompt alias")
            self.assertEqual(effective["caption_max_edge"], 960)
            self.assertEqual(effective["lmstudio_base_url"], "http://localhost:1234")
            self.assertAlmostEqual(effective["people_threshold"], 0.88)
            self.assertEqual(effective["model"], "yolo11x.pt")
            self.assertEqual(effective["creator_tool"], "tool-image")

    def test_load_and_resolve_settings_preserves_lmstudio_ocr_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Family_Archive"
            archive.mkdir()
            image = archive / "img1.jpg"
            image.write_bytes(b"x")

            defaults = {
                "ocr_engine": "lmstudio",
                "ocr_lang": "eng",
                "caption_engine": "lmstudio",
                "lmstudio_base_url": "http://127.0.0.1:1234/v1",
            }
            settings_path = archive / "render_settings.json"
            settings_path.write_text(
                json.dumps({"archive_settings": {"ocr_engine": "lmstudio"}}),
                encoding="utf-8",
            )

            _path, loaded = ars.load_render_settings(
                archive, defaults=defaults, create=False
            )
            effective = ars.resolve_effective_settings(
                image, defaults=defaults, loaded=loaded
            )

            self.assertEqual(effective["ocr_engine"], "lmstudio")

    def test_resolve_effective_settings_defaults_page_split_mode_to_off(self):
        effective = ars.resolve_effective_settings(
            Path("Family_View") / "Family_1980-1985_B08_P01.jpg",
            defaults={},
            loaded=None,
        )
        self.assertEqual(effective["page_split_mode"], "off")


if __name__ == "__main__":
    unittest.main()
