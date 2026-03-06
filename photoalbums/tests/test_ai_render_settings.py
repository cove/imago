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
                "ocr_engine": "docstrange",
                "ocr_lang": "eng",
                "people_threshold": 0.72,
                "object_threshold": 0.30,
                "min_face_size": 40,
                "model": "yolo11n.pt",
                "creator_tool": "tool-default",
            }

            payload = {
                "archive_settings": {
                    "enable_people": False,
                    "ocr_engine": "none",
                    "model": "yolo11x.pt",
                },
                "image_settings": {
                    "img1.jpg": {
                        "enable_people": True,
                        "people_threshold": 0.88,
                        "creator_tool": "tool-image",
                    }
                },
            }
            settings_path = archive / "render_settings.json"
            settings_path.write_text(json.dumps(payload), encoding="utf-8")

            _path, loaded = ars.load_render_settings(archive, defaults=defaults, create=False)
            effective = ars.resolve_effective_settings(image, defaults=defaults, loaded=loaded)
            self.assertTrue(effective["enable_people"])
            self.assertTrue(effective["enable_objects"])
            self.assertEqual(effective["ocr_engine"], "none")
            self.assertAlmostEqual(effective["people_threshold"], 0.88)
            self.assertEqual(effective["model"], "yolo11x.pt")
            self.assertEqual(effective["creator_tool"], "tool-image")

    def test_legacy_tesseract_setting_falls_back_to_default_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Family_Archive"
            archive.mkdir()
            image = archive / "img1.jpg"
            image.write_bytes(b"x")

            defaults = {
                "skip": False,
                "enable_people": True,
                "enable_objects": True,
                "ocr_engine": "docstrange",
                "ocr_lang": "eng",
                "people_threshold": 0.72,
                "object_threshold": 0.30,
                "min_face_size": 40,
                "model": "yolo11n.pt",
                "creator_tool": "tool-default",
            }
            payload = {"archive_settings": {"ocr_engine": "tesseract"}}
            (archive / "render_settings.json").write_text(json.dumps(payload), encoding="utf-8")

            _path, loaded = ars.load_render_settings(archive, defaults=defaults, create=False)
            effective = ars.resolve_effective_settings(image, defaults=defaults, loaded=loaded)
            self.assertEqual(effective["ocr_engine"], "docstrange")


if __name__ == "__main__":
    unittest.main()
