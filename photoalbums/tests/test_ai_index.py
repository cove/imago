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

from photoalbums.lib import ai_index


class TestAIIndex(unittest.TestCase):
    def test_discover_images_archive_and_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Family_Archive"
            view = base / "Family_View"
            archive.mkdir()
            view.mkdir()
            (archive / "a.jpg").write_bytes(b"a")
            (view / "b.png").write_bytes(b"b")
            (base / "other.jpg").write_bytes(b"c")

            files = ai_index.discover_images(
                base,
                include_archive=True,
                include_view=False,
                extensions={".jpg", ".png"},
            )
            self.assertEqual([p.name for p in files], ["a.jpg"])

            files = ai_index.discover_images(
                base,
                include_archive=False,
                include_view=True,
                extensions={".jpg", ".png"},
            )
            self.assertEqual([p.name for p in files], ["b.png"])

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = {
                "/a.jpg": {"image_path": "/a.jpg", "size": 1, "mtime_ns": 2},
                "/b.jpg": {"image_path": "/b.jpg", "size": 3, "mtime_ns": 4},
            }
            ai_index.save_manifest(path, rows)
            loaded = ai_index.load_manifest(path)
            self.assertEqual(loaded, rows)

            raw = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(raw), 2)
            self.assertTrue(all(isinstance(json.loads(line), dict) for line in raw))

    def test_needs_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text("existing", encoding="utf-8")
            stat = image.stat()
            row = {
                "image_path": str(image),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sidecar_path": str(image.with_suffix(".xmp")),
            }
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

            image.write_bytes(b"abcd")
            self.assertTrue(ai_index.needs_processing(image, row, force=False))

    def test_needs_processing_requires_current_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            stat = image.stat()
            row = {
                "image_path": str(image),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sidecar_path": str(image.with_suffix(".xmp")),
            }
            self.assertTrue(ai_index.needs_processing(image, row, force=False))

            image.with_suffix(".xmp").write_text("existing", encoding="utf-8")
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

    def test_build_description(self):
        text = ai_index.build_description(
            people=["Alice", "Bob"],
            objects=["dog", "car"],
            ocr_text="Hello world from a sign",
        )
        self.assertIn("Alice", text)
        self.assertIn("dog", text)
        self.assertIn("Visible text reads:", text)

    def test_parse_args_caption_flags(self):
        args = ai_index.parse_args(
            [
                "--caption-engine",
                "qwen",
                "--caption-model",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "--caption-max-tokens",
                "64",
                "--caption-temperature",
                "0.1",
                "--caption-max-edge",
                "1024",
                "--qwen-attn-implementation",
                "sdpa",
                "--qwen-min-pixels",
                "131072",
                "--qwen-max-pixels",
                "524288",
            ]
        )
        self.assertEqual(args.caption_engine, "qwen")
        self.assertEqual(args.caption_model, "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(args.caption_max_tokens, 64)
        self.assertAlmostEqual(args.caption_temperature, 0.1)
        self.assertEqual(args.caption_max_edge, 1024)
        self.assertEqual(args.qwen_attn_implementation, "sdpa")
        self.assertEqual(args.qwen_min_pixels, 131072)
        self.assertEqual(args.qwen_max_pixels, 524288)

    def test_parse_args_defaults_use_blip_and_docstrange(self):
        args = ai_index.parse_args([])
        self.assertEqual(args.caption_engine, "blip")
        self.assertEqual(args.caption_model, "")
        self.assertEqual(args.ocr_engine, "docstrange")
        self.assertEqual(args.qwen_attn_implementation, "auto")
        self.assertEqual(args.qwen_min_pixels, 0)
        self.assertEqual(args.qwen_max_pixels, 0)
        self.assertEqual(args.caption_max_edge, 0)


if __name__ == "__main__":
    unittest.main()
