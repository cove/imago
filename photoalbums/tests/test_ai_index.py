import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_index


class TestAIIndex(unittest.TestCase):
    def _valid_sidecar_text(self) -> str:
        return "x" * (ai_index.MIN_EXISTING_SIDECAR_BYTES + 1)

    @contextmanager
    def _mock_layout(self, image: Path):
        yield SimpleNamespace(
            page_like=False,
            split_mode="manual",
            split_applied=False,
            subphotos=[],
            content_path=image,
        )

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
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")
            stat = image.stat()
            row = {
                "image_path": str(image),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sidecar_path": str(sidecar),
                "processor_signature": ai_index.PROCESSOR_SIGNATURE,
            }
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

            next_ns = max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            os.utime(image, ns=(next_ns, next_ns))
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
                "processor_signature": ai_index.PROCESSOR_SIGNATURE,
            }
            self.assertTrue(ai_index.needs_processing(image, row, force=False))

            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")
            self.assertFalse(ai_index.needs_processing(image, row, force=False))
            self.assertTrue(ai_index.needs_processing(image, row, force=True))

    def test_needs_processing_skips_when_manifest_missing_but_valid_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")

            self.assertFalse(ai_index.needs_processing(image, None, force=False))
            self.assertTrue(ai_index.needs_processing(image, None, force=True))

    def test_needs_processing_ignores_tiny_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text("tiny", encoding="utf-8")

            self.assertTrue(ai_index.needs_processing(image, None, force=False))

    def test_run_image_analysis_passes_people_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"abc")
            people_matcher = mock.Mock()
            people_matcher.match_image.return_value = [
                SimpleNamespace(
                    name="Alice",
                    score=0.92,
                    certainty=0.92,
                    reviewed_by_human=False,
                    face_id="face-1",
                )
            ]
            object_detector = mock.Mock()
            object_detector.detect_image.return_value = []
            ocr_engine = mock.Mock()
            ocr_engine.read_text.return_value = "Dolores Cordell"
            caption_engine = mock.Mock()
            caption_engine.generate.return_value = SimpleNamespace(
                text="Caption text",
                engine="template",
                fallback=False,
                error="",
            )

            analysis = ai_index._run_image_analysis(
                image_path=image,
                people_matcher=people_matcher,
                object_detector=object_detector,
                ocr_engine=ocr_engine,
                caption_engine=caption_engine,
                requested_caption_engine="template",
                requested_caption_model="",
                ocr_engine_name="none",
                ocr_language="eng",
                people_hint_text="Page caption",
                people_source_path=Path(tmp) / "original.jpg",
                people_bbox_offset=(12, 34),
            )

            people_matcher.match_image.assert_called_once_with(
                image,
                source_path=Path(tmp) / "original.jpg",
                bbox_offset=(12, 34),
                hint_text="Page caption Dolores Cordell",
            )
            self.assertEqual(analysis.people_names, ["Alice"])
            self.assertEqual(analysis.payload["people"][0]["face_id"], "face-1")
            self.assertFalse(analysis.payload["people"][0]["reviewed_by_human"])

    def test_run_force_rewrites_existing_sidecar_and_merges_embedded_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            original = self._valid_sidecar_text()
            sidecar.write_text(original, encoding="utf-8")
            manifest = base / "manifest.jsonl"

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=lambda *args, **kwargs: self._mock_layout(image)),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "read_embedded_source_text", return_value="Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif"),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--force",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            self.assertEqual(
                write_mock.call_args.kwargs["source_text"],
                "Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif",
            )
            self.assertEqual(sidecar.read_text(encoding="utf-8"), original)

    def test_run_skips_processing_when_valid_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")
            manifest = base / "manifest.jsonl"

            with (
                mock.patch.object(ai_index, "_run_image_analysis") as analysis_mock,
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_not_called()
            write_mock.assert_not_called()

    def test_run_rewrites_sidecar_when_image_is_newer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            sidecar = image.with_suffix(".xmp")
            sidecar.write_text(self._valid_sidecar_text(), encoding="utf-8")
            manifest = base / "manifest.jsonl"

            stat = image.stat()
            ai_index.save_manifest(
                manifest,
                {
                    str(image): {
                        "image_path": str(image),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "sidecar_path": str(sidecar),
                        "processor_signature": ai_index.PROCESSOR_SIGNATURE,
                        "settings_signature": "",
                    }
                },
            )

            next_ns = max(sidecar.stat().st_mtime_ns, image.stat().st_mtime_ns) + 5_000_000
            os.utime(image, ns=(next_ns, next_ns))

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )

            with (
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=lambda *args, **kwargs: self._mock_layout(image)),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()

    def test_run_records_final_cast_store_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=[],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["hello"],
                description="Alice",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "template"},
                },
            )
            fake_matcher = mock.Mock()
            fake_matcher.store_signature.side_effect = ["sig-before", "sig-after", "sig-final"]

            with (
                mock.patch.object(ai_index, "_init_people_matcher", return_value=fake_matcher),
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=lambda *args, **kwargs: self._mock_layout(image)),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "template",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_called_once()
            rows = ai_index.load_manifest(manifest)
            self.assertEqual(rows[str(image)]["cast_store_signature"], "sig-final")

    def test_run_stdout_prints_caption_only_and_skips_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=["Alice"],
                object_labels=["dog"],
                ocr_text="hello",
                ocr_keywords=["hello"],
                subjects=["dog"],
                description="Alice with a dog",
                payload={
                    "people": [{"name": "Alice"}],
                    "objects": [{"label": "dog"}],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "qwen"},
                },
            )

            with (
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=lambda *args, **kwargs: self._mock_layout(image)),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch.object(ai_index, "write_xmp_sidecar") as write_mock,
                mock.patch.object(ai_index, "save_manifest") as save_mock,
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
                    ]
                )

            self.assertEqual(result, 0)
            write_mock.assert_not_called()
            save_mock.assert_not_called()
            print_mock.assert_called_once_with("a.jpg: Alice with a dog")

    def test_run_stdout_emits_caption_fallback_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Fallback caption text",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {
                        "requested_engine": "qwen",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "model offline",
                        "model": "Qwen/Qwen3.5-4B",
                    },
                },
            )

            with (
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=lambda *args, **kwargs: self._mock_layout(image)),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
                    ]
                )

            self.assertEqual(result, 0)
            print_mock.assert_has_calls(
                [
                    mock.call("[1/1] warn  a.jpg: caption fallback: model offline", file=sys.stderr),
                    mock.call("a.jpg: Fallback caption text"),
                ]
            )
            self.assertEqual(print_mock.call_count, 2)

    def test_run_stdout_prints_filename_only_for_empty_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {
                        "requested_engine": "lmstudio",
                        "effective_engine": "template",
                        "fallback": True,
                        "error": "model offline",
                        "model": "qwen2.5-vl-instruct",
                    },
                },
            )

            with (
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=lambda *args, **kwargs: self._mock_layout(image)),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "lmstudio",
                    ]
                )

            self.assertEqual(result, 0)
            print_mock.assert_has_calls(
                [
                    mock.call("[1/1] warn  a.jpg: caption fallback: model offline", file=sys.stderr),
                    mock.call("a.jpg"),
                ]
            )
            self.assertEqual(print_mock.call_count, 2)

    def test_run_stdout_uses_qwen_prompt_for_page_like_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "Family_1986_B02_P01.jpg"
            image.write_bytes(b"abc")
            manifest = base / "manifest.jsonl"

            content_bounds = SimpleNamespace(as_dict=lambda: {"x": 0, "y": 0, "width": 100, "height": 100})
            subphoto_bounds = SimpleNamespace(as_dict=lambda: {"x": 0, "y": 0, "width": 80, "height": 80})

            @contextmanager
            def mock_page_layout(*args, **kwargs):
                yield SimpleNamespace(
                    kind="page_view",
                    split_mode="auto",
                    content_bounds=content_bounds,
                    content_path=image,
                    original_path=image,
                    page_like=True,
                    footer_trimmed=False,
                    split_applied=False,
                    fallback_used=True,
                    subphotos=[SimpleNamespace(index=1, bounds=subphoto_bounds, path=image)],
                )

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Subphoto caption",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng", "keywords": [], "chars": 0},
                    "caption": {
                        "requested_engine": "qwen",
                        "effective_engine": "qwen",
                        "fallback": False,
                        "error": "",
                        "model": "Qwen/Qwen3.5-4B",
                    },
                },
            )
            fake_caption_engine = mock.Mock()
            fake_caption_engine.generate.return_value = SimpleNamespace(
                text="Describe this page exactly",
                engine="qwen",
                fallback=False,
                error="",
            )
            fake_ocr_engine = mock.Mock()
            fake_ocr_engine.read_text.return_value = ""

            with (
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=mock_page_layout),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis),
                mock.patch.object(ai_index, "_init_caption_engine", return_value=fake_caption_engine),
                mock.patch.object(ai_index, "OCREngine", return_value=fake_ocr_engine),
                mock.patch.object(ai_index, "extract_keywords", return_value=[]),
                mock.patch("builtins.print") as print_mock,
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
                        "--caption-prompt",
                        "describe this photo in detail",
                    ]
                )

            self.assertEqual(result, 0)
            fake_caption_engine.generate.assert_called_once_with(
                image_path=image,
                people=[],
                objects=[],
                ocr_text="",
            )
            print_mock.assert_called_once_with("Family_1986_B02_P01.jpg: Describe this page exactly")

    def test_run_stdout_forces_processing_even_when_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            photos = base / "Family_View"
            photos.mkdir()
            image = photos / "a.jpg"
            image.write_bytes(b"abc")
            image.with_suffix(".xmp").write_text(self._valid_sidecar_text(), encoding="utf-8")
            manifest = base / "manifest.jsonl"

            analysis = ai_index.ImageAnalysis(
                image_path=image,
                people_names=[],
                object_labels=[],
                ocr_text="",
                ocr_keywords=[],
                subjects=[],
                description="Caption from stdout mode",
                payload={
                    "people": [],
                    "objects": [],
                    "ocr": {"engine": "none", "language": "eng"},
                    "caption": {"engine": "qwen"},
                },
            )

            with (
                mock.patch.object(ai_index, "prepare_image_layout", side_effect=lambda *args, **kwargs: self._mock_layout(image)),
                mock.patch.object(ai_index, "_run_image_analysis", return_value=analysis) as analysis_mock,
                mock.patch.object(ai_index, "_build_flat_payload", return_value=analysis.payload),
                mock.patch("builtins.print"),
            ):
                result = ai_index.run(
                    [
                        "--photos-root",
                        str(base),
                        "--manifest",
                        str(manifest),
                        "--include-view",
                        "--stdout",
                        "--disable-people",
                        "--disable-objects",
                        "--ocr-engine",
                        "none",
                        "--caption-engine",
                        "qwen",
                    ]
                )

            self.assertEqual(result, 0)
            analysis_mock.assert_called_once()

    def test_build_description(self):
        text = ai_index.build_description(
            people=["Alice", "Bob"],
            objects=["dog", "car"],
            ocr_text="Hello world from a sign",
        )
        self.assertIn("Alice", text)
        self.assertIn("dog", text)
        self.assertIn("Visible text reads:", text)

    def test_resolve_caption_prompt_reads_file_and_overrides_inline_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text("Describe this image from file.\n", encoding="utf-8")
            text = ai_index._resolve_caption_prompt("Inline prompt", str(prompt_file))
        self.assertEqual(text, "Describe this image from file.")

    def test_resolve_caption_prompt_exits_for_missing_file(self):
        with self.assertRaises(SystemExit) as exc:
            ai_index._resolve_caption_prompt("", "/tmp/definitely-missing-caption-prompt.txt")
        self.assertIn("Caption prompt file does not exist", str(exc.exception))

    def test_parse_args_caption_flags(self):
        args = ai_index.parse_args(
            [
                "--caption-engine",
                "lmstudio",
                "--caption-model",
                "qwen2.5-vl-instruct",
                "--caption-prompt",
                "Describe this exact image",
                "--caption-prompt-file",
                "/tmp/prompt.txt",
                "--lmstudio-base-url",
                "http://localhost:1234",
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
        self.assertEqual(args.caption_engine, "lmstudio")
        self.assertEqual(args.caption_model, "qwen2.5-vl-instruct")
        self.assertEqual(args.caption_prompt, "Describe this exact image")
        self.assertEqual(args.caption_prompt_file, "/tmp/prompt.txt")
        self.assertEqual(args.lmstudio_base_url, "http://localhost:1234")
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
        self.assertEqual(args.caption_prompt, "")
        self.assertEqual(args.caption_prompt_file, "")
        self.assertEqual(args.lmstudio_base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(args.ocr_engine, "docstrange")
        self.assertFalse(args.stdout)
        self.assertEqual(args.qwen_attn_implementation, "auto")
        self.assertEqual(args.qwen_min_pixels, 0)
        self.assertEqual(args.qwen_max_pixels, 0)
        self.assertEqual(args.caption_max_edge, 0)

    def test_init_caption_engine_forwards_caption_prompt(self):
        with mock.patch.object(ai_index, "CaptionEngine") as engine_ctor:
            ai_index._init_caption_engine(
                engine="lmstudio",
                model_name="qwen2.5-vl-instruct",
                caption_prompt="Describe this exact image",
                max_tokens=64,
                temperature=0.1,
                qwen_attn_implementation="sdpa",
                qwen_min_pixels=131072,
                qwen_max_pixels=524288,
                lmstudio_base_url="http://localhost:1234",
                max_image_edge=1024,
            )

        engine_ctor.assert_called_once_with(
            engine="lmstudio",
            model_name="qwen2.5-vl-instruct",
            caption_prompt="Describe this exact image",
            max_tokens=64,
            temperature=0.1,
            qwen_attn_implementation="sdpa",
            qwen_min_pixels=131072,
            qwen_max_pixels=524288,
            lmstudio_base_url="http://localhost:1234",
            max_image_edge=1024,
            fallback_to_template=True,
        )


if __name__ == "__main__":
    unittest.main()
