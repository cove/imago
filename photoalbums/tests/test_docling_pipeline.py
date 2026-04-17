"""Tests for the local Docling pipeline wrapper."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


def _make_bbox(l, t, r, b):
    from docling_core.types.doc import BoundingBox, CoordOrigin

    return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT)


def _make_picture_item(bbox, caption_text=None):
    from docling_core.types.doc import DocItemLabel

    item = SimpleNamespace(label=DocItemLabel.PICTURE, prov=[SimpleNamespace(bbox=bbox, page_no=1)], captions=[])
    if caption_text is not None:
        item.captions = [SimpleNamespace(cref="#/texts/0")]
    return item


def _make_doc(items, texts=None):
    texts = texts or []

    def iterate_items(*args, **kwargs):
        return [(item, 0) for item in items]

    export_to_dict = lambda: {"items": len(items), "texts": [str(getattr(text, "text", "") or "") for text in texts]}
    return SimpleNamespace(
        iterate_items=iterate_items,
        texts=texts,
        pages={1: SimpleNamespace(size=SimpleNamespace(height=800.0))},
        export_to_dict=export_to_dict,
    )


def _make_convert_result(doc):
    return SimpleNamespace(
        document=doc,
        status="success",
        errors=[],
        pages=[],
        timings={},
        confidence={},
        model_dump=lambda mode="json": {"status": "success", "errors": [], "pages": [], "timings": {}, "confidence": {}},
    )


class TestRunDoclingPipeline(unittest.TestCase):
    def _call(self, items, texts=None, img_w=1000, img_h=800):
        doc = _make_doc(items, texts=texts)
        convert_result = _make_convert_result(doc)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.jpg"
            Image.new("RGB", (200, 100), color="white").save(image_path, format="JPEG", quality=95)
            with (
                mock.patch("docling.document_converter.DocumentConverter.convert", return_value=convert_result),
                mock.patch("photoalbums.lib._docling_pipeline._get_region_result") as mock_rr,
            ):
                from photoalbums.lib.ai_view_regions import RegionResult
                from photoalbums.lib._docling_pipeline import run_docling_pipeline

                mock_rr.return_value = RegionResult
                return run_docling_pipeline(
                    image_path=image_path,
                    img_w=img_w,
                    img_h=img_h,
                    preset="granite_docling",
                    backend="auto_inline",
                    device="auto",
                    retries=3,
                )

    def test_picture_items_map_to_region_results(self):
        result = self._call(
            [
                _make_picture_item(_make_bbox(100, 160, 500, 480)),
                _make_picture_item(_make_bbox(600, 80, 900, 400)),
            ]
        )

        self.assertEqual(len(result.regions), 2)
        self.assertEqual(result.regions[0].x, 100)
        self.assertEqual(result.regions[0].y, 160)
        self.assertEqual(result.regions[0].width, 400)
        self.assertEqual(result.regions[0].height, 320)
        self.assertEqual(result.regions[1].x, 600)

    def test_picture_caption_populates_caption_hint_and_debug_payload(self):
        result = self._call(
            [_make_picture_item(_make_bbox(0.0, 0.0, 0.5, 0.5), caption_text="A nice photo")],
            texts=[SimpleNamespace(text="A nice photo")],
        )

        self.assertEqual(result.regions[0].caption_hint, "A nice photo")
        self.assertEqual(result.debug_payload["config"]["preset"], "granite_docling")
        self.assertEqual(result.debug_payload["config"]["backend"], "auto_inline")
        self.assertEqual(result.debug_payload["config"]["device"], "auto")
        self.assertEqual(result.debug_payload["region_count"], 1)
        self.assertIn("document", result.debug_payload)

    def test_empty_document_logs_warning_and_returns_empty_regions(self):
        with self.assertLogs("photoalbums.lib._docling_pipeline", level=logging.WARNING):
            result = self._call([])
        self.assertEqual(result.regions, [])

    def test_uses_standard_image_pipeline_with_auto_device(self):
        doc = _make_doc([_make_picture_item(_make_bbox(0.0, 0.0, 1.0, 1.0))])
        convert_result = _make_convert_result(doc)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "source.jpg"
            Image.new("RGB", (200, 100), color="white").save(image_path, format="JPEG", quality=95)
            with (
                mock.patch("docling.document_converter.DocumentConverter.__init__", return_value=None) as mock_converter_init,
                mock.patch("docling.document_converter.DocumentConverter.convert", return_value=convert_result),
                mock.patch("photoalbums.lib._docling_pipeline._get_region_result") as mock_rr,
            ):
                from photoalbums.lib.ai_view_regions import RegionResult
                from photoalbums.lib._docling_pipeline import run_docling_pipeline

                mock_rr.return_value = RegionResult
                result = run_docling_pipeline(
                    image_path=image_path,
                    img_w=1000,
                    img_h=800,
                    preset="granite_docling",
                    backend="auto_inline",
                    device="auto",
                    retries=3,
                )

        format_options = mock_converter_init.call_args.kwargs["format_options"]
        image_option = format_options[next(iter(format_options))]
        self.assertEqual(image_option.backend.__name__, "ImageDocumentBackend")
        self.assertEqual(result.debug_payload["config"]["pipeline_options"]["accelerator_options"]["device"], "auto")
        self.assertEqual(result.debug_payload["config"]["pipeline_kind"], "standard_image")

    def test_runtime_failures_retry_then_raise_with_debug_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "source.jpg"
            Image.new("RGB", (200, 100), color="white").save(image_path, format="JPEG", quality=95)
            with (
                mock.patch("docling.document_converter.DocumentConverter.convert", side_effect=RuntimeError("backend boom")) as mock_convert,
                mock.patch("photoalbums.lib._docling_pipeline._get_region_result") as mock_rr,
            ):
                from photoalbums.lib.ai_view_regions import RegionResult
                from photoalbums.lib._docling_pipeline import DoclingPipelineRuntimeError, run_docling_pipeline

                mock_rr.return_value = RegionResult
                with self.assertRaises(DoclingPipelineRuntimeError) as exc:
                    run_docling_pipeline(
                        image_path=image_path,
                        img_w=1000,
                        img_h=800,
                        preset="granite_docling",
                        backend="auto_inline",
                        device="auto",
                        retries=2,
                    )

        self.assertEqual(mock_convert.call_count, 2)
        self.assertIn("backend boom", str(exc.exception))
        self.assertEqual(len(exc.exception.debug_payload["runtime_errors"]), 2)
        self.assertEqual(exc.exception.debug_payload["config"]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
