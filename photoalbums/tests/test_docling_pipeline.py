"""Tests for the Docling VLM pipeline wrapper (_docling_pipeline.py)."""

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
    """Create a mock BoundingBox with l/t/r/b attributes."""
    return SimpleNamespace(l=l, t=t, r=r, b=b)


def _make_picture_item(bbox, caption_text=None):
    """Create a mock picture item with prov and optional caption."""
    from docling_core.types.doc import DocItemLabel
    prov = SimpleNamespace(bbox=bbox)
    item = SimpleNamespace(
        label=DocItemLabel.PICTURE,
        prov=[prov],
        captions=[],
    )
    if caption_text is not None:
        ref = SimpleNamespace(cref="#/texts/0")
        item.captions = [ref]
    return item


def _make_text_item(text):
    return SimpleNamespace(text=text)


def _make_doc(items, texts=None):
    """Create a mock DoclingDocument."""
    texts = texts or []

    def iterate_items(*args, **kwargs):
        return [(item, 0) for item in items]

    return SimpleNamespace(iterate_items=iterate_items, texts=texts)


def _make_convert_result(doc):
    return SimpleNamespace(document=doc)


class TestRunDoclingPipeline(unittest.TestCase):
    """Tests for run_docling_pipeline()."""

    def _call(self, items, texts=None, img_w=1000, img_h=800, source_size=(200, 100)):
        """Helper: mock DocumentConverter.convert() and call run_docling_pipeline."""
        doc = _make_doc(items, texts=texts)
        convert_result = _make_convert_result(doc)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.jpg"
            Image.new("RGB", source_size, color="white").save(image_path, format="JPEG", quality=95)

            with (
                mock.patch("docling.document_converter.DocumentConverter.convert", return_value=convert_result),
                mock.patch("photoalbums.lib._docling_pipeline._get_region_result") as mock_rr,
            ):
                from photoalbums.lib.ai_view_regions import RegionResult
                mock_rr.return_value = RegionResult

                from photoalbums.lib._docling_pipeline import run_docling_pipeline
                return run_docling_pipeline(
                    image_path=image_path,
                    img_w=img_w,
                    img_h=img_h,
                    preset="granite_docling",
                )

    def test_two_picture_items_returns_two_regions(self):
        """Two picture items produce two RegionResult objects with correct pixel coords."""
        items = [
            _make_picture_item(_make_bbox(l=0.1, t=0.2, r=0.5, b=0.6)),
            _make_picture_item(_make_bbox(l=0.6, t=0.1, r=0.9, b=0.5)),
        ]
        regions = self._call(items, img_w=1000, img_h=800)

        self.assertEqual(len(regions), 2)

        r0 = regions[0]
        self.assertEqual(r0.x, 100)   # round(0.1 * 1000)
        self.assertEqual(r0.y, 160)   # round(0.2 * 800)
        self.assertEqual(r0.width, 400)   # round((0.5-0.1)*1000)
        self.assertEqual(r0.height, 320)  # round((0.6-0.2)*800)

        r1 = regions[1]
        self.assertEqual(r1.x, 600)
        self.assertEqual(r1.y, 80)
        self.assertEqual(r1.width, 300)
        self.assertEqual(r1.height, 320)

    def test_picture_with_caption(self):
        """A picture item with a caption ref sets caption_hint."""
        items = [_make_picture_item(_make_bbox(0.0, 0.0, 0.5, 0.5), caption_text="A nice photo")]
        texts = [_make_text_item("A nice photo")]

        regions = self._call(items, texts=texts)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].caption_hint, "A nice photo")

    def test_picture_without_caption(self):
        """A picture item with no caption ref has empty caption_hint."""
        items = [_make_picture_item(_make_bbox(0.0, 0.0, 0.5, 0.5))]
        regions = self._call(items)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].caption_hint, "")

    def test_empty_document_returns_empty_list_with_warning(self):
        """No picture items → empty list and WARNING logged."""
        with self.assertLogs("photoalbums.lib._docling_pipeline", level=logging.WARNING):
            regions = self._call(items=[])
        self.assertEqual(regions, [])

    def test_item_without_prov_is_skipped(self):
        """Picture items with no prov are skipped."""
        from docling_core.types.doc import DocItemLabel
        no_prov = SimpleNamespace(label=DocItemLabel.PICTURE, prov=[], captions=[])
        valid = _make_picture_item(_make_bbox(0.0, 0.0, 0.5, 0.5))
        with self.assertLogs("photoalbums.lib._docling_pipeline", level=logging.WARNING):
            regions = self._call(items=[no_prov])
        self.assertEqual(regions, [])

    def test_non_picture_items_ignored(self):
        """Non-PICTURE items are not converted to RegionResult."""
        from docling_core.types.doc import DocItemLabel
        text_item = SimpleNamespace(label=DocItemLabel.TEXT, prov=[], captions=[])
        picture = _make_picture_item(_make_bbox(0.0, 0.0, 1.0, 1.0))
        regions = self._call(items=[text_item, picture])
        self.assertEqual(len(regions), 1)

    def test_region_indices_are_sequential(self):
        """RegionResult indices start at 0 and increment."""
        items = [
            _make_picture_item(_make_bbox(0.0, 0.0, 0.3, 0.3)),
            _make_picture_item(_make_bbox(0.4, 0.4, 0.7, 0.7)),
            _make_picture_item(_make_bbox(0.8, 0.0, 1.0, 0.5)),
        ]
        regions = self._call(items)
        self.assertEqual([r.index for r in regions], [0, 1, 2])

    def test_original_image_is_passed_to_convert(self):
        """Docling converts the original image path without resizing it first."""
        doc = _make_doc([_make_picture_item(_make_bbox(0.0, 0.0, 1.0, 1.0))])
        convert_result = _make_convert_result(doc)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "source.jpg"
            Image.new("RGB", (400, 200), color="white").save(image_path, format="JPEG", quality=95)

            def convert_side_effect(*args, **kwargs):
                source = next(Path(arg) for arg in args if isinstance(arg, (str, Path)))
                with Image.open(source) as converted:
                    self.assertEqual(converted.size, (400, 200))
                return convert_result

            with (
                mock.patch("docling.document_converter.DocumentConverter.convert", side_effect=convert_side_effect),
                mock.patch("photoalbums.lib._docling_pipeline._get_region_result") as mock_rr,
            ):
                from photoalbums.lib.ai_view_regions import RegionResult
                mock_rr.return_value = RegionResult

                from photoalbums.lib._docling_pipeline import run_docling_pipeline
                run_docling_pipeline(
                    image_path=image_path,
                    img_w=1000,
                    img_h=800,
                    preset="granite_docling",
                )

    def test_uses_preset_without_lmstudio_engine_options(self):
        """Docling preset resolution stays local and does not build LM Studio engine options."""
        doc = _make_doc([_make_picture_item(_make_bbox(0.0, 0.0, 1.0, 1.0))])
        convert_result = _make_convert_result(doc)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "source.jpg"
            Image.new("RGB", (200, 100), color="white").save(image_path, format="JPEG", quality=95)

            from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions

            with (
                mock.patch("docling.datamodel.pipeline_options.VlmConvertOptions.from_preset", wraps=VlmConvertOptions.from_preset) as mock_from_preset,
                mock.patch("docling.datamodel.pipeline_options.VlmPipelineOptions", wraps=VlmPipelineOptions) as mock_pipeline_options,
                mock.patch("docling.datamodel.vlm_engine_options.ApiVlmEngineOptions") as mock_engine,
                mock.patch("docling.document_converter.DocumentConverter.convert", return_value=convert_result),
                mock.patch("photoalbums.lib._docling_pipeline._get_region_result") as mock_rr,
            ):
                from photoalbums.lib.ai_view_regions import RegionResult
                mock_rr.return_value = RegionResult

                from photoalbums.lib._docling_pipeline import run_docling_pipeline
                run_docling_pipeline(
                    image_path=image_path,
                    img_w=1000,
                    img_h=800,
                    preset="granite_docling",
                )

            self.assertEqual(mock_from_preset.call_args.args[0], "granite_docling")
            self.assertEqual(mock_from_preset.call_args.kwargs, {})
            self.assertTrue(mock_pipeline_options.called)
            mock_engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
