import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_ocr


class _FakeResult:
    def __init__(self, *, text: str = "", markdown: str = ""):
        self._text = text
        self._markdown = markdown

    def extract_text(self) -> str:
        return self._text

    def extract_markdown(self) -> str:
        return self._markdown


class _FakeImageProcessor:
    def __init__(self, *, preserve_layout: bool, include_images: bool, ocr_enabled: bool):
        self.args = (preserve_layout, include_images, ocr_enabled)

    def process(self, _path: str):
        return _FakeResult(text="local text")


class _FakeImageProcessorMarkdown:
    def __init__(self, *, preserve_layout: bool, include_images: bool, ocr_enabled: bool):
        self.args = (preserve_layout, include_images, ocr_enabled)

    def process(self, _path: str):
        return _FakeResult(markdown="local markdown")


def _docstrange_module_patch(image_processor_cls):
    mod_docstrange = types.ModuleType("docstrange")
    mod_config = types.ModuleType("docstrange.config")

    class InternalConfig:
        ocr_provider = "nanonets"

    mod_config.InternalConfig = InternalConfig
    mod_processors = types.ModuleType("docstrange.processors")
    mod_image = types.ModuleType("docstrange.processors.image_processor")
    mod_image.ImageProcessor = image_processor_cls
    return patch.dict(
        sys.modules,
        {
            "docstrange": mod_docstrange,
            "docstrange.config": mod_config,
            "docstrange.processors": mod_processors,
            "docstrange.processors.image_processor": mod_image,
        },
        clear=False,
    ), InternalConfig


class TestAIOcr(unittest.TestCase):
    def test_docstrange_uses_local_image_processor(self):
        module_patch, internal_config_cls = _docstrange_module_patch(_FakeImageProcessor)
        with module_patch:
            ocr = ai_ocr.OCREngine(engine="docstrange")
            self.assertEqual(internal_config_cls.ocr_provider, "neural")
            self.assertEqual(ocr._docstrange.args, (True, False, True))
            self.assertEqual(ocr.read_text("sample.jpg"), "local text")

    def test_docstrange_falls_back_to_markdown_text(self):
        module_patch, _ = _docstrange_module_patch(_FakeImageProcessorMarkdown)
        with module_patch:
            ocr = ai_ocr.OCREngine(engine="docstrange")
            self.assertEqual(ocr.read_text("sample.jpg"), "local markdown")

    def test_docstrange_missing_local_modules_raises_runtime_error(self):
        fake_docstrange = types.ModuleType("docstrange")
        with patch.dict(sys.modules, {"docstrange": fake_docstrange}, clear=False):
            with self.assertRaises(RuntimeError):
                ai_ocr.OCREngine(engine="docstrange")


if __name__ == "__main__":
    unittest.main()
