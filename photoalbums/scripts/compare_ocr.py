"""Compare OCR engines on the same page images.

Prototype harness for evaluating the Docling detect-then-recognize OCR path against
the existing VLM OCR path on pages with typed caption names. Prints recognized text
per engine and, for Docling, the per-region bounding boxes and picture associations.

Usage:
    python -m photoalbums.scripts.compare_ocr IMAGE [IMAGE ...]
    python -m photoalbums.scripts.compare_ocr --engines docling,lmstudio IMAGE ...

By default only the Docling engine runs (no model server required). Add 'lmstudio'
or 'local' to --engines to compare against the VLM path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..lib._docling_ocr import run_docling_ocr
from ..lib.ai_ocr import OCREngine
from ..lib.image_limits import allow_large_pillow_images


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    allow_large_pillow_images(Image)
    with Image.open(str(path)) as image:
        return image.size


def _run_docling(path: Path) -> None:
    img_w, img_h = _image_size(path)
    result = run_docling_ocr(path, img_w, img_h)
    print(f"  [docling] {len(result.text_regions)} text region(s), {len(result.picture_boxes)} picture(s)")
    for region in result.text_regions:
        x, y, w, h = region.bbox
        print(f"    {region.label:14s} ({x},{y},{w},{h}): {region.text!r}")
    print(f"  [docling] full text:\n{result.text}\n")


def _run_vlm(engine_name: str, path: Path, *, model: str, base_url: str) -> None:
    engine = OCREngine(engine=engine_name, model_name=model, base_url=base_url)
    text = engine.read_text(path)
    print(f"  [{engine_name}] model={engine.effective_model_name}\n{text}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Page image(s) to OCR.")
    parser.add_argument(
        "--engines",
        default="docling",
        help="Comma-separated engines to run: docling, lmstudio, local. Default: docling.",
    )
    parser.add_argument("--model", default="", help="Model id/path for the VLM engines.")
    parser.add_argument("--base-url", default="", help="LM Studio base URL for the lmstudio engine.")
    args = parser.parse_args(argv)

    engines = [e.strip().lower() for e in str(args.engines).split(",") if e.strip()]
    exit_code = 0
    for image in args.images:
        path = Path(image)
        print(f"=== {path} ===")
        if not path.exists():
            print("  (missing file)\n")
            exit_code = 1
            continue
        for engine_name in engines:
            try:
                if engine_name == "docling":
                    _run_docling(path)
                elif engine_name in {"lmstudio", "local"}:
                    _run_vlm(engine_name, path, model=args.model, base_url=args.base_url)
                else:
                    print(f"  (unknown engine: {engine_name})")
            except Exception as exc:  # pragma: no cover - prototype diagnostics
                print(f"  [{engine_name}] FAILED: {exc}\n")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
