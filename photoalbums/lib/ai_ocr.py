from __future__ import annotations

import os
import re
import sys
from pathlib import Path

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "its",
    "our",
    "their",
    "they",
    "them",
    "into",
    "photo",
    "image",
}


class OCREngine:
    def __init__(self, *, engine: str = "docstrange", language: str = "eng"):
        self.engine = str(engine or "none").strip().lower()
        self.language = str(language or "eng").strip() or "eng"
        self._paddle = None
        self._docstrange = None

        if self.engine == "none":
            return
        if self.engine == "docstrange":
            _ensure_utf8_console_streams()
            try:
                from docstrange.config import InternalConfig  # pylint: disable=import-outside-toplevel
                from docstrange.processors.image_processor import ImageProcessor  # pylint: disable=import-outside-toplevel
            except Exception as exc:  # pragma: no cover - dependency optional
                raise RuntimeError(
                    "Docstrange local OCR engine is unavailable. Install with: pip install docstrange easyocr"
                ) from exc

            # Force local OCR provider to avoid cloud API processing paths.
            InternalConfig.ocr_provider = "neural"
            self._docstrange = ImageProcessor(
                # With docstrange neural OCR, layout mode is the stable local CPU path.
                preserve_layout=True,
                include_images=False,
                ocr_enabled=True,
            )
            return
        if self.engine == "paddle":
            from paddleocr import PaddleOCR  # pylint: disable=import-outside-toplevel
            # PaddleOCR expects "en" rather than "eng".
            paddle_lang = "en" if self.language.lower() == "eng" else self.language
            self._paddle = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
            return
        raise ValueError(f"Unsupported OCR engine: {self.engine}")

    def read_text(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if self.engine == "none":
            return ""
        if self.engine == "docstrange":
            result = self._docstrange.process(str(path))
            if result is None:
                return ""

            text = ""
            if hasattr(result, "extract_text"):
                text = str(result.extract_text() or "").strip()
            if not text and hasattr(result, "extract_markdown"):
                text = str(result.extract_markdown() or "").strip()
            return text
        if self.engine == "paddle":
            lines: list[str] = []
            result = self._paddle.ocr(str(path), cls=True) or []
            for block in result:
                for row in list(block or []):
                    if len(row) < 2:
                        continue
                    parsed = row[1]
                    if isinstance(parsed, (list, tuple)) and parsed:
                        lines.append(str(parsed[0] or "").strip())
            return "\n".join([line for line in lines if line]).strip()
        return ""


def _ensure_utf8_console_streams() -> None:
    if os.name != "nt":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        encoding = str(getattr(stream, "encoding", "") or "").lower()
        if encoding == "utf-8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Some stream wrappers do not allow reconfiguration.
            pass


def extract_keywords(text: str, *, max_keywords: int = 15) -> list[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", str(text or "")):
        value = token.strip().strip("'").lower()
        if len(value) < 3 or value in STOPWORDS:
            continue
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    return [word for word, _ in ranked[: max(1, int(max_keywords))]]
