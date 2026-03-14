from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..naming import parse_album_filename
from .model_store import HF_MODEL_CACHE_DIR


DEFAULT_QWEN_CAPTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LEGACY_QWEN_CAPTION_MODEL_ALIASES = {
    "qwen/qwen3.5-4b": DEFAULT_QWEN_CAPTION_MODEL,
}
DEFAULT_QWEN_AUTO_MAX_IMAGE_EDGE = 1280
DEFAULT_QWEN_AUTO_MAX_PIXELS = 786_432
DEFAULT_LMSTUDIO_MAX_NEW_TOKENS = 256
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LMSTUDIO_TIMEOUT_SECONDS = 180.0
DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE = 2048
LMSTUDIO_VISION_MODEL_HINTS = (
    "vl",
    "vision",
    "llava",
    "minicpm",
    "moondream",
    "pixtral",
    "internvl",
    "phi-3.5-vision",
    "phi-4-multimodal",
    "qvq",
)
QWEN_ATTN_IMPLEMENTATIONS = {"auto", "sdpa", "flash_attention_2", "eager"}
ALBUM_KIND_FAMILY = "family_photo_album"
ALBUM_KIND_PHOTO_ESSAY = "photo_essay"
_ALBUM_REGION_HINTS = (
    ("eastern europe", "Eastern Europe"),
    ("south america", "South America"),
    ("panama canal", "Panama Canal"),
    ("china", "China"),
    ("egypt", "Egypt"),
    ("england", "England"),
    ("europe", "Europe"),
    ("italy", "Italy"),
    ("morocco", "Morocco"),
    ("mexico", "Mexico"),
    ("orient", "Orient"),
    ("panama", "Panama"),
    ("portugal", "Portugal"),
    ("russia", "Russia"),
    ("spain", "Spain"),
)


@dataclass(frozen=True)
class AlbumContext:
    kind: str = ""
    label: str = ""
    focus: str = ""


def clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def join_human(values: list[str]) -> str:
    clean = [str(item or "").strip() for item in values if str(item or "").strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _split_camel_case(value: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value or ""))


def _humanize_hint_text(value: str) -> str:
    return clean_text(_split_camel_case(value).replace("_", " ").replace("-", " "))


def _normalized_hint_text(value: str) -> str:
    return _humanize_hint_text(value).casefold()


def _extract_collection_hint(image_path: str | Path | None) -> str:
    if image_path is None:
        return ""
    path = Path(image_path)
    if path.name:
        collection, _year, _book, _page = parse_album_filename(path.name)
        if collection != "Unknown":
            return _humanize_hint_text(collection)
    for candidate in (path.parent.name, path.parent.parent.name if path.parent != path else ""):
        text = str(candidate or "").strip()
        if not text:
            continue
        if text.lower().startswith("imago-page-"):
            continue
        for suffix in ("_Archive", "_View"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break
        collection, _year, _book, _page = parse_album_filename(text)
        if collection != "Unknown":
            return _humanize_hint_text(collection)
        match = re.search(r"(?P<collection>.+?)_\d{4}(?:-\d{4})?_B", text, flags=re.IGNORECASE)
        if match:
            return _humanize_hint_text(match.group("collection"))
        if text.casefold() not in {"photo albums", "photoalbums"}:
            return _humanize_hint_text(text)
    return ""


def _find_region_hints(*values: str) -> list[str]:
    haystack = " ".join(_normalized_hint_text(value) for value in values if str(value or "").strip())
    if not haystack:
        return []
    matches: list[str] = []
    seen: set[str] = set()
    for needle, label in _ALBUM_REGION_HINTS:
        pattern = rf"(?<![a-z]){re.escape(needle)}(?![a-z])"
        if not re.search(pattern, haystack):
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        matches.append(label)
    return matches


def infer_album_context(
    *,
    image_path: str | Path | None = None,
    ocr_text: str = "",
    allow_ocr: bool = True,
) -> AlbumContext:
    collection_hint = _extract_collection_hint(image_path)
    path_hint = _humanize_hint_text(str(image_path or ""))
    signals = [collection_hint, path_hint]
    if allow_ocr:
        signals.append(ocr_text)
    normalized = " ".join(_normalized_hint_text(value) for value in signals if str(value or "").strip())
    if not normalized:
        return AlbumContext()
    if re.search(r"(?<![a-z])family(?![a-z])", normalized):
        return AlbumContext(
            kind=ALBUM_KIND_FAMILY,
            label="Family Photo Album",
            focus="Family",
        )
    region_hints = _find_region_hints(*signals)
    if region_hints:
        return AlbumContext(
            kind=ALBUM_KIND_PHOTO_ESSAY,
            label="Photo Essay",
            focus=join_human(region_hints),
        )
    return AlbumContext()


def _should_apply_album_prompt_rules(source_path: str | Path | None, album_context: AlbumContext) -> bool:
    if album_context.kind:
        return True
    if source_path is None:
        return False
    joined = " ".join(str(part or "").casefold() for part in Path(source_path).parts)
    return "photo albums" in joined or "cordell" in joined


def _looks_like_uniform_cover_color(image_path: str | Path) -> bool:
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel
    except Exception:
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return False

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > 512:
        scale = 512.0 / float(longest)
        resized = cv2.resize(
            image,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    pixels = resized.reshape(-1, 3).astype("float32")
    if pixels.size == 0:
        return False

    mean = pixels.mean(axis=0)
    delta = np.abs(pixels - mean)
    uniform_ratio = float((delta.max(axis=1) <= 45.0).mean())
    blue_dominant = bool(mean[0] >= 95.0 and mean[0] >= mean[1] + 18.0 and mean[0] >= mean[2] + 18.0)
    white_dominant = bool(float(mean.min()) >= 170.0)
    return uniform_ratio >= 0.72 and (blue_dominant or white_dominant)


def looks_like_album_cover(
    image_path: str | Path,
    *,
    ocr_text: str,
    album_context: AlbumContext | None = None,
) -> bool:
    text = clean_text(ocr_text)
    if not text:
        return False
    context = album_context or infer_album_context(image_path=image_path, ocr_text=text, allow_ocr=True)
    if not context.kind:
        return False
    return _looks_like_uniform_cover_color(image_path)


def build_template_caption(*, people: list[str], objects: list[str], ocr_text: str) -> str:
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)

    parts: list[str] = []
    if people_list and object_list:
        parts.append(
            f"This photo appears to show {join_human(people_list)} with {join_human(object_list)} in view."
        )
    elif people_list:
        parts.append(f"This photo appears to show {join_human(people_list)}.")
    elif object_list:
        parts.append(f"This photo includes {join_human(object_list)}.")

    if text:
        snippet = text[:180].strip()
        if len(text) > len(snippet):
            snippet += "..."
        parts.append(f'Visible text reads: "{snippet}".')
    return " ".join(parts).strip()


def build_page_caption(
    *,
    photo_count: int,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    album_context: AlbumContext | None = None,
) -> str:
    count = max(1, int(photo_count))
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    context = album_context or AlbumContext()

    if context.kind == ALBUM_KIND_FAMILY:
        parts = [f"This Family Photo Album page contains {count} photo(s)."]
    elif context.kind == ALBUM_KIND_PHOTO_ESSAY:
        parts = [f"This Photo Essay page contains {count} photo(s)."]
    else:
        parts = [f"This album page contains {count} photo(s)."]
    if context.kind == ALBUM_KIND_PHOTO_ESSAY and context.focus:
        parts.append(f"The album title suggests {context.focus}.")
    if people_list and object_list:
        parts.append(
            f"Across the page, it appears to show {join_human(people_list)} with {join_human(object_list)} in view."
        )
    elif people_list:
        parts.append(f"Across the page, it appears to show {join_human(people_list)}.")
    elif object_list:
        parts.append(f"Across the page, visible objects include {join_human(object_list)}.")

    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        parts.append(f'Visible text on the page reads: "{snippet}".')
    return " ".join(parts).strip()


def build_cover_page_caption(*, ocr_text: str, album_context: AlbumContext | None = None) -> str:
    text = clean_text(ocr_text)
    context = album_context or AlbumContext()
    if context.kind == ALBUM_KIND_FAMILY:
        parts = ["This appears to be the cover or title page of a Family Photo Album."]
    elif context.kind == ALBUM_KIND_PHOTO_ESSAY:
        parts = ["This appears to be the cover or title page of a Photo Essay."]
        if context.focus:
            parts.append(f"The title suggests {context.focus}.")
    else:
        parts = ["This appears to be the cover or title page of the album book."]
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        parts.append(f'Visible title text reads: "{snippet}".')
    return " ".join(parts).strip()


def _build_qwen_prompt(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | Path | None = None,
) -> str:
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    context = infer_album_context(image_path=source_path, ocr_text=ocr_text, allow_ocr=True)
    lines = [
        "Describe this photo in detail",
    ]
    if source_path is not None:
        path = Path(source_path)
        lines.append(f"Filename hint: {path.name}.")
        if path.parent.name:
            lines.append(f"Folder hint: {path.parent.name}.")
    if _should_apply_album_prompt_rules(source_path, context):
        lines.append("Cordell Photo Albums rules:")
        lines.append("- If the folder or file name contains Family, describe it as a Family Photo Album.")
        lines.append("- If the folder or file name contains a country or region name, describe it as a Photo Essay.")
        lines.append(
            "- If the image is mostly a solid blue or white cover with title text naming a country, region, or family, describe it as the cover of the photo album book."
        )
        if context.label:
            lines.append(f"Album classification hint: {context.label}.")
        if context.focus and context.kind == ALBUM_KIND_PHOTO_ESSAY:
            lines.append(f"Album focus hint: {context.focus}.")
    if people_list:
        lines.append(f"Known people matches: {join_human(people_list)}.")
    if object_list:
        lines.append(f"Detected objects: {join_human(object_list)}.")
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        lines.append(f'OCR text hint: "{snippet}".')
    lines.append("Return plain text only.")
    return "\n".join(lines)


def _normalize_caption(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("assistant:"):
        text = text.split(":", 1)[1].strip()
    text = re.sub(
        r"^\s*the user wants\b.*?(?:\n\s*\n|(?=\*\*1\.)|(?=1\.)|(?=\-\s)|(?=\*\s)|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = clean_text(text.replace("<think>", " ").replace("</think>", " "))
    return text


def resolve_caption_model(engine: str, model_name: str) -> str:
    normalized = str(engine or "").strip().lower()
    if normalized == "blip":
        normalized = "qwen"
    text = str(model_name or "").strip()
    if text and normalized == "qwen":
        alias = LEGACY_QWEN_CAPTION_MODEL_ALIASES.get(text.casefold())
        if alias:
            return alias
    if text:
        return text
    if normalized == "qwen":
        return DEFAULT_QWEN_CAPTION_MODEL
    return ""


def normalize_qwen_attn_implementation(value: str, default: str = "auto") -> str:
    text = str(value or "").strip().lower()
    if text in QWEN_ATTN_IMPLEMENTATIONS:
        return text
    fallback = str(default or "auto").strip().lower()
    if fallback in QWEN_ATTN_IMPLEMENTATIONS:
        return fallback
    return "auto"


def _resize_caption_image(image, max_image_edge: int):
    if int(max_image_edge) <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= int(max_image_edge):
        return image
    scale = float(max_image_edge) / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resampling = getattr(getattr(image, "Resampling", None), "LANCZOS", None)
    if resampling is None:
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel

            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        except Exception:  # pragma: no cover - Pillow always present in runtime
            resampling = 1
    return image.resize(new_size, resampling)


def normalize_lmstudio_base_url(value: str, default: str = DEFAULT_LMSTUDIO_BASE_URL) -> str:
    text = str(value or "").strip() or str(default or DEFAULT_LMSTUDIO_BASE_URL)
    text = text.rstrip("/")
    if text.endswith("/v1"):
        return text
    return f"{text}/v1"


def _resolve_local_hf_snapshot(model_name: str) -> Path | None:
    text = str(model_name or "").strip()
    if "/" not in text:
        return None
    repo_dir = HF_MODEL_CACHE_DIR / f"models--{text.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        return None
    for snapshot in sorted(repo_dir.iterdir()):
        if not snapshot.is_dir():
            continue
        if (snapshot / "config.json").exists() and (
            (snapshot / "preprocessor_config.json").exists() or (snapshot / "processor_config.json").exists()
        ):
            return snapshot
    return None


def _load_qwen_transformers():
    try:
        import torch  # pylint: disable=import-outside-toplevel
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except Exception as exc:
        raise RuntimeError(
            "Qwen captioning requires a compatible transformers/torch install."
        ) from exc

    try:
        from transformers import AutoModelForVision2Seq  # pylint: disable=import-outside-toplevel
    except Exception:
        AutoModelForVision2Seq = None

    return torch, AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText


def _build_data_url(image_path: str | Path, max_image_edge: int) -> str:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    path = Path(image_path)
    image = Image.open(str(path)).convert("RGB")
    try:
        working_image = _resize_caption_image(image, max_image_edge)
        buffer = io.BytesIO()
        if path.suffix.lower() == ".png":
            mime = "image/png"
            working_image.save(buffer, format="PNG")
        else:
            mime = "image/jpeg"
            working_image.save(buffer, format="JPEG", quality=95)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{mime};base64,{data}"
    finally:
        if "working_image" in locals() and working_image is not image:
            working_image.close()
        image.close()


def _decode_lmstudio_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _lmstudio_request_json(url: str, *, payload: dict | None = None, timeout: float) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        message = details or f"HTTP {exc.code}"
        raise RuntimeError(f"LM Studio request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _select_lmstudio_model(base_url: str, requested_model: str, timeout: float) -> str:
    text = str(requested_model or "").strip()
    if text:
        return text
    payload = _lmstudio_request_json(f"{base_url}/models", timeout=timeout)
    model_ids = [
        str(row.get("id") or "").strip()
        for row in list(payload.get("data") or [])
        if str(row.get("id") or "").strip()
    ]
    if not model_ids:
        raise RuntimeError("LM Studio did not return any models. Load a model or pass --caption-model.")
    for model_id in model_ids:
        lowered = model_id.casefold()
        if any(hint in lowered for hint in LMSTUDIO_VISION_MODEL_HINTS):
            return model_id
    return model_ids[0]


class LMStudioCaptioner:
    def __init__(
        self,
        *,
        model_name: str = "",
        prompt_text: str = "",
        max_new_tokens: int = DEFAULT_LMSTUDIO_MAX_NEW_TOKENS,
        temperature: float = 0.2,
        base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        timeout_seconds: float = DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
        max_image_edge: int = 0,
    ):
        self.model_name = str(model_name or "").strip()
        self.prompt_text = str(prompt_text or "").strip()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.base_url = normalize_lmstudio_base_url(base_url)
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.max_image_edge = max(0, int(max_image_edge))
        self._resolved_model_name = ""

    def _resolve_model_name(self) -> str:
        if self._resolved_model_name:
            return self._resolved_model_name
        self._resolved_model_name = _select_lmstudio_model(
            self.base_url,
            self.model_name,
            self.timeout_seconds,
        )
        return self._resolved_model_name

    def describe(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
    ) -> str:
        prompt = self.prompt_text or _build_qwen_prompt(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
        )
        resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
        image_url = _build_data_url(image_path, resize_edge)
        payload = {
            "model": self._resolve_model_name(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "stream": False,
        }
        response = _lmstudio_request_json(
            f"{self.base_url}/chat/completions",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            return ""
        message = dict(choices[0].get("message") or {})
        return _normalize_caption(_decode_lmstudio_text(message.get("content")))


class QwenLocalCaptioner:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_CAPTION_MODEL,
        prompt_text: str = "",
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        attn_implementation: str = "auto",
        min_pixels: int = 0,
        max_pixels: int = 0,
        max_image_edge: int = 0,
    ):
        self.model_name = str(model_name or DEFAULT_QWEN_CAPTION_MODEL).strip() or DEFAULT_QWEN_CAPTION_MODEL
        self.prompt_text = str(prompt_text or "").strip()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.attn_implementation = normalize_qwen_attn_implementation(attn_implementation)
        self.min_pixels = max(0, int(min_pixels))
        self.max_pixels = max(0, int(max_pixels))
        self.max_image_edge = max(0, int(max_image_edge))
        self._processor = None
        self._model = None
        self._torch = None
        self._resolved_attn_implementation = "auto"

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        torch, AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText = _load_qwen_transformers()

        HF_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_MODEL_CACHE_DIR)
        local_snapshot = _resolve_local_hf_snapshot(self.model_name)
        model_ref = str(local_snapshot) if local_snapshot is not None else self.model_name
        local_files_only = local_snapshot is not None
        processor_kwargs = {
            "trust_remote_code": True,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if self.min_pixels > 0:
            processor_kwargs["min_pixels"] = int(self.min_pixels)
        processor_kwargs["max_pixels"] = (
            int(self.max_pixels) if self.max_pixels > 0 else int(DEFAULT_QWEN_AUTO_MAX_PIXELS)
        )
        self._processor = AutoProcessor.from_pretrained(
            model_ref,
            **processor_kwargs,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        resolved_attn = "auto"
        if self.attn_implementation != "auto":
            if self.attn_implementation == "flash_attention_2" and not torch.cuda.is_available():
                resolved_attn = "auto"
            else:
                resolved_attn = self.attn_implementation
                load_kwargs["attn_implementation"] = resolved_attn
        self._resolved_attn_implementation = resolved_attn
        # Prefer dtype over torch_dtype to avoid deprecation warnings on newer transformers.
        vision_exc = None
        if AutoModelForVision2Seq is not None:
            try:
                self._model = AutoModelForVision2Seq.from_pretrained(
                    model_ref,
                    dtype="auto",
                    **load_kwargs,
                )
            except TypeError:
                self._model = AutoModelForVision2Seq.from_pretrained(
                    model_ref,
                    torch_dtype="auto",
                    **load_kwargs,
                )
            except Exception as exc:
                vision_exc = exc
        if self._model is None:
            # Fallback for environments where Vision2Seq auto-mapping is unavailable.
            try:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_ref,
                    dtype="auto",
                    **load_kwargs,
                )
            except TypeError:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_ref,
                    torch_dtype="auto",
                    **load_kwargs,
                )
            except Exception as exc:
                if vision_exc is not None:
                    raise RuntimeError(
                        "Qwen model loading failed for both AutoModelForVision2Seq and AutoModelForImageTextToText."
                    ) from exc
                raise
        self._torch = torch

    def describe(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
    ) -> str:
        self._ensure_loaded()
        from PIL import Image  # pylint: disable=import-outside-toplevel

        prompt = self.prompt_text or _build_qwen_prompt(
            people=people,
            objects=objects,
            ocr_text=ocr_text,
            source_path=source_path or image_path,
        )
        image = Image.open(str(image_path)).convert("RGB")
        try:
            resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_QWEN_AUTO_MAX_IMAGE_EDGE)
            working_image = _resize_caption_image(image, resize_edge)
            if hasattr(self._processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                try:
                    prompt_text = self._processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    prompt_text = self._processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                prompt_text = prompt

            inputs = self._processor(
                text=[prompt_text],
                images=[working_image],
                padding=True,
                return_tensors="pt",
            )

            device = getattr(self._model, "device", None)
            if device is not None:
                for key, value in list(inputs.items()):
                    if hasattr(value, "to"):
                        inputs[key] = value.to(device)

            do_sample = self.temperature > 0
            kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                kwargs["temperature"] = self.temperature
                kwargs["top_p"] = 0.9

            with self._torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **kwargs)

            input_ids = inputs.get("input_ids")
            if hasattr(generated_ids, "shape") and input_ids is not None and hasattr(input_ids, "shape"):
                prompt_tokens = int(input_ids.shape[-1])
                generated_ids = generated_ids[:, prompt_tokens:]

            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return _normalize_caption(decoded[0] if decoded else "")
        finally:
            if "working_image" in locals() and working_image is not image:
                working_image.close()
            image.close()


@dataclass
class CaptionOutput:
    text: str
    engine: str
    fallback: bool = False
    error: str = ""


class CaptionEngine:
    def __init__(
        self,
        *,
        engine: str = "qwen",
        model_name: str = "",
        caption_prompt: str = "",
        max_tokens: int = 96,
        temperature: float = 0.2,
        qwen_attn_implementation: str = "auto",
        qwen_min_pixels: int = 0,
        qwen_max_pixels: int = 0,
        lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        max_image_edge: int = 0,
        fallback_to_template: bool = True,
    ):
        normalized = str(engine or "qwen").strip().lower()
        if normalized == "blip":
            normalized = "qwen"
        if normalized not in {"none", "template", "qwen", "lmstudio"}:
            raise ValueError(f"Unsupported caption engine: {engine}")
        self.engine = normalized
        self.fallback_to_template = bool(fallback_to_template)
        self._captioner = None
        self._model_name = resolve_caption_model(normalized, model_name)
        self._caption_prompt = str(caption_prompt or "").strip()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._qwen_attn_implementation = normalize_qwen_attn_implementation(qwen_attn_implementation)
        self._qwen_min_pixels = max(0, int(qwen_min_pixels))
        self._qwen_max_pixels = max(0, int(qwen_max_pixels))
        self._lmstudio_base_url = normalize_lmstudio_base_url(lmstudio_base_url)
        self._max_image_edge = max(0, int(max_image_edge))

    def generate(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
        source_path: str | Path | None = None,
    ) -> CaptionOutput:
        template = build_template_caption(people=people, objects=objects, ocr_text=ocr_text)
        if self.engine == "none":
            return CaptionOutput(text="", engine="none")
        if self.engine == "template":
            return CaptionOutput(text=template, engine="template")
        if self._captioner is None:
            if self.engine == "lmstudio":
                self._captioner = LMStudioCaptioner(
                    model_name=self._model_name,
                    prompt_text=self._caption_prompt,
                    max_new_tokens=self._max_tokens,
                    temperature=self._temperature,
                    base_url=self._lmstudio_base_url,
                    max_image_edge=self._max_image_edge,
                )
            else:
                self._captioner = QwenLocalCaptioner(
                    model_name=self._model_name,
                    prompt_text=self._caption_prompt,
                    max_new_tokens=self._max_tokens,
                    temperature=self._temperature,
                    attn_implementation=self._qwen_attn_implementation,
                    min_pixels=self._qwen_min_pixels,
                    max_pixels=self._qwen_max_pixels,
                    max_image_edge=self._max_image_edge,
                )
        try:
            caption = self._captioner.describe(
                image_path=image_path,
                people=people,
                objects=objects,
                ocr_text=ocr_text,
                source_path=source_path or image_path,
            )
            if caption:
                return CaptionOutput(text=caption, engine=self.engine)
            if not self.fallback_to_template:
                return CaptionOutput(
                    text="",
                    engine=self.engine,
                    fallback=True,
                    error=f"{self.engine.upper()} returned empty output.",
                )
            return CaptionOutput(
                text=template,
                engine="template",
                fallback=True,
                error=f"{self.engine.upper()} returned empty output.",
            )
        except Exception as exc:
            if not self.fallback_to_template:
                raise
            return CaptionOutput(text=template, engine="template", fallback=True, error=str(exc))
