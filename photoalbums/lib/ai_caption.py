from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .model_store import HF_MODEL_CACHE_DIR


DEFAULT_BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_QWEN_CAPTION_MODEL = "Qwen/Qwen3.5-4B"
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


def build_page_caption(*, photo_count: int, people: list[str], objects: list[str], ocr_text: str) -> str:
    count = max(1, int(photo_count))
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)

    parts = [f"This album page contains {count} photo(s)."]
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


def _build_qwen_prompt(*, people: list[str], objects: list[str], ocr_text: str) -> str:
    people_list = dedupe(people)
    object_list = dedupe(objects)
    text = clean_text(ocr_text)
    lines = [
        "Describe this photo in detail",
    ]
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
    text = str(model_name or "").strip()
    if text:
        return text
    if normalized == "qwen":
        return DEFAULT_QWEN_CAPTION_MODEL
    if normalized == "blip":
        return DEFAULT_BLIP_CAPTION_MODEL
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
    ) -> str:
        prompt = self.prompt_text or _build_qwen_prompt(people=people, objects=objects, ocr_text=ocr_text)
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


class BlipLocalCaptioner:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BLIP_CAPTION_MODEL,
        max_new_tokens: int = 64,
        max_image_edge: int = 0,
    ):
        self.model_name = resolve_caption_model("blip", model_name)
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.max_image_edge = max(0, int(max_image_edge))
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        try:
            import torch  # pylint: disable=import-outside-toplevel
            from transformers import (  # pylint: disable=import-outside-toplevel
                BlipForConditionalGeneration,
                BlipProcessor,
            )
        except Exception as exc:
            raise RuntimeError(
                "BLIP captioning requires transformers and torch. Install with: pip install transformers torch"
            ) from exc

        HF_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_MODEL_CACHE_DIR)
        self._processor = BlipProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name, cache_dir=cache_dir)
        self._model.eval()
        self._torch = torch
        if torch.cuda.is_available():
            self._device = "cuda"
            self._model = self._model.to(self._device)

    def describe(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        ocr_text: str,
    ) -> str:
        del people, objects, ocr_text
        self._ensure_loaded()
        from PIL import Image  # pylint: disable=import-outside-toplevel

        image = Image.open(str(image_path)).convert("RGB")
        try:
            working_image = _resize_caption_image(image, self.max_image_edge)
            inputs = self._processor(images=working_image, return_tensors="pt")
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    inputs[key] = value.to(self._device)
            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=3,
                )
            decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
            return _normalize_caption(decoded[0] if decoded else "")
        finally:
            if "working_image" in locals() and working_image is not image:
                working_image.close()
            image.close()


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
    ) -> str:
        self._ensure_loaded()
        from PIL import Image  # pylint: disable=import-outside-toplevel

        prompt = self.prompt_text or _build_qwen_prompt(people=people, objects=objects, ocr_text=ocr_text)
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
        engine: str = "blip",
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
        normalized = str(engine or "blip").strip().lower()
        if normalized not in {"none", "template", "blip", "qwen", "lmstudio"}:
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
    ) -> CaptionOutput:
        template = build_template_caption(people=people, objects=objects, ocr_text=ocr_text)
        if self.engine == "none":
            return CaptionOutput(text="", engine="none")
        if self.engine == "template":
            return CaptionOutput(text=template, engine="template")
        if self._captioner is None:
            if self.engine == "blip":
                self._captioner = BlipLocalCaptioner(
                    model_name=self._model_name,
                    max_new_tokens=self._max_tokens,
                    max_image_edge=self._max_image_edge,
                )
            elif self.engine == "lmstudio":
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
