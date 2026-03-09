from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_store import HF_MODEL_CACHE_DIR


DEFAULT_BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_QWEN_CAPTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
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
    else:
        parts.append("This photo was indexed with no confident people or object matches.")

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
    else:
        parts.append("Across the page, no confident people or object matches were found.")

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
        "Write one or two concise factual sentences describing this photo.",
        "Do not invent details that are not visible.",
        "Use the context below only as hints and prefer what you can see in the image.",
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
    text = clean_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("assistant:"):
        text = text.split(":", 1)[1].strip()
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
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        attn_implementation: str = "auto",
        min_pixels: int = 0,
        max_pixels: int = 0,
        max_image_edge: int = 0,
    ):
        self.model_name = str(model_name or DEFAULT_QWEN_CAPTION_MODEL).strip() or DEFAULT_QWEN_CAPTION_MODEL
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
        try:
            import torch  # pylint: disable=import-outside-toplevel
            from transformers import (  # pylint: disable=import-outside-toplevel
                AutoModelForImageTextToText,
                AutoModelForVision2Seq,
                AutoProcessor,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen captioning requires transformers and torch. Install with: pip install transformers torch"
            ) from exc

        HF_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_MODEL_CACHE_DIR)
        processor_kwargs = {
            "trust_remote_code": True,
            "cache_dir": cache_dir,
        }
        if self.min_pixels > 0:
            processor_kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels > 0:
            processor_kwargs["max_pixels"] = int(self.max_pixels)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "cache_dir": cache_dir,
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
        try:
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                dtype="auto",
                **load_kwargs,
            )
        except TypeError:
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                **load_kwargs,
            )
        except Exception:
            # Fallback for environments where Vision2Seq auto-mapping is unavailable.
            try:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    dtype="auto",
                    **load_kwargs,
                )
            except TypeError:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    **load_kwargs,
                )
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

        prompt = _build_qwen_prompt(people=people, objects=objects, ocr_text=ocr_text)
        image = Image.open(str(image_path)).convert("RGB")
        try:
            working_image = _resize_caption_image(image, self.max_image_edge)
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
        max_tokens: int = 96,
        temperature: float = 0.2,
        qwen_attn_implementation: str = "auto",
        qwen_min_pixels: int = 0,
        qwen_max_pixels: int = 0,
        max_image_edge: int = 0,
        fallback_to_template: bool = True,
    ):
        normalized = str(engine or "blip").strip().lower()
        if normalized not in {"none", "template", "blip", "qwen"}:
            raise ValueError(f"Unsupported caption engine: {engine}")
        self.engine = normalized
        self.fallback_to_template = bool(fallback_to_template)
        self._captioner = None
        self._model_name = resolve_caption_model(normalized, model_name)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._qwen_attn_implementation = normalize_qwen_attn_implementation(qwen_attn_implementation)
        self._qwen_min_pixels = max(0, int(qwen_min_pixels))
        self._qwen_max_pixels = max(0, int(qwen_max_pixels))
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
            else:
                self._captioner = QwenLocalCaptioner(
                    model_name=self._model_name,
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
