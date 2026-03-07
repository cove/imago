from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_store import HF_MODEL_CACHE_DIR


DEFAULT_QWEN_CAPTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _join_human(values: list[str]) -> str:
    clean = [str(item or "").strip() for item in values if str(item or "").strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def _dedupe(values: list[str]) -> list[str]:
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
    people_list = _dedupe(people)
    object_list = _dedupe(objects)
    text = _clean_text(ocr_text)

    parts: list[str] = []
    if people_list and object_list:
        parts.append(
            f"This photo appears to show {_join_human(people_list)} with {_join_human(object_list)} in view."
        )
    elif people_list:
        parts.append(f"This photo appears to show {_join_human(people_list)}.")
    elif object_list:
        parts.append(f"This photo includes {_join_human(object_list)}.")
    else:
        parts.append("This photo was indexed with no confident people or object matches.")

    if text:
        snippet = text[:180].strip()
        if len(text) > len(snippet):
            snippet += "..."
        parts.append(f'Visible text reads: "{snippet}".')
    return " ".join(parts).strip()


def _build_qwen_prompt(*, people: list[str], objects: list[str], ocr_text: str) -> str:
    people_list = _dedupe(people)
    object_list = _dedupe(objects)
    text = _clean_text(ocr_text)
    lines = [
        "Write one or two concise factual sentences describing this photo.",
        "Do not invent details that are not visible.",
        "Use the context below only as hints and prefer what you can see in the image.",
    ]
    if people_list:
        lines.append(f"Known people matches: {_join_human(people_list)}.")
    if object_list:
        lines.append(f"Detected objects: {_join_human(object_list)}.")
    if text:
        snippet = text[:220].strip()
        if len(text) > len(snippet):
            snippet += "..."
        lines.append(f'OCR text hint: "{snippet}".')
    lines.append("Return plain text only.")
    return "\n".join(lines)


def _normalize_caption(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("assistant:"):
        text = text.split(":", 1)[1].strip()
    return text


class QwenLocalCaptioner:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_CAPTION_MODEL,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
    ):
        self.model_name = str(model_name or DEFAULT_QWEN_CAPTION_MODEL).strip() or DEFAULT_QWEN_CAPTION_MODEL
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self._processor = None
        self._model = None
        self._torch = None

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
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "cache_dir": cache_dir,
        }
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
                images=[image],
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
        engine: str = "template",
        qwen_model: str = DEFAULT_QWEN_CAPTION_MODEL,
        qwen_max_tokens: int = 96,
        qwen_temperature: float = 0.2,
        fallback_to_template: bool = True,
    ):
        normalized = str(engine or "template").strip().lower()
        if normalized not in {"none", "template", "qwen"}:
            raise ValueError(f"Unsupported caption engine: {engine}")
        self.engine = normalized
        self.fallback_to_template = bool(fallback_to_template)
        self._qwen = None
        self._qwen_model = str(qwen_model or DEFAULT_QWEN_CAPTION_MODEL).strip() or DEFAULT_QWEN_CAPTION_MODEL
        self._qwen_max_tokens = int(qwen_max_tokens)
        self._qwen_temperature = float(qwen_temperature)

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
        if self._qwen is None:
            self._qwen = QwenLocalCaptioner(
                model_name=self._qwen_model,
                max_new_tokens=self._qwen_max_tokens,
                temperature=self._qwen_temperature,
            )
        try:
            caption = self._qwen.describe(
                image_path=image_path,
                people=people,
                objects=objects,
                ocr_text=ocr_text,
            )
            if caption:
                return CaptionOutput(text=caption, engine="qwen")
            if not self.fallback_to_template:
                return CaptionOutput(text="", engine="qwen", fallback=True, error="Qwen returned empty output.")
            return CaptionOutput(
                text=template,
                engine="template",
                fallback=True,
                error="Qwen returned empty output.",
            )
        except Exception as exc:
            if not self.fallback_to_template:
                raise
            return CaptionOutput(text=template, engine="template", fallback=True, error=str(exc))
