from __future__ import annotations

import re
from pathlib import Path

from .model_store import HF_MODEL_CACHE_DIR
from .ai_ocr import (
    DEFAULT_QWEN_OCR_MAX_IMAGE_EDGE,
    DEFAULT_QWEN_OCR_MAX_NEW_TOKENS,
    _load_hf_model,
    _normalize_ocr_text,
    _resolve_local_hf_snapshot,
)
from ._caption_album import clean_text
from ._caption_prompts import _build_combined_qwen_prompt
from ._caption_lmstudio import (
    CaptionDetails,
    _extract_structured_json_payload,
    _normalize_gps_value,
    _resize_caption_image,
)

DEFAULT_QWEN_CAPTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LEGACY_QWEN_CAPTION_MODEL_ALIASES = {
    "qwen/qwen3.5-4b": DEFAULT_QWEN_CAPTION_MODEL,
}
DEFAULT_QWEN_AUTO_MAX_PIXELS = 786_432
QWEN_ATTN_IMPLEMENTATIONS = {"auto", "sdpa", "flash_attention_2", "eager"}


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

    return torch, AutoProcessor, AutoModelForImageTextToText


def normalize_qwen_attn_implementation(value: str, default: str = "auto") -> str:
    text = str(value or "").strip().lower()
    if text in QWEN_ATTN_IMPLEMENTATIONS:
        return text
    fallback = str(default or "auto").strip().lower()
    if fallback in QWEN_ATTN_IMPLEMENTATIONS:
        return fallback
    return "auto"


def _parse_qwen_json_output(raw: str) -> CaptionDetails:
    """Parse structured JSON output from a Qwen model inference, with plain-text fallback."""
    text = str(raw or "").strip()
    stripped = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    if stripped:
        text = stripped
    payload = _extract_structured_json_payload(text)
    if payload is not None:
        caption = payload.get("caption")
        if isinstance(caption, str) and caption.strip():
            gps_latitude = _normalize_gps_value(
                str(payload.get("gps_latitude") or ""), axis="lat"
            )
            gps_longitude = _normalize_gps_value(
                str(payload.get("gps_longitude") or ""), axis="lon"
            )
            location_name = clean_text(str(payload.get("location_name") or ""))
            return CaptionDetails(
                text=clean_text(caption),
                gps_latitude=gps_latitude,
                gps_longitude=gps_longitude,
                location_name=location_name,
            )
    return CaptionDetails(text=clean_text(text))


def _parse_qwen_combined_json_output(raw: str) -> tuple[str, str]:
    """Parse structured JSON output from a combined OCR+caption Qwen inference.
    Returns (ocr_text, caption_text).
    """
    text = str(raw or "").strip()
    stripped = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    if stripped:
        text = stripped
    payload = _extract_structured_json_payload(text)
    if payload is not None:
        ocr_text = _normalize_ocr_text(str(payload.get("ocr_text") or ""))
        caption = payload.get("caption")
        if isinstance(caption, str) and caption.strip():
            return ocr_text, clean_text(caption)
    return "", clean_text(text)


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
        stream: bool = False,
    ):
        self.model_name = (
            str(model_name or DEFAULT_QWEN_CAPTION_MODEL).strip()
            or DEFAULT_QWEN_CAPTION_MODEL
        )
        self.prompt_text = str(prompt_text or "").strip()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.attn_implementation = normalize_qwen_attn_implementation(
            attn_implementation
        )
        self.min_pixels = max(0, int(min_pixels))
        self.max_pixels = max(0, int(max_pixels))
        self.max_image_edge = max(0, int(max_image_edge))
        self.stream = bool(stream)
        self._processor = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        torch, AutoProcessor, AutoModelForImageTextToText = _load_qwen_transformers()

        HF_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_MODEL_CACHE_DIR)
        local_snapshot = _resolve_local_hf_snapshot(self.model_name)
        model_ref = (
            str(local_snapshot) if local_snapshot is not None else self.model_name
        )
        local_files_only = local_snapshot is not None
        processor_kwargs = {
            "trust_remote_code": True,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if self.min_pixels > 0:
            processor_kwargs["min_pixels"] = int(self.min_pixels)
        processor_kwargs["max_pixels"] = (
            int(self.max_pixels)
            if self.max_pixels > 0
            else int(DEFAULT_QWEN_AUTO_MAX_PIXELS)
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
            if (
                self.attn_implementation == "flash_attention_2"
                and not torch.cuda.is_available()
            ):
                resolved_attn = "auto"
            else:
                resolved_attn = self.attn_implementation
                load_kwargs["attn_implementation"] = resolved_attn
        # Prefer dtype over torch_dtype to avoid deprecation warnings on newer transformers.
        self._model = _load_hf_model(
            AutoModelForImageTextToText, model_ref, **load_kwargs
        )
        self._torch = torch

    def describe(
        self,
        image_path: str | Path,
        *,
        prompt: str,
    ) -> CaptionDetails:
        self._ensure_loaded()
        return _parse_qwen_json_output(self._infer_raw(image_path, prompt))

    def _infer_raw(
        self, image_path: str | Path, prompt: str, max_new_tokens: int | None = None
    ) -> str:
        """Run a single inference pass and return the raw decoded string."""
        from PIL import Image  # pylint: disable=import-outside-toplevel

        max_tokens = (
            int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        )
        image = Image.open(str(image_path)).convert("RGB")
        try:
            working_image = _resize_caption_image(
                image, int(DEFAULT_QWEN_OCR_MAX_IMAGE_EDGE)
            )
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
            kwargs: dict = {"max_new_tokens": max_tokens, "do_sample": do_sample}
            if do_sample:
                kwargs["temperature"] = self.temperature
                kwargs["top_p"] = 0.9
            if self.stream:
                import threading  # pylint: disable=import-outside-toplevel
                from transformers import (
                    TextIteratorStreamer,
                )  # pylint: disable=import-outside-toplevel

                tokenizer = getattr(self._processor, "tokenizer", self._processor)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore[arg-type]
                kwargs["streamer"] = streamer
                thread = threading.Thread(
                    target=self._model.generate,
                    kwargs={**inputs, **kwargs},
                    daemon=True,
                )
                tokens: list[str] = []
                with self._torch.inference_mode():
                    thread.start()
                    for token in streamer:
                        tokens.append(token)
                        partial = "".join(tokens)
                        display = partial[-120:] if len(partial) > 120 else partial
                        print(f"\r  {display}", end="", flush=True)
                    thread.join()
                print("\r\033[K", end="", flush=True)
                return "".join(tokens)
            with self._torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **kwargs)
            input_ids = inputs.get("input_ids")
            if (
                hasattr(generated_ids, "shape")
                and input_ids is not None
                and hasattr(input_ids, "shape")
            ):
                prompt_tokens = int(input_ids.shape[-1])
                generated_ids = generated_ids[:, prompt_tokens:]
            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return decoded[0] if decoded else ""
        finally:
            if "working_image" in locals() and working_image is not image:
                working_image.close()
            image.close()

    def describe_combined(
        self,
        image_path: str | Path,
        *,
        people: list[str],
        objects: list[str],
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> tuple[str, str]:
        """Single inference that returns (ocr_text, caption)."""
        self._ensure_loaded()
        prompt = _build_combined_qwen_prompt(
            people=people,
            objects=objects,
            source_path=source_path or image_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=photo_count,
            is_cover_page=is_cover_page,
        )
        max_tokens = self.max_new_tokens + DEFAULT_QWEN_OCR_MAX_NEW_TOKENS
        raw = self._infer_raw(image_path, prompt, max_new_tokens=max_tokens)
        return _parse_qwen_combined_json_output(raw)
