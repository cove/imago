from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .model_store import HF_MODEL_CACHE_DIR
from .ai_model_settings import default_lmstudio_base_url, default_ocr_model
from ._caption_lmstudio import (
    _decode_lmstudio_text,
    _extract_structured_json_payload,
    _format_lmstudio_debug_response,
    _lanczos_resize,
)
from ._prompt_skill import required_section_text

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

DEFAULT_LOCAL_OCR_MODEL = "qwen/qwen3.5-9b"
# DEFAULT_LOCAL_OCR_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_LOCAL_OCR_MAX_NEW_TOKENS = 5128
DEFAULT_LOCAL_OCR_MAX_PIXELS = 4_194_304
DEFAULT_LOCAL_OCR_MAX_IMAGE_EDGE = 2048
DEFAULT_LMSTUDIO_OCR_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LMSTUDIO_OCR_TIMEOUT_SECONDS = 300.0
DEFAULT_LOCAL_OCR_PROMPT = (
    "Extract all visible text from this image.\n"
    "- Return only the extracted text, copied exactly as it appears.\n"
    "- Preserve line breaks when they are visually clear.\n"
    "- Do not describe the image.\n"
    "- Do not invent, guess, or fill in text that is not clearly visible.\n"
    "- Do not use placeholder or example text such as 'The quick brown fox' or 'Lorem ipsum'.\n"
    "- If there is no readable text, return an empty string."
)
_LEGACY_OCR_ENGINE_ALIASES = {
    "docstrange": "local",
    "qwen": "local",
}
_NO_TEXT_RESPONSES = {
    "none",
    "no text",
    "no visible text",
    "no readable text",
    "there is no text",
    "there is no visible text",
    "there is no readable text",
    "no text visible",
}
_OCR_REASONING_MARKERS = (
    "the user wants",
    "analyze the image",
    "transcribe the text found",
    "refine the transcription",
    "let's look",
    "looking closer",
    "actually,",
    "wait,",
    "no clear text",
)


def _normalize_lmstudio_ocr_base_url(value: str) -> str:
    text = str(value or "").strip() or default_lmstudio_base_url() or DEFAULT_LMSTUDIO_OCR_BASE_URL
    text = text.rstrip("/")
    if not text.endswith("/v1"):
        text = f"{text}/v1"
    return text


def ocr_system_prompt() -> str:
    return required_section_text("System Prompt - OCR")


def _emit_prompt_debug(
    debug_recorder,
    *,
    step: str,
    engine: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    source_path: str | Path | None = None,
    response: str = "",
    finish_reason: str = "",
    metadata: dict | None = None,
) -> None:
    if not callable(debug_recorder):
        return
    debug_recorder(
        step=str(step or "").strip(),
        engine=str(engine or "").strip(),
        model=str(model or "").strip(),
        prompt=str(prompt or ""),
        system_prompt=str(system_prompt or ""),
        source_path=source_path,
        prompt_source="default",
        response=str(response or ""),
        finish_reason=str(finish_reason or "").strip(),
        metadata=dict(metadata or {}),
    )


def _lmstudio_ocr_post(base_url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"LM Studio OCR request failed: {details or f'HTTP {exc.code}'}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {base_url}: {exc.reason}") from exc


def _lmstudio_ocr_select_model(base_url: str, timeout: float, requested_model: str = "") -> str:
    request = urllib.request.Request(f"{base_url}/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {base_url}: {exc.reason}") from exc
    model_ids = [
        str(row.get("id") or "").strip() for row in list(data.get("data") or []) if str(row.get("id") or "").strip()
    ]
    if not model_ids:
        raise RuntimeError("LM Studio did not return any models. Load a model in LM Studio first.")
    requested = str(requested_model or "").strip()
    if requested:
        if requested in model_ids:
            return requested
        raise RuntimeError(f"LM Studio OCR model '{requested}' is not loaded. Loaded models: {', '.join(model_ids)}")
    return model_ids[0]


def _build_ocr_data_url(image_path, max_image_edge: int, max_pixels: int) -> str:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    path = Path(image_path)
    image = Image.open(str(path)).convert("RGB")
    try:
        working = _resize_for_ocr(image, max_image_edge, max_pixels)
        buf = io.BytesIO()
        if path.suffix.lower() == ".png":
            mime = "image/png"
            working.save(buf, format="PNG")
        else:
            mime = "image/jpeg"
            working.save(buf, format="JPEG", quality=95)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:{mime};base64,{data}"
    finally:
        if "working" in locals() and working is not image:
            working.close()
        image.close()


def _normalize_ocr_engine(value: str) -> str:
    text = str(value or "none").strip().lower()
    return _LEGACY_OCR_ENGINE_ALIASES.get(text, text)


def _looks_like_ocr_reasoning(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.casefold()
    marker_hits = sum(1 for marker in _OCR_REASONING_MARKERS if marker in lowered)
    if marker_hits >= 2:
        return True
    if lowered.startswith("the user wants"):
        return True
    if re.search(r"^\s*\d+\.\s+\*\*", text):
        return True
    return False


def _normalize_ocr_text(value: object) -> str:
    if isinstance(value, list):
        value = _decode_lmstudio_text(value)
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("assistant:"):
        text = text.split(":", 1)[1].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"^\s*(?:the (?:visible|extracted) text(?: in (?:the )?image)? is|extracted text|ocr text)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if text.casefold() in _NO_TEXT_RESPONSES:
        return ""
    if _looks_like_ocr_reasoning(text):
        return ""
    return text


def _lmstudio_ocr_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ocr_payload",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }


def _recover_truncated_ocr_text(raw: str) -> str | None:
    """Extract partial text from a truncated JSON response (finish_reason=length)."""
    match = re.search(r'"text"\s*:\s*"', raw)
    if not match:
        return None
    fragment = raw[match.end() :]
    # Strip trailing incomplete escape sequences (e.g. \u4, \u, or lone \)
    fragment = re.sub(r"\\(?:[uU][0-9a-fA-F]{0,3}|.?)$", "", fragment)
    try:
        return json.loads('"' + fragment + '"')
    except json.JSONDecodeError:
        return None


def _fix_json_escaping(text: str) -> str:
    """Fix common JSON escaping issues that can cause parsing failures."""
    # Fix excessive backslash escaping that can occur in OCR responses
    # Replace sequences like \\\\\\\\\ with proper escaping
    text = re.sub(r"\\\\\\\\\\\\", r"\\", text)
    text = re.sub(r"\\\\\\\\", r"\\", text)
    text = re.sub(r"\\\\", r"\\", text)

    # Fix other common escaping issues
    text = re.sub(r'\\\\"', r'"', text)
    text = re.sub(r'\\"', r'"', text)

    # Remove any remaining invalid escape sequences that aren't valid JSON escapes
    text = re.sub(r'\\[^"\\/bfnrtu]', r"", text)

    return text


def _parse_lmstudio_structured_ocr(value: object, *, finish_reason: str = "") -> str:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(
            "LM Studio returned empty structured OCR content. "
            "Check that the loaded model supports structured output and that the LM Studio server is current."
            f"{finish_note}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Try to fix common JSON escaping issues before falling back to extraction
        fixed_text = _fix_json_escaping(text)
        try:
            payload = json.loads(fixed_text)
        except json.JSONDecodeError:
            payload = _extract_structured_json_payload(text)
            if payload is None:
                if str(finish_reason or "").strip() == "length":
                    recovered = _recover_truncated_ocr_text(text)
                    if recovered is not None:
                        return _normalize_ocr_text(recovered)
                snippet = text[:180] + ("..." if len(text) > 180 else "")
                raise RuntimeError(
                    f"LM Studio returned invalid structured OCR JSON: {exc.msg}; raw={snippet!r}.{finish_note}"
                ) from exc
    if not isinstance(payload, dict):
        snippet = text[:180] + ("..." if len(text) > 180 else "")
        raise RuntimeError(
            f"LM Studio returned structured OCR JSON that is not an object; raw={snippet!r}.{finish_note}"
        )
    extracted = payload.get("text")
    if not isinstance(extracted, str):
        snippet = text[:180] + ("..." if len(text) > 180 else "")
        raise RuntimeError(f"LM Studio structured OCR JSON is missing a text string; raw={snippet!r}.{finish_note}")
    return _normalize_ocr_text(extracted)


def _resize_for_ocr(image, max_image_edge: int, max_pixels: int):
    width, height = image.size
    working = image
    longest = max(width, height)
    if int(max_image_edge) > 0 and longest > int(max_image_edge):
        scale = float(max_image_edge) / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        working = _lanczos_resize(image, new_size)
        width, height = working.size

    pixels = int(width) * int(height)
    if int(max_pixels) > 0 and pixels > int(max_pixels):
        scale = math.sqrt(float(max_pixels) / float(max(1, pixels)))
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resized = _lanczos_resize(working, new_size)
        if working is not image:
            working.close()
        working = resized
    return working


def _load_hf_model(cls, model_ref: str, **load_kwargs):
    """Load a HuggingFace model, falling back from dtype= to torch_dtype= for older transformers."""
    try:
        return cls.from_pretrained(model_ref, dtype="auto", **load_kwargs)
    except TypeError:
        return cls.from_pretrained(model_ref, torch_dtype="auto", **load_kwargs)


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


def _resolve_local_model_ref(model_name: str) -> tuple[str, bool]:
    text = str(model_name or DEFAULT_LOCAL_OCR_MODEL).strip() or DEFAULT_LOCAL_OCR_MODEL
    candidate = Path(text).expanduser()
    if candidate.exists():
        return str(candidate), True
    local_snapshot = _resolve_local_hf_snapshot(text)
    if local_snapshot is not None:
        return str(local_snapshot), True
    raise RuntimeError(
        "Local HF OCR is configured for local-only inference. "
        "Download the model into the local Hugging Face cache under models/photoalbums/hf "
        f"or set LOCAL_OCR_MODEL to a local model path. Current model: {text}"
    )


def _load_hf_transformers():
    try:
        import torch  # pylint: disable=import-outside-toplevel
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except Exception as exc:
        raise RuntimeError("Local HF inference requires a compatible transformers/torch install.") from exc
    return torch, AutoProcessor, AutoModelForImageTextToText


class OCREngine:
    def __init__(
        self,
        *,
        engine: str = "local",
        language: str = "eng",
        model_name: str = "",
        base_url: str = "",
    ):
        self.engine = _normalize_ocr_engine(engine)
        self.language = str(language or "eng").strip() or "eng"
        self.base_url = _normalize_lmstudio_ocr_base_url(base_url) if self.engine == "lmstudio" else ""
        self._model_name = str(
            model_name or os.environ.get("LOCAL_OCR_MODEL") or os.environ.get("QWEN_OCR_MODEL") or default_ocr_model()
        ).strip()
        self._processor = None
        self._model = None
        self._torch = None
        self._lmstudio_model: str = ""

        if self.engine in {"none", "local", "lmstudio"}:
            return
        raise ValueError(f"Unsupported OCR engine: {engine}")

    @property
    def effective_model_name(self) -> str:
        """Return the actual model name used, resolved after any lazy API lookup."""
        if self.engine == "local":
            return str(self._model_name or DEFAULT_LOCAL_OCR_MODEL)
        if self.engine == "lmstudio":
            return str(self._lmstudio_model or self._model_name)
        return ""

    def _ensure_loaded(self) -> None:
        if self.engine != "local":
            return
        if self._processor is not None and self._model is not None:
            return
        model_ref, local_files_only = _resolve_local_model_ref(self._model_name or DEFAULT_LOCAL_OCR_MODEL)
        torch, AutoProcessor, AutoModelForImageTextToText = _load_hf_transformers()

        HF_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(HF_MODEL_CACHE_DIR)
        self._processor = AutoProcessor.from_pretrained(
            model_ref,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            max_pixels=DEFAULT_LOCAL_OCR_MAX_PIXELS,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        self._model = _load_hf_model(AutoModelForImageTextToText, model_ref, **load_kwargs)
        self._torch = torch

    def _read_text_lmstudio(self, image_path: str | Path, *, debug_recorder=None, debug_step: str = "ocr") -> str:
        if not self._lmstudio_model:
            self._lmstudio_model = _lmstudio_ocr_select_model(
                self.base_url,
                DEFAULT_LMSTUDIO_OCR_TIMEOUT_SECONDS,
                self._model_name,
            )
        data_url = _build_ocr_data_url(image_path, DEFAULT_LOCAL_OCR_MAX_IMAGE_EDGE, DEFAULT_LOCAL_OCR_MAX_PIXELS)
        payload = {
            "model": self._lmstudio_model,
            "messages": [
                {
                    "role": "system",
                    "content": ocr_system_prompt(),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": DEFAULT_LOCAL_OCR_PROMPT},
                    ],
                },
            ],
            "response_format": _lmstudio_ocr_response_format(),
            "max_tokens": DEFAULT_LOCAL_OCR_MAX_NEW_TOKENS,
            "temperature": 0.0,
            "stream": False,
        }
        raw_response = ""
        finish_reason = ""
        error_text = ""
        try:
            response = _lmstudio_ocr_post(self.base_url, payload, DEFAULT_LMSTUDIO_OCR_TIMEOUT_SECONDS)
            choices = list(response.get("choices") or [])
            if not choices:
                return ""
            message = dict(choices[0].get("message") or {})
            finish_reason = str(choices[0].get("finish_reason") or "")
            raw_response = _format_lmstudio_debug_response(message.get("content"))
            if not raw_response:
                raw_response = _format_lmstudio_debug_response(message)
            return _parse_lmstudio_structured_ocr(
                message.get("content"),
                finish_reason=finish_reason,
            )
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            metadata = {}
            if error_text:
                metadata["error"] = error_text
            _emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name,
                prompt=DEFAULT_LOCAL_OCR_PROMPT,
                system_prompt=ocr_system_prompt(),
                source_path=image_path,
                response=raw_response,
                finish_reason=finish_reason,
                metadata=metadata,
            )

    def read_text(self, image_path: str | Path, *, debug_recorder=None, debug_step: str = "ocr") -> str:
        path = Path(image_path)
        if self.engine == "none":
            return ""
        if self.engine == "lmstudio":
            return self._read_text_lmstudio(path, debug_recorder=debug_recorder, debug_step=debug_step)
        if self.engine != "local":
            return ""

        self._ensure_loaded()

        from PIL import Image  # pylint: disable=import-outside-toplevel

        image = Image.open(str(path)).convert("RGB")
        prompt_text = ""
        response_text = ""
        error_text = ""
        try:
            working_image = _resize_for_ocr(
                image,
                DEFAULT_LOCAL_OCR_MAX_IMAGE_EDGE,
                DEFAULT_LOCAL_OCR_MAX_PIXELS,
            )
            if hasattr(self._processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": DEFAULT_LOCAL_OCR_PROMPT},
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
                prompt_text = DEFAULT_LOCAL_OCR_PROMPT

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

            with self._torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=DEFAULT_LOCAL_OCR_MAX_NEW_TOKENS,
                    do_sample=False,
                )

            input_ids = inputs.get("input_ids")
            if hasattr(generated_ids, "shape") and input_ids is not None and hasattr(input_ids, "shape"):
                prompt_tokens = int(input_ids.shape[-1])
                generated_ids = generated_ids[:, prompt_tokens:]

            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            response_text = decoded[0] if decoded else ""
            return _normalize_ocr_text(response_text)
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            metadata = {}
            if error_text:
                metadata["error"] = error_text
            _emit_prompt_debug(
                debug_recorder,
                step=debug_step,
                engine=self.engine,
                model=self.effective_model_name,
                prompt=prompt_text,
                source_path=path,
                response=response_text,
                metadata=metadata,
            )
            if "working_image" in locals() and working_image is not image:
                working_image.close()
            image.close()


def extract_keywords(text: str, *, max_keywords: int = 15) -> list[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", str(text or "")):
        value = token.strip().strip("'").lower()
        if len(value) < 3 or value in STOPWORDS:
            continue
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    return [word for word, _ in ranked[: max(1, int(max_keywords))]]
