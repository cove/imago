from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ._caption_album import clean_text


DEFAULT_LMSTUDIO_MAX_NEW_TOKENS = 256
DEFAULT_LMSTUDIO_BASE_URL = "http://192.168.4.72:1234/v1"
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


@dataclass(frozen=True)
class CaptionDetails:
    text: str
    gps_latitude: str = ""
    gps_longitude: str = ""
    location_name: str = ""

    def __str__(self) -> str:
        return self.text

    def __contains__(self, item: object) -> bool:
        return str(item or "") in self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CaptionDetails):
            return (
                self.text == other.text
                and self.gps_latitude == other.gps_latitude
                and self.gps_longitude == other.gps_longitude
                and self.location_name == other.location_name
            )
        if isinstance(other, str):
            return self.text == other
        return False


_CAPTION_REASONING_MARKERS = (
    "the user wants",
    "analyze the input data",
    "visual analysis",
    "synthesize the visual content",
    "filename hint",
    "folder hint",
    "album title hint",
    "ocr/text in image",
    "ocr text hint",
    "detected objects",
    "album classification hint",
    "album focus hint",
    "cordell photo albums rules",
    "based on the rules",
    "i need to",
    "i should",
)


def _looks_like_reasoning_or_prompt_echo(value: str) -> bool:
    """Return True if the text looks like a model reasoning trace or prompt echo rather than a caption."""
    text = clean_text(value)
    if not text:
        return False
    lowered = text.casefold()
    marker_hits = sum(1 for marker in _CAPTION_REASONING_MARKERS if marker in lowered)
    if marker_hits >= 2:
        return True
    if text.startswith(("**1.", "1.", "* **filename:**", "- **filename:**")):
        return True
    if re.search(r"(?:^|\s)(?:\*\*|\*|-)?\s*(?:filename|folder|ocr/text in image|detected objects)\s*:", lowered):
        return True
    if re.search(r"(?:^|\s)(?:\*\*|\*|-)?\s*(?:album classification hint|album focus hint)\s*:", lowered):
        return True
    return False


def _extract_structured_json_payload(text: str) -> dict[str, object] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _lanczos_resize(image, new_size: tuple[int, int]):
    resampling = getattr(getattr(image, "Resampling", None), "LANCZOS", None)
    if resampling is None:
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel

            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        except Exception:  # pragma: no cover - Pillow always present in runtime
            resampling = 1
    return image.resize(new_size, resampling)


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
    return _lanczos_resize(image, new_size)


def _format_decimal_coordinate(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _parse_dms_coordinate(value: str) -> float | None:
    match = re.search(
        r"(?P<deg>\d{1,3})\s*[°º]\s*(?P<min>\d{1,2})\s*[′']\s*(?P<sec>\d{1,2}(?:\.\d+)?)?\s*[″\"]?\s*(?P<hem>[NSEW])",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    deg = float(match.group("deg"))
    minutes = float(match.group("min"))
    seconds = float(match.group("sec") or 0.0)
    hemisphere = str(match.group("hem") or "").upper()
    decimal = deg + (minutes / 60.0) + (seconds / 3600.0)
    if hemisphere in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _parse_decimal_coordinate(value: str, *, positive_hemisphere: str, negative_hemisphere: str) -> float | None:
    text = clean_text(value).replace("−", "-")
    if not text:
        return None
    match = re.fullmatch(
        rf"(?P<number>[+-]?\d{{1,3}}(?:\.\d+)?)\s*(?P<hem>[{positive_hemisphere}{negative_hemisphere}])?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    decimal = float(match.group("number"))
    hemisphere = str(match.group("hem") or "").upper()
    if hemisphere == negative_hemisphere.upper():
        decimal = -abs(decimal)
    elif hemisphere == positive_hemisphere.upper():
        decimal = abs(decimal)
    return decimal


def _normalize_gps_value(value: str, *, axis: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    decimal: float | None = None
    if any(symbol in text for symbol in ("°", "º", "′", "″", "'")):
        decimal = _parse_dms_coordinate(text)
    if decimal is None:
        if axis == "lat":
            decimal = _parse_decimal_coordinate(text, positive_hemisphere="N", negative_hemisphere="S")
        else:
            decimal = _parse_decimal_coordinate(text, positive_hemisphere="E", negative_hemisphere="W")
    if decimal is None:
        return ""
    return _format_decimal_coordinate(decimal)


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


def _lmstudio_caption_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "caption_payload",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "gps_latitude": {"type": "string"},
                    "gps_longitude": {"type": "string"},
                    "location_name": {"type": "string"},
                },
                "required": ["caption", "gps_latitude", "gps_longitude", "location_name"],
                "additionalProperties": False,
            },
        },
    }


def _lmstudio_error_preview(value: str, *, limit: int = 180) -> str:
    text = clean_text(value)
    if not text:
        return "<empty>"
    snippet = text[: max(1, int(limit))]
    if len(text) > len(snippet):
        snippet += "..."
    return snippet


def _parse_lmstudio_structured_caption_payload(
    value: object,
    *,
    finish_reason: str = "",
) -> tuple[str, list[tuple[str, str]], str, str, str]:
    raw = _decode_lmstudio_text(value)
    text = str(raw or "").strip()
    finish_note = f" finish_reason={finish_reason}." if str(finish_reason or "").strip() else ""
    if not text:
        raise RuntimeError(
            "LM Studio returned empty structured caption content. "
            "Check that the loaded model supports structured output and that the LM Studio server is current."
            f"{finish_note}"
        )
    # Strip <think>...</think> blocks produced by reasoning models so that intermediate
    # JSON objects inside the thinking block are not mistaken for the structured response.
    # If stripping empties the text, keep the original so _extract_structured_json_payload
    # can still find JSON embedded inside an unclosed <think> block.
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if stripped:
        text = stripped
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _extract_structured_json_payload(text)
        if payload is None:
            preview = _lmstudio_error_preview(text)
            raise RuntimeError(
                f"LM Studio returned invalid structured caption JSON: {exc.msg}; raw={preview!r}.{finish_note}"
            ) from exc
    if not isinstance(payload, dict):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio returned structured caption JSON that is not an object; raw={preview!r}.{finish_note}"
        )
    caption = payload.get("caption")
    if not isinstance(caption, str):
        preview = _lmstudio_error_preview(text)
        raise RuntimeError(
            f"LM Studio structured caption JSON is missing a caption string; raw={preview!r}.{finish_note}"
        )
    gps_latitude = _normalize_gps_value(str(payload.get("gps_latitude") or ""), axis="lat")
    gps_longitude = _normalize_gps_value(str(payload.get("gps_longitude") or ""), axis="lon")
    location_name = clean_text(payload.get("location_name"))
    return caption, gps_latitude, gps_longitude, location_name


def _parse_lmstudio_structured_caption(
    value: object,
    *,
    finish_reason: str = "",
) -> CaptionDetails:
    caption, gps_latitude, gps_longitude, location_name = _parse_lmstudio_structured_caption_payload(
        value,
        finish_reason=finish_reason,
    )
    return CaptionDetails(
        text=clean_text(caption),
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        location_name=location_name,
    )


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


def _lmstudio_stream_tokens(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = list(chunk.get("choices") or [])
                if not choices:
                    continue
                delta = dict(choices[0].get("delta") or {})
                content = delta.get("content")
                if content:
                    yield str(content)
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


def normalize_lmstudio_base_url(value: str, default: str = DEFAULT_LMSTUDIO_BASE_URL) -> str:
    text = str(value or "").strip() or str(default or DEFAULT_LMSTUDIO_BASE_URL)
    text = text.rstrip("/")
    if text.endswith("/v1"):
        return text
    return f"{text}/v1"


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
        stream: bool = False,
    ):
        self.model_name = str(model_name or "").strip()
        self.prompt_text = str(prompt_text or "").strip()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.base_url = normalize_lmstudio_base_url(base_url)
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.max_image_edge = max(0, int(max_image_edge))
        self.stream = bool(stream)
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
        prompt: str,
    ) -> CaptionDetails:
        resize_edge = int(self.max_image_edge) if self.max_image_edge > 0 else int(DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
        image_url = _build_data_url(image_path, resize_edge)
        payload = {
            "model": self._resolve_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a photo caption writer. "
                        "Return only valid JSON matching the response_format schema. "
                        "Put the final caption text in the caption field. "
                        "If any visible text is not in English, add an English translation in parentheses directly after each non-English phrase in the caption — for example: '敦煌历史文物展览 (Dunhuang Historical Relics Exhibition)'. "
                        "If the location is known confidently enough for online geocoding, set location_name to a concise English geocoding query such as 'Mogao Caves, Dunhuang, Gansu, China'. "
                        "Only set gps_latitude and gps_longitude when exact coordinates are explicitly visible in the image or OCR text. "
                        "If the exact GPS is not explicitly known, set both GPS fields to empty strings. "
                        "If no confident geocoding query is available, set location_name to an empty string. "
                        "Never mention raw filenames, folder names, or internal ids. "
                        "Do not include reasoning or extra fields."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "response_format": _lmstudio_caption_response_format(),
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "stream": self.stream,
        }
        if self.stream:
            print(f"  Running LM Studio model ({self._resolve_model_name()})...", end="", flush=True)
            tokens: list[str] = []
            for token in _lmstudio_stream_tokens(
                f"{self.base_url}/chat/completions",
                payload,
                self.timeout_seconds,
            ):
                tokens.append(token)
            print(f"\r\033[K", end="", flush=True)
            return _parse_lmstudio_structured_caption(
                "".join(tokens),
            )
        response = _lmstudio_request_json(
            f"{self.base_url}/chat/completions",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        choices = list(response.get("choices") or [])
        if not choices:
            return CaptionDetails(text="")
        message = dict(choices[0].get("message") or {})
        return _parse_lmstudio_structured_caption(
            message.get("content"),
            finish_reason=str(choices[0].get("finish_reason") or ""),
        )
