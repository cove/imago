from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request

from .ai_model_settings import default_ctm_model, default_ctm_validation_settings, default_lmstudio_base_url
from .xmp_sidecar import read_ctm_from_archive_xmp, write_ctm_to_archive_xmp

CTM_SCHEMA_NAME = "photoalbum_ctm_restoration"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_CREATOR_TOOL = "https://github.com/cove/imago"


@dataclass(slots=True)
class CTMResult:
    matrix: list[float]
    confidence: float
    warnings: list[str] = field(default_factory=list)
    reasoning_summary: str = ""
    model_name: str = ""
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": list(self.matrix),
            "confidence": float(self.confidence),
            "warnings": list(self.warnings),
            "reasoning_summary": str(self.reasoning_summary),
            "model_name": str(self.model_name),
            "source_path": str(self.source_path),
        }


class CTMValidationError(RuntimeError):
    pass


def _json_schema() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": CTM_SCHEMA_NAME,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 9,
                        "maxItems": 9,
                    },
                    "confidence": {"type": "number"},
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reasoning_summary": {"type": "string"},
                },
                "required": ["matrix", "confidence", "warnings", "reasoning_summary"],
                "additionalProperties": False,
            },
        },
    }


def _request_payload(*, image_path: Path, model_name: str, strict: bool) -> dict[str, object]:
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    user_text = (
        "Estimate a 3x3 color transformation matrix that reduces red-shift caused by cyan dye failure. "
        "Return only valid structured JSON. The matrix must be row-major with 9 numeric coefficients. "
        "Keep the transform conservative and suitable for archival chromatic restoration."
    )
    if strict:
        user_text += " If unsure, still return the best conservative matrix and explain warnings briefly."
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an archival color restoration assistant. "
                    "You analyze stitched historical scans and estimate conservative 3x3 CTMs."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "response_format": _json_schema(),
        "temperature": 0.0,
    }


def _post_json(url: str, payload: dict[str, object], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise CTMValidationError("LM Studio response did not include choices")
    message = dict(choices[0]).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    raise CTMValidationError("LM Studio response did not include text content")


def parse_ctm_response(text: str, *, model_name: str = "", source_path: str = "") -> CTMResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CTMValidationError(f"Invalid CTM JSON: {exc}") from exc
    matrix = payload.get("matrix")
    confidence = payload.get("confidence")
    warnings = payload.get("warnings")
    reasoning_summary = payload.get("reasoning_summary")
    if not isinstance(matrix, list) or len(matrix) != 9:
        raise CTMValidationError("CTM response must contain exactly 9 matrix coefficients")
    normalized_matrix: list[float] = []
    for value in matrix:
        try:
            number = float(value)
        except Exception as exc:
            raise CTMValidationError("CTM matrix coefficients must be numeric") from exc
        if not math.isfinite(number):
            raise CTMValidationError("CTM matrix coefficients must be finite")
        normalized_matrix.append(number)
    try:
        confidence_value = float(confidence)
    except Exception as exc:
        raise CTMValidationError("CTM confidence must be numeric") from exc
    if not math.isfinite(confidence_value):
        raise CTMValidationError("CTM confidence must be finite")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise CTMValidationError("CTM warnings must be a list of strings")
    if not isinstance(reasoning_summary, str):
        raise CTMValidationError("CTM reasoning_summary must be a string")
    return CTMResult(
        matrix=normalized_matrix,
        confidence=confidence_value,
        warnings=[str(item) for item in warnings],
        reasoning_summary=reasoning_summary.strip(),
        model_name=str(model_name or "").strip(),
        source_path=str(source_path or "").strip(),
    )


def validate_ctm_result(result: CTMResult, *, settings: dict[str, float] | None = None) -> CTMResult:
    effective = dict(default_ctm_validation_settings())
    if settings:
        effective.update(settings)
    min_confidence = float(effective.get("min_confidence", 0.0))
    max_abs_coefficient = float(effective.get("max_abs_coefficient", 10.0))
    max_row_sum = float(effective.get("max_row_sum", 10.0))
    max_clipping_ratio = float(effective.get("max_clipping_ratio", 1.0))

    if len(result.matrix) != 9:
        raise CTMValidationError("CTM must contain 9 coefficients")
    if result.confidence < min_confidence:
        result.warnings.append(f"confidence_below_threshold:{result.confidence:.3f}<{min_confidence:.3f}")
    for value in result.matrix:
        if abs(value) > max_abs_coefficient:
            raise CTMValidationError(f"CTM coefficient {value} exceeds max_abs_coefficient {max_abs_coefficient}")
    row_sums = [sum(abs(value) for value in result.matrix[index : index + 3]) for index in range(0, 9, 3)]
    if any(total > max_row_sum for total in row_sums):
        raise CTMValidationError(f"CTM row sum exceeds max_row_sum {max_row_sum}")

    clipping_ratio = estimate_clipping_ratio(result.matrix)
    if clipping_ratio > max_clipping_ratio:
        raise CTMValidationError(f"CTM clipping ratio {clipping_ratio:.4f} exceeds limit {max_clipping_ratio:.4f}")
    if clipping_ratio > 0.0:
        result.warnings.append(f"estimated_clipping_ratio:{clipping_ratio:.4f}")
    return result


def estimate_clipping_ratio(matrix: list[float]) -> float:
    if len(matrix) != 9:
        raise CTMValidationError("CTM must contain 9 coefficients")
    samples = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (0.5, 0.5, 1.0),
    ]
    out_of_range = 0
    total = 0
    for r, g, b in samples:
        outputs = [
            matrix[0] * r + matrix[1] * g + matrix[2] * b,
            matrix[3] * r + matrix[4] * g + matrix[5] * b,
            matrix[6] * r + matrix[7] * g + matrix[8] * b,
        ]
        for value in outputs:
            total += 1
            if value < 0.0 or value > 1.0:
                out_of_range += 1
    return float(out_of_range) / float(total or 1)


def generate_ctm_for_image(
    image_path: str | Path,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    validation_settings: dict[str, float] | None = None,
) -> CTMResult:
    image_path = Path(image_path)
    selected_model = str(model_name or default_ctm_model()).strip()
    if not selected_model:
        raise CTMValidationError("No CTM model configured")
    url = str(base_url or default_lmstudio_base_url()).rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for strict in (False, True, True):
        try:
            payload = _post_json(
                url,
                _request_payload(image_path=image_path, model_name=selected_model, strict=strict),
                timeout=timeout_seconds,
            )
            text = _extract_text_from_response(payload)
            result = parse_ctm_response(text, model_name=selected_model, source_path=str(image_path))
            return validate_ctm_result(result, settings=validation_settings)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise CTMValidationError(f"Failed to generate CTM for {image_path}: {last_error}")


def apply_ctm_to_jpeg(jpeg_path: str | Path, matrix: list[float] | tuple[float, ...]) -> None:
    jpeg_path = Path(jpeg_path)
    if len(matrix) != 9:
        raise CTMValidationError("CTM matrix must contain exactly 9 coefficients")

    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"CTM apply failed due to missing image dependencies:{exc}") from exc

    with Image.open(jpeg_path) as image:
        working = image.convert("RGB")
        pixels = np.array(working, dtype=np.uint8)

    transform = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    rgb = pixels.astype(np.float32) / 255.0
    corrected = np.einsum("...c,dc->...d", rgb, transform)
    corrected = np.clip(corrected, 0.0, 1.0)
    output = np.rint(corrected * 255.0).astype(np.uint8)

    Image.fromarray(output, mode="RGB").save(jpeg_path, format="JPEG", quality=100, subsampling=0)


def generate_and_store_ctm(
    image_path: str | Path,
    *,
    archive_sidecar_path: str | Path | None = None,
    force: bool = False,
    base_url: str | None = None,
    model_name: str | None = None,
    validation_settings: dict[str, float] | None = None,
) -> tuple[Path, CTMResult]:
    image_path = Path(image_path)
    archive_sidecar = Path(archive_sidecar_path) if archive_sidecar_path is not None else image_path.with_suffix(".xmp")
    if not force:
        cached = read_ctm_from_archive_xmp(archive_sidecar)
        if cached is not None:
            return archive_sidecar, cached
    result = generate_ctm_for_image(
        image_path,
        base_url=base_url,
        model_name=model_name,
        validation_settings=validation_settings,
    )
    write_ctm_to_archive_xmp(
        archive_sidecar,
        matrix=result.matrix,
        confidence=result.confidence,
        warnings=result.warnings,
        reasoning_summary=result.reasoning_summary,
        creator_tool=DEFAULT_CREATOR_TOOL,
        source_image_path=image_path.name,
        model_name=result.model_name,
    )
    return archive_sidecar, result
