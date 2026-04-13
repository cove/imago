"""Detect individual photo regions within stitched view JPGs using a vision model.

The prompt asks for normalised bounding boxes, but some model/server combinations
still emit pixel coordinates. Results are written directly into the view image's
XMP sidecar as an MWG-RS RegionList. The XMP is both the cache and the ground truth.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ._lmstudio_helpers import emit_prompt_debug as _emit_prompt_debug
from .ai_model_settings import default_lmstudio_base_url, default_view_region_model, default_view_region_models
from ._caption_lmstudio import normalize_lmstudio_base_url

if TYPE_CHECKING:
    from .prompt_debug import PromptDebugSession

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_IMAGE_EDGE = 100
_MAX_RETRIES = 3
_MAX_SINGLE_REGION_PAGE_FRACTION = 0.90
_MIN_TOTAL_REGION_PAGE_FRACTION = 0.50
_MAX_OVERLAP_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionResult:
    """A detected photo region in pixel coordinates (top-left origin)."""

    index: int
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    caption_hint: str = ""
    person_names: list[str] = field(default_factory=list)


@dataclass
class RegionWithCaption:
    region: RegionResult
    caption: str
    caption_ambiguous: bool = False


@dataclass(frozen=True)
class RegionFailure:
    """Why a single region (or region pair) failed validation."""

    region_index: int
    reason: str  # "zero_area", "full_page", "overlap", "insufficient_page_coverage"
    severity: str  # "hard" — always requires repair
    overlap_with: int | None = None  # index of the overlapping peer
    overlap_fraction: float | None = None  # fraction of the smaller box that was overlapped
    page_fraction: float | None = None  # fraction of the page covered by the accepted region set


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a candidate region set."""

    valid: bool  # True when failures is empty
    kept: list[RegionResult]  # regions that survive all checks, sorted by index
    failures: list[RegionFailure]  # per-region and pairwise failure records


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def pixel_to_mwgrs(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert pixel top-left bounds to MWG-RS normalised centre-point coords.

    Returns (cx, cy, nw, nh) all in [0, 1] relative to image dimensions.
    MWG-RS uses centre x/y, fractional width/height.
    """
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


# ---------------------------------------------------------------------------
# Caption association
# ---------------------------------------------------------------------------


def associate_captions(
    regions: list[RegionResult],
    captions: list[dict],
    img_width: int,
) -> list[RegionWithCaption]:
    """Assign captions to regions using nearest-centre spatial heuristic.

    captions is a list of dicts with keys: text, x, y, w, h (pixel coords).
    If no caption has position data, or captions are equidistant within 10% of
    image width to two regions, all regions receive the first caption and
    caption_ambiguous=True.
    """
    ambiguity_threshold = img_width * 0.10

    def region_centre(r: RegionResult) -> tuple[float, float]:
        return r.x + r.width / 2.0, r.y + r.height / 2.0

    def caption_centre(c: dict) -> tuple[float, float] | None:
        try:
            cx = float(c["x"]) + float(c["w"]) / 2.0
            cy = float(c["y"]) + float(c["h"]) / 2.0
            return cx, cy
        except (KeyError, TypeError, ValueError):
            return None

    def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    results: list[RegionWithCaption] = []

    for cap in captions:
        cc = caption_centre(cap)
        if cc is None:
            # No position — broadcast to all
            broadcast_text = str(cap.get("text") or "").strip()
            return [RegionWithCaption(r, broadcast_text, caption_ambiguous=True) for r in regions]

        dists = [distance(region_centre(r), cc) for r in regions]
        if len(dists) < 2:
            best_idx = 0
        else:
            sorted_dists = sorted(dists)
            if sorted_dists[1] - sorted_dists[0] < ambiguity_threshold:
                # Ambiguous — broadcast
                broadcast_text = str(cap.get("text") or "").strip()
                return [RegionWithCaption(r, broadcast_text, caption_ambiguous=True) for r in regions]
            best_idx = dists.index(sorted_dists[0])

        results.append(RegionWithCaption(regions[best_idx], str(cap.get("text") or "").strip()))

    # Regions that got no caption assigned
    assigned = {rwc.region.index for rwc in results}
    for r in regions:
        if r.index not in assigned:
            results.append(RegionWithCaption(r, ""))

    results.sort(key=lambda rwc: rwc.region.index)
    return results


# ---------------------------------------------------------------------------
# LM Studio vision call
# ---------------------------------------------------------------------------

_REGION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "view_regions",
        "strict": False,
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "label": {"type": "string"},
                },
                "required": ["box_2d", "label"],
                "additionalProperties": False,
            },
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a Vision-Coordinate Engine. "
    "Your ONLY task is to detect the bounding boxes of physical photograph prints on a scrapbook or photo album page. "
    "Detect only the outer boundaries of the printed photo rectangles. "
    "Ignore objects inside photos (people, cars, buildings, scenery). "
    "Ignore text labels, handwritten notes, caption strips, and album background paper. "
    "Output ONLY a raw JSON array in the format: "
    '[{"box_2d": [ymin, xmin, ymax, xmax], "label": "photograph"}]. '
    "Use a normalized scale of 0–1000 for all coordinates "
    "(ymin/ymax are thousandths of image height, xmin/xmax are thousandths of image width)."
)


_USER_PROMPT = (
    "Step 1: Scan the page and count how many distinct rectangular photo prints are present. "
    "Step 2: For each photo, identify the full outer boundary of the print ? "
    "not the subject inside it. The album background paper, caption strips, and text labels "
    "between prints are separators, not photos. "
    "Step 3: Return a JSON array of bounding boxes ? one per photo, no overlaps. "
    'Each element: {"box_2d": [ymin, xmin, ymax, xmax], "label": "photograph"}. '
    "Coordinates are integers 0?1000 (ymin < ymax, xmin < xmax). "
    "Self-correction: if a box is entirely inside another box, remove the outer box."
)

_STRICT_USER_PROMPT = (
    "Carefully identify every distinct physical photograph print on this album page. "
    "For each print: detect the full outer boundary of the paper rectangle itself ? "
    "not objects inside the photo. "
    "Ignore album background paper, text labels, handwritten notes, and caption strips. "
    "Return ONLY a valid JSON array. "
    'Each element MUST be {"box_2d": [ymin, xmin, ymax, xmax], "label": "photograph"}. '
    "Coordinates are integers 0?1000 (ymin < ymax, xmin < xmax). No other text or keys."
)


def _prompt_image_dims(img_w: int, img_h: int) -> str:
    return f"This image is {img_w}?{img_h} pixels. " if img_w > 0 and img_h > 0 else ""


def _build_prompt(img_w: int, img_h: int, instructions: str) -> str:
    return f"{_prompt_image_dims(img_w, img_h)}{instructions}"


def _build_user_prompt(img_w: int, img_h: int) -> str:
    return _build_prompt(img_w, img_h, _USER_PROMPT)


def _build_user_prompt_strict(img_w: int, img_h: int) -> str:
    return _build_prompt(img_w, img_h, _STRICT_USER_PROMPT)


def _build_repair_prompt(
    prior_regions: list[RegionResult],
    failures: list[RegionFailure],
    *,
    img_w: int,
    img_h: int,
) -> str:
    """Build a retry prompt that feeds back the previous region set and validation errors."""
    lines = [
        "The previous detection produced invalid regions that must be corrected.",
        "",
        "Prior region set (all regions as returned, box_2d = [ymin, xmin, ymax, xmax] in 0–1000):",
    ]
    for r in prior_regions:
        if img_w > 0 and img_h > 0:
            ymin = round(r.y / img_h * 1000)
            xmin = round(r.x / img_w * 1000)
            ymax = round((r.y + r.height) / img_h * 1000)
            xmax = round((r.x + r.width) / img_w * 1000)
            lines.append(f"  index={r.index}: box_2d=[{ymin}, {xmin}, {ymax}, {xmax}]")
        else:
            lines.append(f"  index={r.index}: x={r.x}, y={r.y}, width={r.width}, height={r.height}")
    lines.append("")
    lines.append("Validation failures that MUST be fixed in your new response:")
    for f in failures:
        if f.reason == "overlap":
            pct = int(round((f.overlap_fraction or 0) * 100))
            lines.append(
                f"  Region {f.region_index} overlaps region {f.overlap_with} by {pct}% of the smaller box"
                f" (limit: 5%) — these two regions must not overlap."
            )
        elif f.reason == "zero_area":
            lines.append(
                f"  Region {f.region_index} has zero or negative area —"
                f" it must be a valid non-degenerate rectangle."
            )
        elif f.reason == "full_page":
            lines.append(
                f"  Region {f.region_index} covers ≥90% of the page —"
                f" this is the album background, not a photo."
            )
        elif f.reason == "insufficient_page_coverage":
            pct = int(round((f.page_fraction or 0) * 100))
            lines.append(
                f"  The combined region set covers only {pct}% of the page"
                f" (minimum: 50%) — you likely missed one or more photographs."
            )
        else:
            lines.append(f"  Region {f.region_index}: {f.reason}.")
    lines.extend(
        [
            "",
            "Return a COMPLETE revised JSON array for this same image that fixes ALL failures listed above.",
            "Each physical photograph must have exactly one non-overlapping bounding box.",
            "",
            "Detect only the outer boundary of each physical photo print — "
            "ignore album background paper, text labels, and objects inside photos.",
        ]
    )
    return "\n".join(lines)


def _build_data_url_with_size(image_path: Path, max_edge: int) -> tuple[str, int, int]:
    """Return a base64 data URL plus the resized dimensions sent to the model."""
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    if max_edge <= 0 and image_path.suffix.lower() in {".jpg", ".jpeg"}:
        jpeg_bytes = image_path.read_bytes()
        allow_large_pillow_images(Image)
        with Image.open(io.BytesIO(jpeg_bytes)) as image:
            width, height = image.size
        data = base64.b64encode(jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{data}", width, height

    allow_large_pillow_images(Image)
    image = Image.open(str(image_path)).convert("RGB")
    try:
        w, h = image.size
        longest = max(w, h)
        if longest > max_edge > 0:
            scale = max_edge / longest
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        resized_w, resized_h = image.size
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{data}", resized_w, resized_h
    finally:
        image.close()


def _lmstudio_post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"LM Studio request failed: {details or f'HTTP {exc.code}'}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LM Studio is unreachable at {url}: {exc.reason}") from exc


def _call_vision_model(
    image_path: Path,
    *,
    model: str,
    base_url: str,
    timeout: float,
    img_w: int,
    img_h: int,
    strict_prompt: bool = False,
    prior_regions: list[RegionResult] | None = None,
    prior_failures: list[RegionFailure] | None = None,
    album_context: str = "",
    page_caption: str = "",
    people_roster: dict[str, str] | None = None,
    attempt_number: int = 1,
    debug_recorder=None,
) -> list[RegionResult]:
    image_url, resized_w, resized_h = _build_data_url_with_size(image_path, DEFAULT_MAX_IMAGE_EDGE)
    if prior_regions is not None and prior_failures is not None:
        user_text = _build_repair_prompt(prior_regions, prior_failures, img_w=img_w, img_h=img_h)
    else:
        user_text = _build_user_prompt_strict(img_w, img_h) if strict_prompt else _build_user_prompt(img_w, img_h)
    clean_album_context = str(album_context or "").strip()
    if clean_album_context:
        user_text = user_text + f" Album context: {clean_album_context}."
    clean_page_caption = str(page_caption or "").strip()
    if clean_page_caption:
        user_text = user_text + f" Page caption context: {clean_page_caption}."
    roster_entries = []
    for shorthand, full_name in sorted((people_roster or {}).items()):
        clean_shorthand = str(shorthand or "").strip().lower()
        clean_full_name = str(full_name or "").strip()
        if clean_shorthand and clean_full_name:
            roster_entries.append(f"{clean_shorthand} -> {clean_full_name}")
    if roster_entries:
        user_text = (
            user_text
            + " People roster for expanding hyphenated shorthand when the caption context supports it: "
            + "; ".join(roster_entries)
            + "."
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "response_format": _REGION_RESPONSE_FORMAT,
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    is_repair = prior_regions is not None and prior_failures is not None
    debug_metadata = {
        "attempt_number": int(attempt_number),
        "strict_prompt": bool(strict_prompt),
        "repair_prompt": is_repair,
        "prior_region_count": len(prior_regions) if prior_regions is not None else 0,
        "prior_failure_count": len(prior_failures) if prior_failures is not None else 0,
        "base_url": str(base_url or ""),
        "original_image_width": int(img_w),
        "original_image_height": int(img_h),
        "sent_image_width": int(resized_w),
        "sent_image_height": int(resized_h),
    }
    try:
        response = _lmstudio_post(f"{base_url}/chat/completions", payload, timeout)
    except Exception as exc:
        _emit_prompt_debug(
            debug_recorder,
            step="view_regions",
            engine="lmstudio",
            model=model,
            prompt=user_text,
            system_prompt=_SYSTEM_PROMPT,
            source_path=image_path,
            prompt_source="runtime",
            response="",
            metadata={**debug_metadata, "error": str(exc)},
        )
        raise
    response_text = json.dumps(response, ensure_ascii=False, sort_keys=True)
    choices = list(response.get("choices") or [])
    if not choices:
        _emit_prompt_debug(
            debug_recorder,
            step="view_regions",
            engine="lmstudio",
            model=model,
            prompt=user_text,
            system_prompt=_SYSTEM_PROMPT,
            source_path=image_path,
            prompt_source="runtime",
            response=response_text,
            metadata={**debug_metadata, "error": "LM Studio returned no choices in response"},
        )
        raise RuntimeError("LM Studio returned no choices in response")
    finish_reason = str(choices[0].get("finish_reason") or "").strip()
    content = choices[0].get("message", {}).get("content", "")
    # The model reports the pixel dimensions it saw, and we scale back to the original image.
    try:
        results = _parse_region_response(
            content,
            img_w=img_w,
            img_h=img_h,
        )
    except Exception as exc:
        _emit_prompt_debug(
            debug_recorder,
            step="view_regions",
            engine="lmstudio",
            model=model,
            prompt=user_text,
            system_prompt=_SYSTEM_PROMPT,
            source_path=image_path,
            prompt_source="runtime",
            response=response_text,
            finish_reason=finish_reason,
            metadata={**debug_metadata, "error": str(exc)},
        )
        raise
    _emit_prompt_debug(
        debug_recorder,
        step="view_regions",
        engine="lmstudio",
        model=model,
        prompt=user_text,
        system_prompt=_SYSTEM_PROMPT,
        source_path=image_path,
        prompt_source="runtime",
        response=response_text,
        finish_reason=finish_reason,
        metadata={
            **debug_metadata,
            "returned_region_count": len(results),
        },
    )
    return results



def _parse_region_response(
    content: str,
    *,
    img_w: int = 0,
    img_h: int = 0,
) -> list[RegionResult]:
    """Parse a box_2d region response from the model.

    The model returns a JSON array of {"box_2d": [ymin, xmin, ymax, xmax], "label": "photograph"}
    where coordinates are in the 0–1000 range (thousandths of image dimensions).
    """
    text = str(content or "").strip()
    if not text:
        raise RuntimeError("Empty response from vision model")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        if start >= 0:
            try:
                payload = json.loads(text[start:])
            except json.JSONDecodeError:
                raise RuntimeError(f"Could not parse JSON array from model response: {text[:200]!r}")
        else:
            raise RuntimeError(f"No JSON array in model response: {text[:200]!r}")
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON array from vision model, got: {type(payload).__name__}")
    results: list[RegionResult] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            box = list(item["box_2d"])
            if len(box) != 4:
                log.warning("Skipping region with wrong box_2d length %r", item)
                continue
            ymin, xmin, ymax, xmax = (max(0.0, min(1000.0, float(v))) for v in box)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping malformed region entry %r: %s", item, exc)
            continue
        if ymin >= ymax or xmin >= xmax:
            log.warning("Skipping degenerate box_2d %r", box)
            continue
        if img_w > 0 and img_h > 0:
            x = max(0, int(round(xmin / 1000 * img_w)))
            y = max(0, int(round(ymin / 1000 * img_h)))
            w = max(1, int(round((xmax - xmin) / 1000 * img_w)))
            h = max(1, int(round((ymax - ymin) / 1000 * img_h)))
        else:
            # No image dims supplied — store as-is (test use only)
            x, y = int(xmin), int(ymin)
            w, h = max(1, int(xmax - xmin)), max(1, int(ymax - ymin))
        results.append(
            RegionResult(
                index=len(results),
                x=x,
                y=y,
                width=w,
                height=h,
            )
        )
    return results


def _intersection_area(left: RegionResult, right: RegionResult) -> int:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def validate_region_set(
    regions: list[RegionResult],
    *,
    img_w: int,
    img_h: int,
) -> ValidationResult:
    """Validate a candidate region set and return a structured result.

    Checks each region for zero-area, single-region full-page coverage, pairwise
    overlap, and minimum combined page coverage across the accepted region set.
    All failures are hard-invalid and require repair before the set can be accepted.
    Returns ValidationResult with valid=True only when failures is empty.
    """
    img_area = img_w * img_h
    failures: list[RegionFailure] = []
    clamped: list[RegionResult] = []

    for region in regions:
        left = max(0, region.x)
        top = max(0, region.y)
        right = min(img_w, region.x + region.width)
        bottom = min(img_h, region.y + region.height)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            failures.append(RegionFailure(region_index=region.index, reason="zero_area", severity="hard"))
            continue
        if img_area > 0 and (width * height) / img_area >= _MAX_SINGLE_REGION_PAGE_FRACTION:
            failures.append(RegionFailure(region_index=region.index, reason="full_page", severity="hard"))
            continue
        clamped.append(replace(region, x=left, y=top, width=width, height=height))

    ranked = sorted(clamped, key=lambda r: (-r.confidence, -(r.width * r.height), r.index))
    kept: list[RegionResult] = []
    for candidate in ranked:
        overlap_failure: RegionFailure | None = None
        for existing in kept:
            intersection = _intersection_area(candidate, existing)
            smaller_area = min(candidate.width * candidate.height, existing.width * existing.height)
            if smaller_area > 0 and (intersection / smaller_area) >= _MAX_OVERLAP_FRACTION:
                frac = intersection / smaller_area
                overlap_failure = RegionFailure(
                    region_index=candidate.index,
                    reason="overlap",
                    severity="hard",
                    overlap_with=existing.index,
                    overlap_fraction=round(frac, 3),
                )
                break
        if overlap_failure is not None:
            failures.append(overlap_failure)
        else:
            kept.append(candidate)

    total_coverage_fraction = 0.0
    if img_area > 0:
        total_coverage_fraction = sum(region.width * region.height for region in kept) / img_area
    if kept and total_coverage_fraction < _MIN_TOTAL_REGION_PAGE_FRACTION:
        failures.append(
            RegionFailure(
                region_index=-1,
                reason="insufficient_page_coverage",
                severity="hard",
                page_fraction=round(total_coverage_fraction, 3),
            )
        )

    return ValidationResult(
        valid=len(failures) == 0,
        kept=sorted(kept, key=lambda r: r.index),
        failures=failures,
    )


def validate_regions_for_write(
    regions: list[RegionResult],
    *,
    img_w: int,
    img_h: int,
) -> list[RegionResult]:
    """Validate regions and return only those that pass all checks.

    Convenience wrapper around validate_region_set that returns the kept list.
    When any failures exist the full set should be retried via repair; this wrapper
    is retained for call sites that only need the surviving region list.
    """
    return validate_region_set(regions, img_w=img_w, img_h=img_h).kept


# ---------------------------------------------------------------------------
# XMP region read-back
# ---------------------------------------------------------------------------


def _xmp_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".xmp")


def _has_xmp_regions(xmp_path: Path) -> bool:
    """Return True if the XMP sidecar already contains an mwg-rs:RegionList."""
    if not xmp_path.is_file():
        return False
    try:
        content = xmp_path.read_text(encoding="utf-8", errors="replace")
        return "mwg-rs:RegionList" in content
    except OSError:
        return False


def _read_regions_from_xmp(xmp_path: Path, img_w: int, img_h: int) -> list[RegionResult]:
    """Parse RegionResult list from an existing XMP sidecar.

    Reads mwg-rs stArea:x/y/w/h (normalised centre-point) and converts back
    to pixel top-left coordinates. Returns empty list on any parse failure.
    """
    import xml.etree.ElementTree as ET  # pylint: disable=import-outside-toplevel

    MWGRS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
    STAREA_NS = "http://ns.adobe.com/xap/1.0/sType/Area#"
    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

    try:
        tree = ET.parse(str(xmp_path))
    except ET.ParseError:
        return []

    results: list[RegionResult] = []
    for li in tree.iter(f"{{{RDF_NS}}}li"):
        cx_t = li.get(f"{{{STAREA_NS}}}x")
        cy_t = li.get(f"{{{STAREA_NS}}}y")
        nw_t = li.get(f"{{{STAREA_NS}}}w")
        nh_t = li.get(f"{{{STAREA_NS}}}h")
        name_t = li.get(f"{{{MWGRS_NS}}}Name") or ""
        if not all((cx_t, cy_t, nw_t, nh_t)):
            continue
        try:
            cx = float(cx_t)
            cy = float(cy_t)
            nw = float(nw_t)
            nh = float(nh_t)
        except (TypeError, ValueError):
            continue
        px = max(0, int(round((cx - nw / 2.000) * img_w)))
        py = max(0, int(round((cy - nh / 2.000) * img_h)))
        pw = max(1, int(round(nw * img_w)))
        ph = max(1, int(round(nh * img_h)))
        idx = len(results)
        try:
            idx = int(name_t.split("_")[-1]) - 1
        except (ValueError, IndexError):
            pass
        results.append(RegionResult(index=idx, x=px, y=py, width=pw, height=ph))
    return results


# ---------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    with Image.open(str(image_path)) as img:
        return img.size  # (width, height)


def _failed_regions_debug_path(image_path: str | Path, attempt_number: int | None = None) -> Path:
    path = Path(image_path)
    filename = f"{path.stem}.view-regions.failed-boxes.jpg"
    if attempt_number is not None:
        filename = f"{path.stem}.view-regions.failed-boxes.attempt-{attempt_number:02d}.jpg"
    return path.parent / "_debug" / filename


def _accepted_regions_debug_path(image_path: str | Path, attempt_number: int | None = None) -> Path:
    path = Path(image_path)
    filename = f"{path.stem}.view-regions.accepted-boxes.jpg"
    if attempt_number is not None:
        filename = f"{path.stem}.view-regions.accepted-boxes.attempt-{attempt_number:02d}.jpg"
    return path.parent / "_debug" / filename


def _failed_regions_debug_paths(image_path: str | Path) -> list[Path]:
    path = Path(image_path)
    debug_dir = path.parent / "_debug"
    return [*debug_dir.glob(f"{path.stem}.view-regions.failed-boxes*.jpg")]


def _accepted_regions_debug_paths(image_path: str | Path) -> list[Path]:
    path = Path(image_path)
    debug_dir = path.parent / "_debug"
    return [*debug_dir.glob(f"{path.stem}.view-regions.accepted-boxes*.jpg")]


def _clear_failed_regions_debug_image(image_path: str | Path) -> None:
    for debug_path in _failed_regions_debug_paths(image_path):
        try:
            if debug_path.is_file():
                debug_path.unlink()
        except OSError as exc:
            log.warning("Failed to remove failed-region debug image %s: %s", debug_path, exc)


def _clear_accepted_regions_debug_image(image_path: str | Path) -> None:
    for debug_path in _accepted_regions_debug_paths(image_path):
        try:
            if debug_path.is_file():
                debug_path.unlink()
        except OSError as exc:
            log.warning("Failed to remove accepted-region debug image %s: %s", debug_path, exc)


def _clear_regions_debug_images(image_path: str | Path) -> None:
    _clear_failed_regions_debug_image(image_path)
    _clear_accepted_regions_debug_image(image_path)


def _write_accepted_regions_debug_image(
    image_path: str | Path,
    regions: list[RegionResult],
    *,
    attempt_number: int | None = None,
) -> Path | None:
    from .ai_view_region_render import render_regions_debug  # pylint: disable=import-outside-toplevel

    overlay_regions: list[dict[str, object]] = []
    for region in sorted(regions, key=lambda item: item.index):
        label_parts = []
        caption_hint = str(region.caption_hint or "").strip()
        if caption_hint:
            label_parts.append(caption_hint)
        if region.person_names:
            label_parts.append(", ".join(str(name).strip() for name in region.person_names if str(name).strip()))
        overlay_regions.append(
            {
                "index": region.index,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "caption": " | ".join(label_parts),
            }
        )

    output_path = _accepted_regions_debug_path(image_path, attempt_number=attempt_number)
    try:
        render_regions_debug(image_path, overlay_regions, output_path)
    except Exception as exc:
        log.warning("Failed to write accepted-region debug image %s: %s", output_path, exc)
        return None
    return output_path


def _write_failed_regions_debug_image(
    image_path: str | Path,
    regions: list[RegionResult],
    failures: list[RegionFailure],
    *,
    attempt_number: int | None = None,
) -> Path | None:
    if not regions:
        return None

    from .ai_view_region_render import render_regions_debug  # pylint: disable=import-outside-toplevel

    page_level_notes: list[str] = []
    notes_by_index: dict[int, list[str]] = {}
    for failure in failures:
        if failure.reason == "overlap":
            other_label = ""
            if failure.overlap_with is not None:
                other_label = f" with #{failure.overlap_with + 1}"
            note = f"overlap{other_label}"
        elif failure.reason == "zero_area":
            note = "zero area"
        elif failure.reason == "full_page":
            note = "covers >=90% of page"
        elif failure.reason == "insufficient_page_coverage":
            pct = int(round((failure.page_fraction or 0) * 100))
            note = f"page coverage {pct}% < 50%"
        else:
            note = failure.reason.replace("_", " ")

        if failure.region_index >= 0:
            notes_by_index.setdefault(failure.region_index, []).append(note)
        else:
            page_level_notes.append(note)

    overlay_regions: list[dict[str, object]] = []
    sorted_regions = sorted(regions, key=lambda region: region.index)
    for idx, region in enumerate(sorted_regions):
        notes = list(notes_by_index.get(region.index, []))
        if idx == 0 and page_level_notes:
            notes = page_level_notes + notes
        overlay_regions.append(
            {
                "index": region.index,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "caption": "; ".join(notes),
            }
        )

    output_path = _failed_regions_debug_path(image_path, attempt_number=attempt_number)
    try:
        render_regions_debug(image_path, overlay_regions, output_path)
    except Exception as exc:
        log.warning("Failed to write failed-region debug image %s: %s", output_path, exc)
        return None
    return output_path


# ---------------------------------------------------------------------------
# Docling detection path
# ---------------------------------------------------------------------------


def _detect_regions_docling(
    path: Path,
    *,
    xmp_path: Path,
    model: str,
    base_url: str,
    timeout: float,
    img_w: int,
    img_h: int,
    force: bool,
    debug_recorder=None,
) -> list[RegionResult]:
    """Run the docling detection path: send a single prompt, parse <doctag> response.

    Writes view_regions pipeline step for no_regions and validation_failed outcomes
    to prevent infinite re-runs. Returns validated regions on success, [] otherwise.
    """
    from ._docling_parser import parse_doctag_response  # pylint: disable=import-outside-toplevel
    from .xmp_sidecar import write_pipeline_step  # pylint: disable=import-outside-toplevel

    if not force:
        # Check pipeline step to avoid re-running on known-bad outcomes
        from .xmp_sidecar import read_pipeline_step  # pylint: disable=import-outside-toplevel

        existing_step = read_pipeline_step(xmp_path, "view_regions") or {}
        existing_result = str(existing_step.get("result") or "").strip()
        if existing_result in ("no_regions", "validation_failed"):
            log.info(
                "Skipping docling detection for %s: pipeline step already recorded result=%r",
                path,
                existing_result,
            )
            return []

    image_url, _resized_w, _resized_h = _build_data_url_with_size(path, DEFAULT_MAX_IMAGE_EDGE)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Convert this page to docling."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    debug_metadata = {
        "engine": "docling",
        "model": model,
        "original_image_width": int(img_w),
        "original_image_height": int(img_h),
    }

    try:
        response = _lmstudio_post(f"{base_url}/chat/completions", payload, timeout)
    except Exception as exc:
        _emit_prompt_debug(
            debug_recorder,
            step="view_regions",
            engine="docling",
            model=model,
            prompt="Convert this page to docling.",
            source_path=path,
            prompt_source="runtime",
            response="",
            metadata={**debug_metadata, "error": str(exc)},
        )
        log.error("Docling LM Studio call failed for %s: %s", path, exc)
        return []

    choices = list(response.get("choices") or [])
    if not choices:
        log.error("Docling: LM Studio returned no choices for %s", path)
        return []

    content = choices[0].get("message", {}).get("content", "")
    response_text = json.dumps(response, ensure_ascii=False, sort_keys=True)

    _emit_prompt_debug(
        debug_recorder,
        step="view_regions",
        engine="docling",
        model=model,
        prompt="Convert this page to docling.",
        source_path=path,
        prompt_source="runtime",
        response=response_text,
        metadata=debug_metadata,
    )

    regions = parse_doctag_response(content, img_w, img_h)

    if not regions:
        write_pipeline_step(xmp_path, "view_regions", model=model, extra={"result": "no_regions"})
        log.info("Docling: no regions detected for %s", path)
        return []

    result = validate_region_set(regions, img_w=img_w, img_h=img_h)
    if result.failures:
        write_pipeline_step(xmp_path, "view_regions", model=model, extra={"result": "validation_failed"})
        reasons = ", ".join(f.reason for f in result.failures)
        log.error(
            "Docling: region validation failed for %s (%d failure(s): %s); use --force to retry",
            path,
            len(result.failures),
            reasons,
        )
        return []

    _write_accepted_regions_debug_image(path, result.kept)
    return result.kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_regions(
    image_path: str | Path,
    *,
    force: bool = False,
    model: str = "",
    base_url: str = "",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    album_context: str = "",
    page_caption: str = "",
    people_roster: dict[str, str] | None = None,
    prompt_debug: PromptDebugSession | None = None,
) -> list[RegionResult]:
    """Detect photo regions in a view JPG, writing results to the XMP sidecar.

    If force=False and the XMP sidecar already contains an mwg-rs:RegionList,
    returns the cached regions without calling the model.

    On model call failure, retries up to _MAX_RETRIES times with a stricter
    prompt. Returns [] and logs an error if all retries fail.
    """
    path = Path(image_path)
    xmp_path = _xmp_path_for(path)

    if not force and _has_xmp_regions(xmp_path):
        try:
            img_w, img_h = _image_dimensions(path)
            cached = _read_regions_from_xmp(xmp_path, img_w, img_h)
            if cached:
                _clear_regions_debug_images(path)
                _write_accepted_regions_debug_image(path, cached)
                return cached
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to read cached XMP regions for %s: %s", path, exc)

    resolved_models = [str(model).strip()] if str(model or "").strip() else default_view_region_models()
    if not resolved_models:
        fallback_model = str(default_view_region_model() or "").strip()
        if fallback_model:
            resolved_models = [fallback_model]
    resolved_url = normalize_lmstudio_base_url(base_url or default_lmstudio_base_url())

    img_w, img_h = _image_dimensions(path)

    # --- Docling code path ---
    docling_model = next((m for m in resolved_models if "docling" in m.lower()), None)
    if docling_model:
        return _detect_regions_docling(
            path,
            xmp_path=xmp_path,
            model=docling_model,
            base_url=resolved_url,
            timeout=timeout,
            img_w=img_w,
            img_h=img_h,
            force=force,
            debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
        )

    _clear_regions_debug_images(path)

    prior_regions: list[RegionResult] | None = None
    prior_failures: list[RegionFailure] | None = None
    last_exc: Exception | None = None
    errors: list[str] = []

    for resolved_model in resolved_models:
        prior_regions = None
        prior_failures = None
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                raw = _call_vision_model(
                    path,
                    model=resolved_model,
                    base_url=resolved_url,
                    timeout=timeout,
                    img_w=img_w,
                    img_h=img_h,
                    strict_prompt=(attempt > 0 and prior_failures is None),
                    prior_regions=prior_regions,
                    prior_failures=prior_failures,
                    album_context=album_context,
                    page_caption=page_caption,
                    people_roster=people_roster,
                    attempt_number=attempt + 1,
                    debug_recorder=(prompt_debug.record if prompt_debug is not None else None),
                )
            except Exception as exc:
                last_exc = exc
                log.warning("Region detection attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, exc)
                prior_regions = None
                prior_failures = None
                continue

            result = validate_region_set(raw, img_w=img_w, img_h=img_h)
            if not result.failures:
                _write_accepted_regions_debug_image(path, result.kept, attempt_number=attempt + 1)
                _write_accepted_regions_debug_image(path, result.kept)
                return result.kept

            # Validation failures - set up repair context for the next attempt.
            prior_regions = raw
            prior_failures = result.failures
            attempt_debug_path = _write_failed_regions_debug_image(
                path,
                raw,
                result.failures,
                attempt_number=attempt + 1,
            )
            reasons = ", ".join(f.reason for f in result.failures)
            log.warning(
                "Region validation attempt %d/%d: %d failure(s) (%s), retrying with repair prompt",
                attempt + 1,
                _MAX_RETRIES,
                len(result.failures),
                reasons,
            )
            if attempt_debug_path is not None:
                log.warning("Wrote failed-region debug image for attempt %d: %s", attempt + 1, attempt_debug_path)

        if prior_failures:
            errors.append(f"{resolved_model}: validation failed ({len(prior_failures)} failure(s))")
        elif last_exc is not None:
            errors.append(f"{resolved_model}: {last_exc}")

    if prior_failures:
        failed_debug_path = _write_failed_regions_debug_image(path, prior_regions or [], prior_failures)
        log.error(
            "All %d region detection attempts failed validation for %s (%d failure(s))",
            _MAX_RETRIES,
            path,
            len(prior_failures),
        )
        if failed_debug_path is not None:
            log.warning("Wrote failed-region debug image: %s", failed_debug_path)
    else:
        summary = "; ".join(errors) if errors else str(last_exc)
        log.error("All configured region detection models failed for %s: %s", path, summary)
    return []
