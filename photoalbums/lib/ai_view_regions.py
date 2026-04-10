"""Detect individual photo regions within stitched view JPGs using a vision model.

The model (LM Studio / OpenAI-compatible) returns pixel bounding boxes for each
photo; results are written directly into the view image's XMP sidecar as an
MWG-RS RegionList. The XMP is both the cache and the ground truth.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .ai_model_settings import default_lmstudio_base_url, default_view_region_model
from ._caption_lmstudio import normalize_lmstudio_base_url

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_IMAGE_EDGE = 2048
_MAX_RETRIES = 3


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


@dataclass
class RegionWithCaption:
    region: RegionResult
    caption: str
    caption_ambiguous: bool = False


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def pixel_to_mwgrs(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
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
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                            "confidence": {"type": "number"},
                            "caption_hint": {"type": "string"},
                        },
                        "required": ["index", "x", "y", "width", "height", "confidence", "caption_hint"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["regions"],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a photo-album digitization assistant. "
    "You will be shown a stitched view image containing one or more photographs from a physical photo album page. "
    "The photos may be packed edge-to-edge with NO border or background between them. "
    "Identify each distinct photograph by looking for contextual seams: changes in scene, perspective, "
    "lighting, photographic style, or subject matter — NOT just visible gaps or borders. "
    "Return a JSON object with a 'regions' array. "
    "Each entry gives the bounding box of one photograph as NORMALISED coordinates (top-left origin, values from 0.000 to 1.000) "
    "where 1.000 is the full image width or height), a confidence score between 0 and 1, and a brief caption_hint."
)

_USER_PROMPT = (
    "Examine this album page image carefully. "
    "Identify every individual photograph visible. "
    "Return their bounding boxes as normalised coordinates (0.000 to 1.000) in a JSON regions array. "
    "x=0.000 is the left edge, x=1.000 is the right edge, y=0.000 is the top, y=1.000 is the bottom."
)

_USER_PROMPT_STRICT = (
    "Examine this album page image carefully. "
    "Return ONLY a valid JSON object with a 'regions' array. "
    "Each region MUST have: index (integer starting at 0), "
    "x, y, width, height (floats from 0.000 to 1.000, top-left origin, normalised to image size), "
    "confidence (float 0-1), caption_hint (string). "
    "No other text."
)


def _build_data_url(image_path: Path, max_edge: int) -> str:
    """Return a base64 data URL for the image, downscaled to max_edge on longest side."""
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from .image_limits import allow_large_pillow_images  # pylint: disable=import-outside-toplevel

    allow_large_pillow_images(Image)
    image = Image.open(str(image_path)).convert("RGB")
    try:
        w, h = image.size
        longest = max(w, h)
        if longest > max_edge > 0:
            scale = max_edge / longest
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    finally:
        image.close()


def _lmstudio_post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
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
) -> list[RegionResult]:
    image_url = _build_data_url(image_path, DEFAULT_MAX_IMAGE_EDGE)
    user_text = _USER_PROMPT_STRICT if strict_prompt else _USER_PROMPT
    if img_w > 0 and img_h > 0:
        user_text = user_text + f" The full image is {img_w}×{img_h} pixels."
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
    response = _lmstudio_post(f"{base_url}/chat/completions", payload, timeout)
    choices = list(response.get("choices") or [])
    if not choices:
        raise RuntimeError("LM Studio returned no choices in response")
    content = choices[0].get("message", {}).get("content", "")
    # Model returns normalised 0–1 coords; convert to pixel using original image dims.
    return _parse_region_response(content, img_w=img_w, img_h=img_h)


def _parse_region_response(
    content: str,
    *,
    img_w: int = 0,
    img_h: int = 0,
) -> list[RegionResult]:
    """Parse region response from the model.

    The model returns normalised 0–1 coords (top-left origin).
    If img_w/img_h are provided, coords are converted to pixel space.
    """
    text = str(content or "").strip()
    if not text:
        raise RuntimeError("Empty response from vision model")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            try:
                payload = json.loads(text[start:])
            except json.JSONDecodeError:
                raise RuntimeError(f"Could not parse JSON from model response: {text[:200]!r}")
        else:
            raise RuntimeError(f"No JSON object in model response: {text[:200]!r}")
    raw_regions = list(payload.get("regions") or [])
    results: list[RegionResult] = []
    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        try:
            nx = max(0.000, min(1.000, float(item["x"])))
            ny = max(0.000, min(1.000, float(item["y"])))
            nw = max(0.000, min(1.000, float(item["width"])))
            nh = max(0.000, min(1.000, float(item["height"])))
            if nw <= 0 or nh <= 0:
                continue
            if img_w > 0 and img_h > 0:
                px = max(0, int(round(nx * img_w)))
                py = max(0, int(round(ny * img_h)))
                pw = max(1, int(round(nw * img_w)))
                ph = max(1, int(round(nh * img_h)))
            else:
                # No image dims supplied — store as-is (0–1 range as pixel ints, test use only)
                px, py, pw, ph = int(nx), int(ny), max(1, int(nw)), max(1, int(nh))
            results.append(RegionResult(
                index=int(item.get("index") or len(results)),
                x=px, y=py, width=pw, height=ph,
                confidence=max(0.0, min(1.0, float(item.get("confidence") or 1.0))),
                caption_hint=str(item.get("caption_hint") or "").strip(),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping malformed region entry %r: %s", item, exc)
    return results


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
                return cached
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to read cached XMP regions for %s: %s", path, exc)

    resolved_model = model or default_view_region_model()
    resolved_url = normalize_lmstudio_base_url(base_url or default_lmstudio_base_url())

    img_w, img_h = _image_dimensions(path)

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            regions = _call_vision_model(
                path,
                model=resolved_model,
                base_url=resolved_url,
                timeout=timeout,
                img_w=img_w,
                img_h=img_h,
                strict_prompt=(attempt > 0),
            )
            return regions
        except Exception as exc:
            last_exc = exc
            log.warning("Region detection attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, exc)

    log.error("All %d region detection attempts failed for %s: %s", _MAX_RETRIES, path, last_exc)
    return []
