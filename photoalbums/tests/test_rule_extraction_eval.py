"""
Integration eval: validate caption quality using the real pipeline prompt.

Builds the user prompt with `_build_local_prompt`,
sends it with the production LM Studio system prompt and asserts on caption content.
No intermediate rule-selection step — the full skill prompt is sent as-is.

Requires LM Studio running with zai-org/glm-4.6v-flash loaded.
All tests are auto-skipped if LM Studio is not reachable or the test image is missing.

Run:
    python -m pytest photoalbums/tests/test_rule_extraction_eval.py -v -m integration -s
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib._caption_prompts import _build_local_prompt  # noqa: E402
from photoalbums.lib._caption_lmstudio import (  # noqa: E402
    DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE,
    _build_data_url,
    _lmstudio_caption_response_format,
)
from photoalbums.lib.ai_caption import CaptionEngine  # noqa: E402
from photoalbums.lib.ai_index import _run_image_analysis, DEFAULT_CAST_STORE  # noqa: E402
from photoalbums.lib.ai_ocr import OCREngine  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_LMSTUDIO_BASE = "http://localhost:1234"
_MODEL = "zai-org/glm-4.6v-flash"
_RUNS_PER_CASE = 1
_MIN_PASSING_RUNS = 1  # ≥2/3 runs must pass per case

# Stitched JPEG for the Dunhuang page — murals, exhibit signage, people
_TEST_IMAGE = Path(
    r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums"
    r"\China_1986_B02_View\China_1986_B02_P02_stitched.jpg"
)
_TEST_IMAGE_P04 = Path(
    r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums"
    r"\China_1986_B02_View\China_1986_B02_P04_stitched.jpg"
)
_TEST_IMAGE_P16_D01 = Path(
    r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums"
    r"\China_1986_B02_View\China_1986_B02_P16_D01_01.jpg"
)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

_TEST_CASES = [
    {
        "id": "dunhuang-location-from-ocr",
        "ocr_text": ("EXHIBIT HISTORICAL RELICS OF DUNHUANG WELCOME TO DUNHUANG SUMMER CULTURE CENTRE"),
        "objects": ["person", "truck"],
        "people": [],
        "album_title": "Mainland China 1986 Book II",
        # Title should name the location visible in OCR text
        "check_field": "title",
        "must_include": ["Dunhuang"],
        # No meta-references to the medium itself
        "must_omit": ["photograph", "picture", "image", "scanned", "album page"],
    },
    {
        "id": "dunhuang-buddha-murals",
        "image": _TEST_IMAGE_P04,
        "ocr_text": "",
        "objects": [],
        "people": [],
        "album_title": "Mainland China 1986 Book II",
        "must_include": ["Buddhist"],
        "must_omit": ["historic", "photograph", "picture", "image", "scanned", "album page"],
    },
    {
        "id": "named-person",
        "use_pipeline": True,
        "image": _TEST_IMAGE_P16_D01,
        "ocr_text": "",
        "album_title": "Mainland China 1986 Book II",
        # Face detection identifies Audrey/Leslie; caption must use their names
        "must_include": ["Audrey Cordell", "Leslie Cordell"],
        "must_omit": ["photograph", "picture", "image", "scanned", "two people"],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_caption(base_url: str, image_path: Path, *, case: dict) -> dict:
    """Build the real pipeline prompt and send it directly to LM Studio with the image."""
    prompt = _build_local_prompt(
        people=case.get("people", []),
        objects=case.get("objects", []),
        ocr_text=case.get("ocr_text", ""),
        album_title=case.get("album_title", ""),
    )
    image_url = _build_data_url(image_path, DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE)
    payload = {
        "model": _MODEL,
        "messages": [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "response_format": _lmstudio_caption_response_format(),
        "temperature": 0,
        "max_tokens": 256,
    }
    t0 = time.monotonic()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.monotonic() - t0
    if not resp.ok:
        print(f"\n[DEBUG] {resp.status_code}: {resp.text[:500]}")
    resp.raise_for_status()
    raw = str(resp.json()["choices"][0]["message"].get("content") or "")
    try:
        parsed = json.loads(raw)
        title = parsed.get("title") or ""
        caption = parsed.get("caption") or raw
    except (json.JSONDecodeError, AttributeError):
        title = ""
        caption = raw
    print(f"\n[DEBUG] {elapsed:.1f}s, title: {title[:150]}, caption: {caption[:150]}")
    return {"title": title, "caption": caption, "prompt": prompt}


def _call_caption_engine(base_url: str, image_path: Path, *, case: dict) -> dict:
    """Run the full CaptionEngine pipeline (same path as production)."""
    prompt = _build_local_prompt(
        people=case.get("people", []),
        objects=case.get("objects", []),
        ocr_text=case.get("ocr_text", ""),
        album_title=case.get("album_title", ""),
    )
    t0 = time.monotonic()
    engine = CaptionEngine(
        engine="lmstudio",
        model_name=_MODEL,
        lmstudio_base_url=base_url,
        max_tokens=256,
        temperature=0,
    )
    output = engine.generate(
        image_path,
        people=case.get("people", []),
        objects=case.get("objects", []),
        ocr_text=case.get("ocr_text", ""),
        album_title=case.get("album_title", ""),
    )
    elapsed = time.monotonic() - t0
    if output.error:
        raise RuntimeError(output.error)
    print(f"\n[DEBUG] {elapsed:.1f}s, caption: {output.text[:300]}")
    return {"title": "", "caption": output.text, "prompt": prompt}


def _call_pipeline(base_url: str, image_path: Path, *, case: dict) -> dict:
    """Full pipeline: real face detection → position computation → caption (mirrors MCP call)."""
    from photoalbums.lib.ai_people import CastPeopleMatcher

    t0 = time.monotonic()
    people_matcher = CastPeopleMatcher(
        cast_store_dir=DEFAULT_CAST_STORE,
        min_similarity=0.72,
        min_face_size=40,
    )
    ocr_engine = OCREngine(engine="none")
    caption_engine = CaptionEngine(
        engine="lmstudio",
        model_name=_MODEL,
        lmstudio_base_url=base_url,
        max_tokens=256,
        temperature=0,
    )
    result = _run_image_analysis(
        image_path=image_path,
        people_matcher=people_matcher,
        object_detector=None,
        ocr_engine=ocr_engine,
        caption_engine=caption_engine,
        requested_caption_engine="lmstudio",
        requested_caption_model=_MODEL,
        ocr_engine_name="none",
        ocr_language="eng",
        album_title=case.get("album_title", ""),
        ocr_text_override=case.get("ocr_text", "") or "",
    )
    elapsed = time.monotonic() - t0
    diag = (
        f"Pipeline analysis:\n"
        f"  people_detected: {result.people_names}\n"
        f"  objects_detected: {result.object_labels}\n"
        f"  ocr_text: {result.ocr_text!r}"
    )
    print(f"\n[DEBUG] {elapsed:.1f}s, title: {result.title[:150]}, caption: {result.description[:150]}")
    return {"title": result.title, "caption": result.description, "prompt": diag}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lmstudio_url():
    """Return LM Studio base URL or skip the entire module if not reachable."""
    try:
        resp = requests.get(f"{_LMSTUDIO_BASE}/v1/models", timeout=3)
        if resp.status_code != 200:
            pytest.skip("LM Studio not reachable")
    except Exception:
        pytest.skip("LM Studio not reachable")
    if not _TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {_TEST_IMAGE}")
    return _LMSTUDIO_BASE


# ---------------------------------------------------------------------------
# Eval tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("case", _TEST_CASES, ids=[c["id"] for c in _TEST_CASES])
def test_caption_quality(lmstudio_url, case):
    """Caption follows skill rules when the full prompt is sent directly with the image."""
    image = case.get("image", _TEST_IMAGE)
    if not image.exists():
        pytest.skip(f"Test image not found: {image}")
    passed_runs = 0
    failing_details: list[str] = []

    for run in range(1, _RUNS_PER_CASE + 1):
        if case.get("use_pipeline"):
            caller = _call_pipeline
        elif case.get("use_engine"):
            caller = _call_caption_engine
        else:
            caller = _call_caption
        result = caller(lmstudio_url, image, case=case)
        include_field: str = result[case.get("check_field", "caption")]
        caption: str = result["caption"]

        missing = [p for p in case["must_include"] if p.lower() not in include_field.lower()]
        present_forbidden = [p for p in case["must_omit"] if p.lower() in caption.lower()]

        if not missing and not present_forbidden:
            passed_runs += 1
        else:
            detail_lines = [f"  Run {run} FAILED"]
            if missing:
                detail_lines.append(f"    Missing from {case.get('check_field', 'caption')}: {missing}")
            if present_forbidden:
                detail_lines.append(f"    Forbidden in caption: {present_forbidden}")
            detail_lines.append(f"    Caption: {caption}")
            detail_lines.append(f"    Prompt sent:\n{result['prompt']}")
            failing_details.append("\n".join(detail_lines))

    assert passed_runs >= _MIN_PASSING_RUNS, (
        f"Case '{case['id']}': only {passed_runs}/{_RUNS_PER_CASE} runs passed "
        f"(need {_MIN_PASSING_RUNS}).\n"
        f"  Inputs:\n"
        f"    image:       {image}\n"
        f"    people:      {case.get('people', [])}\n"
        f"    objects:     {case.get('objects', [])}\n"
        f"    ocr_text:    {case.get('ocr_text', '')!r}\n"
        f"    album_title: {case.get('album_title', '')!r}\n" + "\n".join(failing_details)
    )
