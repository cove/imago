from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import requests

from photoalbums.common import get_photo_albums_dir
from photoalbums.lib._caption_lmstudio import (
    _build_data_url,
    _extract_structured_json_payload,
    normalize_lmstudio_base_url,
)
from photoalbums.lib.ai_model_settings import default_caption_model, default_lmstudio_base_url
from photoalbums.lib.ai_view_regions import render_regions_overlay
from photoalbums.lib.xmp_sidecar import read_region_list

TARGETS_PATH = Path(__file__).resolve().parents[1] / "evals" / "context_reasoning_poc_targets.json"
RUN_ENV = "PHOTOALBUMS_RUN_CONTEXT_POC_EVALS"
BASE_URL_ENV = "PHOTOALBUMS_CONTEXT_POC_BASE_URL"
MODEL_ENV = "PHOTOALBUMS_CONTEXT_POC_MODEL"
IPTC_NS = "{http://iptc.org/std/Iptc4xmpExt/2008-02-29/}"
RDF_NS = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def _load_targets() -> dict:
    return json.loads(TARGETS_PATH.read_text(encoding="utf-8"))


def test_poc_targets_capture_current_context_reasoning_goals():
    payload = _load_targets()
    cases = {case["id"]: case for case in payload["cases"]}

    family = cases["family_p04_context"]
    expected_people = {
        (candidate["person_name"], candidate["region_index"])
        for candidate in family["expected"]["identity_candidates"]
    }
    assert ("Gloria Jean Dilbeck (nee Laymon)", 0) in expected_people
    assert ("Gloria Jean Dilbeck (nee Laymon)", 4) in expected_people
    assert ("Gladys", 1) in expected_people
    assert ("Jim Peterson", 5) in expected_people
    conflict_checks = family["expected"]["conflict_checks"]
    assert conflict_checks[0]["region_index"] == 3
    assert "Rod, Diana, and Frank" in conflict_checks[0]["conflict"]
    assert family["expected"]["location"]["address"] == "2240 Lorain Rd, San Marino, CA 91108"
    continuity = family["expected"]["identity_continuity"]
    assert continuity["candidate_caption"] == "Jim-Miriam-Donald"
    assert continuity["target_person"] == "Gloria Jean Dilbeck (nee Laymon)"
    assert continuity["expected_policy"] == "review-only-not-auto-write"

    china = cases["china_p16_huaqing"]
    expected_location = china["expected"]["location"]
    assert expected_location["caption_evidence"] == "HUA QING HOT SPRINGS"
    assert "38 Huaqing Rd" in expected_location["location_query"]
    assert china["context_images"] == [
        "MainlandChina_1986_B02_Archive/MainlandChina_1986_B02_P16_S01.tif",
        "MainlandChina_1986_B02_Archive/MainlandChina_1986_B02_P16_S02.tif",
    ]


def _poc_schema() -> dict:
    region = {
        "type": "object",
        "properties": {
            "region_number": {"type": "integer"},
            "caption": {"type": "string"},
            "people_names": {"type": "array", "items": {"type": "string"}},
            "date_text": {"type": "string"},
            "location_text": {"type": "string"},
        },
        "required": ["region_number", "caption", "people_names", "date_text", "location_text"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "context_caption_read_poc",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"regions": {"type": "array", "items": region}},
                "required": ["regions"],
                "additionalProperties": False,
            },
        },
    }


def _location_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "context_location_poc",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "caption_text": {"type": "string"},
                    "location_name": {"type": "string"},
                    "location_query": {"type": "string"},
                    "evidence": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["caption_text", "location_name", "location_query", "evidence", "supported"],
                "additionalProperties": False,
            },
        },
    }


def _continuity_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "identity_continuity_poc",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "target_person": {"type": "string"},
                    "candidate_description": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "caption_conflict": {"type": "boolean"},
                    "write_action": {"type": "string"},
                    "likely_caption_name": {"type": "string"},
                    "confidence": {"type": "string"},
                    "evidence": {"type": "string"},
                    "conflicts": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "target_person",
                    "candidate_description",
                    "supported",
                    "caption_conflict",
                    "write_action",
                    "likely_caption_name",
                    "confidence",
                    "evidence",
                    "conflicts",
                ],
                "additionalProperties": False,
            },
        },
    }


def _location_scope_schema() -> dict:
    region = {
        "type": "object",
        "properties": {
            "region_number": {"type": "integer"},
            "caption_text": {"type": "string"},
            "location_text": {"type": "string"},
            "date_text": {"type": "string"},
            "applies_to_region": {"type": "boolean"},
            "write_action": {"type": "string"},
            "uncertainty_reason": {"type": "string"},
        },
        "required": [
            "region_number",
            "caption_text",
            "location_text",
            "date_text",
            "applies_to_region",
            "write_action",
            "uncertainty_reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "multi_location_scope_poc",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"regions": {"type": "array", "items": region}},
                "required": ["regions"],
                "additionalProperties": False,
            },
        },
    }


def _layout_schema() -> dict:
    bad_region = {
        "type": "object",
        "properties": {
            "region_number": {"type": "integer"},
            "problem": {"type": "string"},
            "write_action": {"type": "string"},
        },
        "required": ["region_number", "problem", "write_action"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "layout_overlap_poc",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "layout_usable": {"type": "boolean"},
                    "bad_regions": {"type": "array", "items": bad_region},
                    "evidence": {"type": "string"},
                },
                "required": ["layout_usable", "bad_regions", "evidence"],
                "additionalProperties": False,
            },
        },
    }


def _caption_read_prompt(case: dict) -> str:
    return "\n".join(
        [
            "For each numbered photo region, read the typed caption directly under or beside that numbered region.",
            "Split hyphen-separated person names into people_names.",
            "Region numbers are the # labels in the overlay.",
            f"Album title: {case['album_title']}",
        ]
    )


def _location_prompt(case: dict) -> str:
    return "\n".join(
        [
            "Read the typed caption line visible on this numbered album-page overlay.",
            "Use the caption to identify the real place.",
            "The text may say HUA QING or HUAQING.",
            "Return a precise location only if the caption supports it.",
            "Include city, province or state, and country when the album title supplies that context.",
            f"Album title: {case['album_title']}",
        ]
    )


def _continuity_prompt(case: dict) -> str:
    target = case["expected"]["identity_continuity"]
    return "\n".join(
        [
            "Evaluate one possible identity using visual continuity across two crops.",
            "This is review-only evidence. Be conservative.",
            "Do not identify a person when the face is blurry, hidden, or turned away unless the clothing, hair, body position, companions, and caption context strongly agree.",
            f"Anchor crop caption: {target['anchor_caption']}",
            f"Candidate crop caption: {target['candidate_caption']}",
            f"Target person to test in candidate crop: {target['target_person']}",
            f"Candidate subject: {target['candidate_description']}",
            "If the candidate crop caption points to a different name than the target person, set caption_conflict=true.",
            "Set write_action to review whenever caption_conflict=true, even if visual continuity looks strong.",
            "Only use write_action=auto_write when there is no caption conflict and visual evidence is strong.",
            "If a caption name is a better explanation than the target person, put that name in likely_caption_name.",
        ]
    )


def _location_scope_prompt() -> str:
    return "\n".join(
        [
            "For each numbered photo region, read the caption text that applies to it.",
            "Extract only the location and date text that apply to that numbered region.",
            "This page has multiple captions and locations; do not smear one location across the whole page.",
            "Use write_action=auto_write only when location scope is clear, otherwise review.",
        ]
    )


def _layout_prompt() -> str:
    return "\n".join(
        [
            "Inspect the numbered overlay boxes.",
            "Decide if the photo region layout is usable for per-photo metadata.",
            "Flag overlapping, merged, or misaligned boxes as bad_regions.",
            "If any region box clearly overlaps another photo or caption area, set layout_usable=false and write_action=review for those regions.",
        ]
    )


def _append_image(content: list[dict], label: str, image_path: Path, max_image_edge: int) -> None:
    if not image_path.exists():
        pytest.skip(f"POC image missing: {image_path}")
    content.append({"type": "text", "text": label})
    content.append({"type": "image_url", "image_url": {"url": _build_data_url(image_path, max_image_edge)}})


def _append_page_overlay(content: list[dict], page_path: Path) -> None:
    from PIL import Image

    with Image.open(page_path) as image:
        regions = read_region_list(page_path.with_suffix(".xmp"), *image.size)
    if not regions:
        return
    with tempfile.TemporaryDirectory() as tmp:
        overlay_path = Path(tmp) / f"{page_path.stem}_overlay.jpg"
        render_regions_overlay(page_path, regions, overlay_path)
        _append_image(content, f"numbered page overlay: {page_path.name}", overlay_path, 0)


def _image_content(album_root: Path, case: dict, *, prompt: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    page_path = album_root / case["page_image"]
    _append_page_overlay(content, page_path)
    return content


def _continuity_image_content(album_root: Path, case: dict, *, prompt: str) -> list[dict]:
    target = case["expected"]["identity_continuity"]
    page_path = album_root / case["page_image"]
    content: list[dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": f"XMP face region context: {json.dumps(_face_region_context(page_path))}"})
    _append_image(content, f"anchor crop: {target['anchor_caption']}", album_root / target["anchor_image"], 2400)
    _append_image(
        content,
        f"candidate crop: {target['candidate_caption']}",
        album_root / target["candidate_image"],
        2400,
    )
    return content


def _face_region_context(page_path: Path) -> dict:
    from PIL import Image

    with Image.open(page_path) as image:
        page_width, page_height = image.size
    photo_regions = read_region_list(page_path.with_suffix(".xmp"), page_width, page_height)
    by_region: dict[int, list[dict]] = {int(region["index"]) + 1: [] for region in photo_regions}
    try:
        root = ET.parse(page_path.with_suffix(".xmp")).getroot()
    except ET.ParseError:
        return {"photo_regions": by_region}

    for item in root.findall(f".//{IPTC_NS}ImageRegion/{RDF_NS}Bag/{RDF_NS}li"):
        name = "; ".join(
            str(row.text or "").strip()
            for row in item.findall(f".//{IPTC_NS}Name/{RDF_NS}Alt/{RDF_NS}li")
            if str(row.text or "").strip()
        )
        boundary = item.find(f"{IPTC_NS}RegionBoundary")
        if not name or boundary is None:
            continue
        rb_x = float(boundary.findtext(f"{IPTC_NS}rbX", default="0") or 0)
        rb_y = float(boundary.findtext(f"{IPTC_NS}rbY", default="0") or 0)
        rb_w = float(boundary.findtext(f"{IPTC_NS}rbW", default="0") or 0)
        rb_h = float(boundary.findtext(f"{IPTC_NS}rbH", default="0") or 0)
        face_center_x = (rb_x + rb_w / 2) * page_width
        face_center_y = (rb_y + rb_h / 2) * page_height
        for region in photo_regions:
            if (
                region["x"] <= face_center_x <= region["x"] + region["width"]
                and region["y"] <= face_center_y <= region["y"] + region["height"]
            ):
                by_region.setdefault(int(region["index"]) + 1, []).append(
                    {
                        "name": name,
                        "box": {"x": rb_x, "y": rb_y, "w": rb_w, "h": rb_h},
                    }
                )
                break
    return {"photo_regions": by_region}


def _proposed_case(case_id: str) -> dict:
    return next(case for case in _load_targets()["proposed_next_tests"] if case["id"] == case_id)


@pytest.fixture(scope="module")
def context_poc_client() -> tuple[str, str]:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"Set {RUN_ENV}=1 to run the opt-in context reasoning POC evals.")

    base_url = normalize_lmstudio_base_url(os.environ.get(BASE_URL_ENV, ""), default=default_lmstudio_base_url())
    model = os.environ.get(MODEL_ENV, "").strip() or default_caption_model()
    if not model:
        pytest.skip(f"No model configured. Set {MODEL_ENV}.")

    try:
        response = requests.get(f"{base_url}/models", timeout=3)
    except Exception as exc:
        pytest.skip(f"LM Studio not reachable at {base_url}: {exc}")
    if response.status_code != 200:
        pytest.skip(f"LM Studio not reachable at {base_url}: HTTP {response.status_code}")
    return base_url, model


def _call_context_poc(case: dict, *, base_url: str, model: str, album_root: Path) -> dict:
    if case["id"] == "family_p04_context":
        response_format = _poc_schema()
        prompt = _caption_read_prompt(case)
        is_valid = _is_caption_read_payload
    else:
        response_format = _location_schema()
        prompt = _location_prompt(case)
        is_valid = _is_location_payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only JSON matching the schema. Inspect visible typed captions carefully. "
                    "Do not invent names, dates, or locations that are not supported by the visible caption."
                ),
            },
            {"role": "user", "content": _image_content(album_root, case, prompt=prompt)},
        ],
        "response_format": response_format,
        "temperature": 0,
        "max_tokens": 1200,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=300)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content")
    text = str(content or "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _extract_structured_json_payload(text, is_valid=is_valid)
    if not is_valid(parsed):
        parsed = _extract_structured_json_payload(text, is_valid=is_valid)
    if not isinstance(parsed, dict):
        raise AssertionError(f"LM Studio returned non-dict POC payload: {content!r}")
    return parsed


def _regions_by_number(result: dict) -> dict[int, dict]:
    return {
        int(region.get("region_number") or 0): region
        for region in result.get("regions", [])
        if int(region.get("region_number") or 0) > 0
    }


def _is_caption_read_payload(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("regions"), list)


def _is_location_payload(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("supported"), bool)


def _is_continuity_payload(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("supported"), bool)


def _is_layout_payload(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("layout_usable"), bool)


def _parse_poc_payload(content: object, is_valid) -> dict:
    text = str(content or "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _extract_structured_json_payload(text, is_valid=is_valid)
    if not is_valid(parsed):
        parsed = _extract_structured_json_payload(text, is_valid=is_valid)
    if not isinstance(parsed, dict):
        raise AssertionError(f"LM Studio returned non-dict POC payload: {content!r}")
    return parsed


def _overlapping_region_numbers(page_path: Path, *, min_overlap_fraction: float = 0.02) -> set[int]:
    from PIL import Image

    with Image.open(page_path) as image:
        regions = read_region_list(page_path.with_suffix(".xmp"), *image.size)
    overlapping: set[int] = set()
    for index, left in enumerate(regions):
        left_area = float(left["width"] * left["height"])
        for right in regions[index + 1 :]:
            overlap_width = max(
                0,
                min(left["x"] + left["width"], right["x"] + right["width"]) - max(left["x"], right["x"]),
            )
            overlap_height = max(
                0,
                min(left["y"] + left["height"], right["y"] + right["height"]) - max(left["y"], right["y"]),
            )
            overlap_area = float(overlap_width * overlap_height)
            if not overlap_area:
                continue
            right_area = float(right["width"] * right["height"])
            if overlap_area / min(left_area, right_area) >= min_overlap_fraction:
                overlapping.add(int(left["index"]) + 1)
                overlapping.add(int(right["index"]) + 1)
    return overlapping


def _call_overlay_poc(
    *,
    base_url: str,
    model: str,
    album_root: Path,
    case: dict,
    prompt: str,
    response_format: dict,
    is_valid,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only JSON matching the schema. Be conservative and surface ambiguity.",
            },
            {"role": "user", "content": _image_content(album_root, case, prompt=prompt)},
        ],
        "response_format": response_format,
        "temperature": 0,
        "max_tokens": 2000,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=300)
    response.raise_for_status()
    return _parse_poc_payload(response.json()["choices"][0]["message"].get("content"), is_valid)


@pytest.mark.integration
@pytest.mark.parametrize("case", _load_targets()["cases"], ids=lambda case: case["id"])
def test_context_reasoning_poc_eval(context_poc_client, case):
    base_url, model = context_poc_client
    result = _call_context_poc(case, base_url=base_url, model=model, album_root=get_photo_albums_dir())

    if case["id"] == "family_p04_context":
        regions = _regions_by_number(result)
        assert "Gloria" in regions[1]["people_names"], result
        assert "Gladys" in regions[1]["people_names"], result
        assert "Gloria" in regions[5]["people_names"], result
        assert "Jim" in regions[5]["people_names"], result
        assert "Gloria" not in regions[4]["people_names"], result
        assert "San Marino" in regions[1]["location_text"], result
    else:
        locations = " ".join(str(value) for value in result.values())
        assert result.get("supported") is True, result
        assert "Huaqing" in locations or "Hua Qing" in locations, result
        assert "China" in locations or "Xi" in locations, result


@pytest.mark.integration
def test_identity_continuity_poc_eval(context_poc_client):
    base_url, model = context_poc_client
    case = next(case for case in _load_targets()["cases"] if case["id"] == "family_p04_context")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only JSON matching the schema. Make conservative identity-continuity judgments. "
                    "Captions are evidence and conflicts must be surfaced."
                ),
            },
            {
                "role": "user",
                "content": _continuity_image_content(
                    get_photo_albums_dir(),
                    case,
                    prompt=_continuity_prompt(case),
                ),
            },
        ],
        "response_format": _continuity_schema(),
        "temperature": 0,
        "max_tokens": 1000,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=300)
    response.raise_for_status()
    result = _parse_poc_payload(
        response.json()["choices"][0]["message"].get("content"),
        _is_continuity_payload,
    )

    assert result["caption_conflict"] is True, result
    assert result["write_action"] == "review", result
    combined = " ".join(str(value) for value in result.values())
    assert "Miriam" in combined or "caption" in combined, result


@pytest.mark.integration
def test_multi_location_scope_poc_eval(context_poc_client):
    base_url, model = context_poc_client
    case = _proposed_case("multi_location_scope_family_b05_p23")
    result = _call_overlay_poc(
        base_url=base_url,
        model=model,
        album_root=get_photo_albums_dir(),
        case=case,
        prompt=_location_scope_prompt(),
        response_format=_location_scope_schema(),
        is_valid=_is_caption_read_payload,
    )
    regions = _regions_by_number(result)

    assert "SAN MARINO" in regions[1]["location_text"].upper(), result
    assert "SAN MARINO" in regions[2]["location_text"].upper(), result
    assert "BERKELEY" in regions[6]["location_text"].upper(), result
    assert "ALTADENA" in regions[4]["location_text"].upper(), result
    assert not regions[3]["location_text"], result


@pytest.mark.integration
def test_layout_overlap_poc_eval(context_poc_client):
    base_url, model = context_poc_client
    case = _proposed_case("layout_overlap_rejection_china_b01_p13")
    result = _call_overlay_poc(
        base_url=base_url,
        model=model,
        album_root=get_photo_albums_dir(),
        case=case,
        prompt=_layout_prompt(),
        response_format=_layout_schema(),
        is_valid=_is_layout_payload,
    )
    bad_regions = {int(row["region_number"]) for row in result["bad_regions"]}
    deterministic_bad_regions = _overlapping_region_numbers(get_photo_albums_dir() / case["page_image"])

    assert result["layout_usable"] is False, result
    assert {3, 6, 7}.issubset(deterministic_bad_regions)
    assert {3, 7}.issubset(bad_regions), result
    assert all(row["write_action"] == "review" for row in result["bad_regions"]), result
