"""One-off, lightweight validator for crop captions and shown locations.

For each _Photos crop it shows the model the full album page (context) AND the
crop (target) in a single pass, asks for the caption written on the page and the
location the photo shows, then writes both to the crop's XMP sidecar.

This is intentionally standalone: it does NOT use the `verify-crops` pipeline step
(`ai_verify_crops`) or its verdict/retry/routing machinery. It only reuses generic
building blocks -- the LM Studio client, the Nominatim geocoder, and the canonical
XMP sidecar writer.

- Caption (page+crop): transcribed verbatim from the page, written to dc:description
  (empty when there is no caption near the photo).
- Location (page+crop): a place name, geocoded via Nominatim, written to
  Iptc4xmpExt:LocationShown (+ imago location). Left untouched when the model can't
  identify a location or the geocode misses -- never wiped.

Defaults to a dry run (preview only). Pass --run to write the sidecars.

Usage:
    uv run python -m photoalbums.scripts.recheck_crop_caption_location [--album X] [--page NN] [--crop D05-00] [--run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.lib._caption_lmstudio import (
    DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_TIMEOUT_SECONDS,
    _build_data_url,
    _extract_structured_json_payload,
    _format_lmstudio_debug_response,
    _lmstudio_request_json,
    _select_lmstudio_model,
    normalize_lmstudio_base_url,
)
from photoalbums.lib.ai_geocode import NominatimGeocoder
from photoalbums.lib.ai_index_runner import _write_sidecar_and_record
from photoalbums.lib.ai_location import _canonical_location_shown_entry, _resolve_geocoded_location_payload
from photoalbums.lib.ai_model_settings import default_caption_model, default_lmstudio_base_url
from photoalbums.lib.ai_sidecar_state import read_ai_sidecar_state
from photoalbums.lib.caption_layout_migration import _resolve_parent_page_sidecar
from photoalbums.lib.xmp_review import load_ai_xmp_review
from photoalbums.naming import DERIVED_VIEW_RE, is_photos_dir

MAX_TOKENS = 1024
TEMPERATURE = 0.0
MAX_IMAGE_EDGE = DEFAULT_LMSTUDIO_AUTO_MAX_IMAGE_EDGE

SYSTEM_PROMPT = (
    "- You inspect a scanned photo-album page and one cropped photo taken from it, and report two things about"
    " the crop: its caption and the location it shows.\n"
    "- You are shown the full album page for context and the cropped photo as the target.\n"
    "- Return only valid JSON matching the response_format schema.\n"
    "- `caption`: copy the caption text the album owner wrote near the target photo on the page, verbatim -- keep"
    " the original wording, casing, and punctuation. A caption may sit beneath, beside, or above the photo, and"
    " one caption may apply to several adjacent photos.\n"
    "- `has_caption`: true only when caption text is actually written near the target photo on the page; false"
    " otherwise. When false, return an empty `caption`. Never invent, translate, or describe -- only transcribe"
    " real caption text a human can read on the page.\n"
    "- `location_shown`: a Nominatim-queryable place name for where the target photo was taken or what it depicts,"
    " as a single string that ends with a country name (e.g. 'Crystal Lake, California, United States'). Use"
    " visible landmarks, signs, the caption, and the album context.\n"
    "- `has_location`: true only when you can identify the location with reasonable confidence; false otherwise."
    " When false, return an empty `location_shown`.\n"
    "- `reasoning`: one short sentence covering the caption and the location."
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "crop_caption_location_payload",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "has_caption": {"type": "boolean"},
                    "location_shown": {"type": "string"},
                    "has_location": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
                "required": ["caption", "has_caption", "location_shown", "has_location", "reasoning"],
                "additionalProperties": False,
            },
        },
    }


def _build_prompt(
    *,
    page_name: str,
    crop_name: str,
    album_title: str,
    page_description: str,
    current_caption: str,
    current_location: str,
) -> str:
    parts = [
        "Report the caption and the shown location for the target crop. Read the caption text from the album"
        " page. Return empty/false fields when a caption or location cannot be determined.",
        f"Album page file: {page_name}",
        f"Target crop file: {crop_name}",
        f"Album title: {album_title or '(unknown)'}",
        "Album page description (context, not authoritative):",
        page_description or "(none)",
        "Current crop caption (may be wrong -- do not trust it):",
        current_caption or "(empty)",
        "Current crop location (may be wrong -- do not trust it):",
        current_location or "(empty)",
    ]
    return "\n\n".join(part for part in parts if part).strip()


def _call_model(
    *,
    page_image: Path,
    crop_image: Path,
    prompt: str,
    model_name: str,
    base_url: str,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, object]:
    resolved_model = _select_lmstudio_model(base_url, model_name, DEFAULT_LMSTUDIO_TIMEOUT_SECONDS)
    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Supporting album page (read the caption here)."},
                    {"type": "image_url", "image_url": {"url": _build_data_url(page_image, MAX_IMAGE_EDGE)}},
                    {"type": "text", "text": "Target crop."},
                    {"type": "image_url", "image_url": {"url": _build_data_url(crop_image, MAX_IMAGE_EDGE)}},
                ],
            },
        ],
        "response_format": _response_format(),
        "max_tokens": int(max_tokens),
        "temperature": TEMPERATURE,
        "stream": False,
    }
    response = _lmstudio_request_json(
        f"{base_url}/chat/completions", payload=payload, timeout=DEFAULT_LMSTUDIO_TIMEOUT_SECONDS
    )
    choices = list(response.get("choices") or [])
    if not choices:
        raise RuntimeError("LM Studio returned no choices.")
    finish_reason = _clean(choices[0].get("finish_reason"))
    if finish_reason == "length":
        raise RuntimeError(f"response truncated at max_tokens={max_tokens}; rerun with a higher --max-tokens")
    content = dict(choices[0].get("message") or {}).get("content")
    parsed = content if isinstance(content, dict) else None
    if parsed is None and isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = _extract_structured_json_payload(content)  # tolerant: handles fences / trailing commas
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Model response was not valid JSON: {_format_lmstudio_debug_response(content)}")
    return {
        "caption": _clean(parsed.get("caption")),
        "has_caption": bool(parsed.get("has_caption")),
        "location_shown": _clean(parsed.get("location_shown")),
        "has_location": bool(parsed.get("has_location")),
        "reasoning": _clean(parsed.get("reasoning")),
    }


def _existing_location_payload(state: dict) -> dict[str, object]:
    return dict((state.get("detections") or {}).get("location") or {})


def _write_crop_sidecar(
    crop_image: Path,
    state: dict,
    *,
    description: str,
    location_payload: dict[str, object] | None,
    locations_shown: list[dict] | None,
    backup: bool = False,
) -> bool:
    """Write the crop sidecar via the canonical ai-index writer, preserving everything
    except the caption (description) and, when supplied, the location.

    person_names and subjects are NOT exposed by read_ai_sidecar_state, so they are
    sourced from load_ai_xmp_review; passing them through is required or the writer
    (which manages those elements) would clear PersonInImage / dc:subject.

    Returns True if a backup (.xmp.bak) was created."""
    crop_xmp = crop_image.with_suffix(".xmp")
    backed_up = False
    if backup and crop_xmp.is_file():
        bak = crop_xmp.with_name(crop_xmp.name + ".bak")
        if not bak.exists():  # keep the pristine pre-script original; don't clobber on re-runs
            shutil.copy2(crop_xmp, bak)
            backed_up = True
    review = load_ai_xmp_review(crop_xmp)
    review = review if isinstance(review, dict) else {}
    detections = dict(state.get("detections") or {})
    if location_payload is not None:
        detections["location"] = location_payload
    if locations_shown is not None:
        detections["locations_shown"] = locations_shown
    _write_sidecar_and_record(
        crop_xmp,
        crop_image,
        person_names=list(review.get("person_names") or []),
        subjects=list(review.get("subjects") or []),
        title=_clean(state.get("title")),
        title_source=_clean(state.get("title_source")),
        description=description,
        album_title=_clean(state.get("album_title")),
        location_payload=dict(location_payload or _existing_location_payload(state)),
        source_text=_clean(state.get("source_text")),
        ocr_text=_clean(state.get("ocr_text")),
        ocr_lang=_clean(state.get("ocr_lang")),
        author_text=_clean(state.get("author_text")),
        scene_text=_clean(state.get("scene_text")),
        detections_payload=detections,
        stitch_key=_clean(state.get("stitch_key")),
        ocr_authority_source=_clean(state.get("ocr_authority_source")),
        create_date=_clean(state.get("create_date")),
        dc_date=_clean(state.get("dc_date")),
        date_time_original=_clean(state.get("date_time_original")),
        replace_dc_date=False,
        ocr_ran=bool(state.get("ocr_ran")),
        people_detected=bool(state.get("people_detected")),
        people_identified=bool(state.get("people_identified")),
        title_page_location=None,
    )
    return backed_up


def _process_crop(
    item: dict[str, object],
    *,
    model_name: str,
    base_url: str,
    geocoder: NominatimGeocoder,
    dry_run: bool,
    backup: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, object]:
    crop_image = Path(str(item["crop_image"]))
    page_image = Path(str(item["page_image"]))
    page_xmp = Path(str(item["page_xmp"]))
    state = read_ai_sidecar_state(crop_image.with_suffix(".xmp"))
    if not isinstance(state, dict):
        raise RuntimeError(f"Could not read crop sidecar state: {crop_image.with_suffix('.xmp')}")
    page_state = read_ai_sidecar_state(page_xmp) if page_xmp.is_file() else {}
    page_state = page_state if isinstance(page_state, dict) else {}

    current_caption = _clean(state.get("description"))
    current_loc = _existing_location_payload(state)
    current_loc_name = _clean(current_loc.get("display_name")) or _clean(current_loc.get("query"))

    prompt = _build_prompt(
        page_name=page_image.name,
        crop_name=crop_image.name,
        album_title=_clean(page_state.get("album_title")) or _clean(state.get("album_title")),
        page_description=_clean(page_state.get("description")),
        current_caption=current_caption,
        current_location=current_loc_name,
    )
    resolved = _call_model(
        page_image=page_image,
        crop_image=crop_image,
        prompt=prompt,
        model_name=model_name,
        base_url=base_url,
        max_tokens=max_tokens,
    )

    new_caption = str(resolved["caption"]) if resolved["has_caption"] else ""
    caption_changed = new_caption != current_caption

    new_location_payload: dict[str, object] | None = None
    new_locations_shown: list[dict] | None = None
    new_loc_name = current_loc_name
    geocode_note = ""
    location_changed = False
    loc_name = str(resolved["location_shown"])
    if resolved["has_location"] and loc_name:
        try:
            payload, geo_err = _resolve_geocoded_location_payload(
                geocoder, query=loc_name, artifact_step="recheck-crop-caption-location"
            )
        except Exception as exc:
            payload, geo_err = {}, str(exc)
        if payload:
            new_location_payload = payload
            new_locations_shown = [
                _canonical_location_shown_entry(
                    {
                        "name": loc_name,
                        "country": payload.get("country", ""),
                        "state": payload.get("state", ""),
                        "city": payload.get("city", ""),
                        "sublocation": payload.get("sublocation", ""),
                        "gps_latitude": payload.get("gps_latitude"),
                        "gps_longitude": payload.get("gps_longitude"),
                    }
                )
            ]
            new_loc_name = _clean(payload.get("display_name")) or loc_name
            location_changed = new_loc_name != current_loc_name
            geocode_note = "geocoded"
        else:
            geocode_note = f"geocode miss: {geo_err}" if geo_err else "geocode miss"

    wrote = False
    backed_up = False
    if (caption_changed or location_changed) and not dry_run:
        backed_up = _write_crop_sidecar(
            crop_image,
            state,
            description=new_caption,
            location_payload=new_location_payload,
            locations_shown=new_locations_shown,
            backup=backup,
        )
        wrote = True

    return {
        "crop_path": crop_image,
        "backed_up": backed_up,
        "current_caption": current_caption,
        "new_caption": new_caption,
        "has_caption": bool(resolved["has_caption"]),
        "caption_changed": caption_changed,
        "current_location": current_loc_name,
        "new_location": new_loc_name,
        "has_location": bool(resolved["has_location"]),
        "location_changed": location_changed,
        "geocode_note": geocode_note,
        "reasoning": str(resolved["reasoning"]),
        "wrote": wrote,
        "dry_run": dry_run,
    }


def _print_crop_result(result: dict[str, object]) -> None:
    name = Path(str(result["crop_path"])).name
    verb = "PLAN" if result["dry_run"] else "WROTE"
    print(f"   {name}")
    if result["caption_changed"]:
        before = result["current_caption"] or "(empty)"
        after = result["new_caption"] or "(none)"
        print(f"      caption  {verb}: {before!r} -> {after!r}")
    else:
        print(f"      caption  KEEP: {result['current_caption'] or '(empty)'!r}")
    if result["location_changed"]:
        before = result["current_location"] or "(empty)"
        after = result["new_location"] or "(none)"
        note = f" [{result['geocode_note']}]" if result["geocode_note"] else ""
        print(f"      location {verb}: {before!r} -> {after!r}{note}")
    else:
        note = f" [{result['geocode_note']}]" if result["geocode_note"] else ""
        print(f"      location KEEP: {result['current_location'] or '(empty)'!r}{note}")
    if result["backed_up"]:
        print(f"      backup: {name}.xmp.bak")
    if result["reasoning"]:
        print(f"      reason: {result['reasoning']}")


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # non-positive or NaN
        return "--:--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def _iter_crop_images(root: Path, album_filter: str, page_filter: str, crop_filter: str):
    negate = album_filter.startswith("!")  # "!Family" -> process albums that do NOT match
    needle = album_filter[1:] if negate else album_filter
    for photos_dir in sorted(d for d in root.iterdir() if d.is_dir() and is_photos_dir(d)):
        if needle and (needle in photos_dir.name.casefold()) == negate:
            continue
        for path in sorted(photos_dir.glob("*_V.jpg")):
            if not DERIVED_VIEW_RE.search(path.stem):
                continue
            if page_filter and f"_P{page_filter}_" not in path.name:
                continue
            if crop_filter and crop_filter not in path.name:
                continue
            yield path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate crop captions + shown locations (page+crop) and write the crop XMP sidecars.",
    )
    parser.add_argument("--photos-root", default=str(PHOTO_ALBUMS_DIR), help="Photo Albums root directory.")
    parser.add_argument(
        "--album",
        default="",
        help="Optional substring filter against the _Photos dir name. Prefix with ! to negate (e.g. !Family).",
    )
    parser.add_argument("--page", default="", help="Optional page number filter (e.g. 03).")
    parser.add_argument("--crop", default="", help="Optional substring filter against the crop filename (e.g. D05-00).")
    parser.add_argument("--model", default="", help="Model name (defaults to configured caption model).")
    parser.add_argument("--base-url", default="", help="LM Studio base URL (defaults to configured URL).")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N crops (0 = no limit).")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help=f"Model max_tokens (default {MAX_TOKENS}).")
    parser.add_argument("--run", action="store_true", help="Write the sidecars in place. Omit for a preview dry run.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="With --run, copy each crop's .xmp to .xmp.bak before overwriting (skips if a .bak already exists).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.photos_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Photo Albums root does not exist: {root}")

    try:
        import PIL  # noqa: F401  (sidecar writing needs Pillow for image dimensions)
    except ImportError:
        print(
            "ERROR: Pillow (PIL) is not importable in this interpreter. Run with the project venv, "
            "e.g. `uv run python -m photoalbums.scripts.recheck_crop_caption_location ...`.",
            file=sys.stderr,
        )
        return 2

    album_filter = str(args.album or "").casefold()
    page_filter = f"{int(args.page):02d}" if str(args.page or "").strip().isdigit() else ""
    crop_filter = str(args.crop or "").strip()
    dry_run = not bool(args.run)
    backup = bool(args.backup)
    if backup and dry_run:
        print("note: --backup has no effect during a dry run (nothing is written); pass --run to write + back up.")

    model_name = str(args.model or "").strip()
    if not model_name:
        model_name = "google/gemma-4-12b-qat" if sys.platform == "win32" else default_caption_model()
    base_url = normalize_lmstudio_base_url(
        str(args.base_url or "").strip() or default_lmstudio_base_url() or DEFAULT_LMSTUDIO_BASE_URL
    )
    geocoder = NominatimGeocoder()

    cap_changed = 0
    loc_changed = 0
    unchanged = 0
    backups = 0
    failures = 0

    print(f"{'dry-run (preview only)' if dry_run else 'run'} model={model_name} base_url={base_url} root={root}")

    # --- Discovery pass: pair each crop with its parent page (for the total + ETA) ---
    print("scanning for crops to validate ...")
    work: list[dict[str, object]] = []
    for crop_image in _iter_crop_images(root, album_filter, page_filter, crop_filter):
        if args.limit and len(work) >= args.limit:
            break
        page_xmp = _resolve_parent_page_sidecar(crop_image.with_suffix(".xmp"))
        if page_xmp is None or not page_xmp.with_suffix(".jpg").is_file():
            print(f"SKIP  {crop_image.name}: parent page image not found")
            continue
        work.append({"crop_image": crop_image, "page_image": page_xmp.with_suffix(".jpg"), "page_xmp": page_xmp})

    total = len(work)
    print(f"found {total} crop(s) to validate")

    # --- Processing pass: one page+crop call per crop, with rate + ETA ---
    start = time.monotonic()
    for index, item in enumerate(work, 1):
        crop_name = Path(str(item["crop_image"])).name
        done = index - 1
        elapsed = time.monotonic() - start
        per_item = elapsed / done if done else 0.0
        rate_per_min = (done / elapsed * 60.0) if (done and elapsed > 0) else 0.0
        eta = per_item * (total - done) if per_item else 0.0
        rate_text = f"{rate_per_min:.1f}/min" if rate_per_min else "--/min"
        print(
            f"[{index}/{total}] {100 * done / total:5.1f}% | {rate_text} "
            f"| elapsed {_fmt_duration(elapsed)} eta {_fmt_duration(eta)} | {crop_name}"
        )
        try:
            result = _process_crop(
                item,
                model_name=model_name,
                base_url=base_url,
                geocoder=geocoder,
                dry_run=dry_run,
                backup=backup,
                max_tokens=int(args.max_tokens),
            )
        except Exception as exc:
            failures += 1
            print(f"FAIL  {crop_name}: {exc}")
            continue
        if result["caption_changed"]:
            cap_changed += 1
        if result["location_changed"]:
            loc_changed += 1
        if not result["caption_changed"] and not result["location_changed"]:
            unchanged += 1
        if result["backed_up"]:
            backups += 1
        _print_crop_result(result)

    total_elapsed = time.monotonic() - start
    avg_rate = (total / total_elapsed * 60.0) if (total and total_elapsed > 0) else 0.0
    action = "would_change" if dry_run else "changed"
    summary = (
        f"done crops={total} caption_{action}={cap_changed} location_{action}={loc_changed} "
        f"unchanged={unchanged} "
        + (f"backups={backups} " if backup else "")
        + f"failures={failures} "
        + f"| elapsed {_fmt_duration(total_elapsed)} "
        + (f"avg {avg_rate:.1f}/min" if avg_rate else "avg --/min")
    )
    print(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
