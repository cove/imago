from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.lib._caption_lmstudio import (
    _build_data_url,
    _lmstudio_caption_response_format,
    _lmstudio_page_caption_response_format,
    _lmstudio_request_json,
)
from photoalbums.lib._caption_prompts import _build_local_prompt

DEFAULT_EVAL_DIR = REPO_ROOT / "photoalbums" / "evals"
DEFAULT_RESULTS_DIR = DEFAULT_EVAL_DIR / "results"
DEFAULT_PHOTOS_ROOT = Path(r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums")
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_IMAGE_MAX_EDGE = 1600
DIRECT_GENERATION_KEYS = (
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local prompt/model eval sweeps against LM Studio.")
    parser.add_argument(
        "--eval-file",
        action="append",
        default=[],
        help="Path to an eval JSON file. Repeatable. Defaults to all JSON files under photoalbums/evals.",
    )
    parser.add_argument(
        "--eval-filter",
        default="",
        help="Substring filter applied to eval file name or eval id.",
    )
    parser.add_argument(
        "--prompt-file",
        action="append",
        default=[],
        help=(
            "Optional prompt template file. Repeatable. Supports {album_title}, {ocr_text}, {photo}, "
            "and {production_prompt} placeholders."
        ),
    )
    parser.add_argument(
        "--matrix-file",
        default="",
        help="Optional JSON file describing run variants. May be a list or an object with a variants array.",
    )
    parser.add_argument(
        "--photos-root",
        default=str(DEFAULT_PHOTOS_ROOT),
        help="Root directory used to resolve the eval JSON photo field.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where result JSON files are written.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout for each LM Studio request.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_eval_files(args: argparse.Namespace) -> list[Path]:
    if args.eval_file:
        files = [Path(value).resolve() for value in args.eval_file]
    else:
        files = sorted(path.resolve() for path in DEFAULT_EVAL_DIR.glob("*.json"))
    filter_text = str(args.eval_filter or "").strip().casefold()
    if not filter_text:
        return files
    result: list[Path] = []
    for path in files:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        eval_id = str(payload.get("id") or path.stem)
        if filter_text in path.name.casefold() or filter_text in eval_id.casefold():
            result.append(path)
    return result


def _resolve_prompt_variants(prompt_files: list[str]) -> list[dict[str, str]]:
    if not prompt_files:
        return [{"id": "production", "path": "", "template": ""}]
    variants: list[dict[str, str]] = []
    for raw_path in prompt_files:
        path = Path(raw_path).resolve()
        variants.append(
            {
                "id": path.stem,
                "path": str(path),
                "template": path.read_text(encoding="utf-8"),
            }
        )
    return variants


def _resolve_matrix_variants(matrix_file: str) -> list[dict[str, Any]]:
    if not str(matrix_file or "").strip():
        return [{"id": "default"}]
    payload = _load_json(Path(matrix_file).resolve())
    if isinstance(payload, list):
        variants = payload
    elif isinstance(payload, dict):
        variants = payload.get("variants") or []
    else:
        raise ValueError(f"Unsupported matrix file shape: {matrix_file}")
    result: list[dict[str, Any]] = []
    for idx, item in enumerate(list(variants), 1):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row.setdefault("id", f"variant-{idx}")
        result.append(row)
    return result or [{"id": "default"}]


def _resolve_response_format(name: str) -> dict[str, object]:
    text = str(name or "page_caption_payload").strip()
    if text == "page_caption_payload":
        return _lmstudio_page_caption_response_format()
    if text == "caption_payload":
        return _lmstudio_caption_response_format()
    raise ValueError(f"Unsupported response_format: {text}")


def _build_production_prompt(eval_payload: dict[str, Any], image_path: Path) -> str:
    context = dict(eval_payload.get("input_context") or {})
    return _build_local_prompt(
        people=list(context.get("people") or []),
        objects=list(context.get("objects") or []),
        ocr_text=str(context.get("ocr_text") or ""),
        source_path=str(image_path),
        album_title=str(context.get("album_title") or ""),
        printed_album_title=str(context.get("printed_album_title") or context.get("album_title") or ""),
        context_ocr_text=str(context.get("context_ocr_text") or ""),
    )


def _render_prompt(template: str, *, production_prompt: str, eval_payload: dict[str, Any], image_path: Path) -> str:
    if not template:
        return production_prompt
    context = dict(eval_payload.get("input_context") or {})
    values = {
        "album_title": str(context.get("album_title") or ""),
        "printed_album_title": str(context.get("printed_album_title") or context.get("album_title") or ""),
        "ocr_text": str(context.get("ocr_text") or ""),
        "context_ocr_text": str(context.get("context_ocr_text") or ""),
        "photo": str(image_path),
        "production_prompt": production_prompt,
    }
    return template.format(**values)


def _merge_run_config(eval_payload: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    settings = dict(eval_payload.get("run_config") or {})
    settings.update(dict(variant or {}))
    settings.setdefault("model", "")
    settings.setdefault("endpoint", "http://127.0.0.1:1234/v1/chat/completions")
    settings.setdefault("response_format", "page_caption_payload")
    settings.setdefault("temperature", DEFAULT_TEMPERATURE)
    settings.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    settings.setdefault("stream", False)
    settings.setdefault("image_max_edge", DEFAULT_IMAGE_MAX_EDGE)
    return settings


def _call_lmstudio(
    *,
    image_path: Path,
    prompt_text: str,
    settings: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": str(settings.get("model") or ""),
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": _build_data_url(image_path, int(settings.get("image_max_edge") or 0))},
                    },
                ],
            },
        ],
        "response_format": _resolve_response_format(str(settings.get("response_format") or "")),
        "max_tokens": int(settings.get("max_tokens") or DEFAULT_MAX_TOKENS),
        "temperature": float(settings.get("temperature") or DEFAULT_TEMPERATURE),
        "stream": bool(settings.get("stream", False)),
    }
    for key in DIRECT_GENERATION_KEYS:
        if key in settings:
            payload[key] = settings[key]
    response = _lmstudio_request_json(
        str(settings.get("endpoint") or ""),
        payload=payload,
        timeout=timeout_seconds,
    )
    choices = list(response.get("choices") or [])
    if not choices:
        raise RuntimeError("LM Studio returned no choices.")
    choice = dict(choices[0] or {})
    message = dict(choice.get("message") or {})
    raw_response = str(message.get("content") or "")
    parse_error = ""
    parsed_json: Any = None
    try:
        parsed_json = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "finish_reason": str(choice.get("finish_reason") or ""),
        "raw_response": raw_response,
        "parsed_json": parsed_json,
        "valid_json": parse_error == "",
        "parse_error": parse_error,
    }


def _apply_checks(eval_payload: dict[str, Any], call_result: dict[str, Any]) -> dict[str, Any]:
    checks = dict(eval_payload.get("checks") or {})
    failures: list[str] = []
    raw_response = str(call_result.get("raw_response") or "")
    parsed_json = call_result.get("parsed_json")
    check_field = str(checks.get("field") or "")
    field_text = raw_response
    if check_field and isinstance(parsed_json, dict):
        field_text = str(parsed_json.get(check_field) or "")
    if bool(checks.get("require_valid_json")) and not bool(call_result.get("valid_json")):
        failures.append("require_valid_json")
    for value in list(checks.get("must_contain") or []):
        if str(value).lower() not in field_text.lower():
            failures.append(f"missing:{value}")
    for value in list(checks.get("must_not_contain") or []):
        if str(value).lower() in field_text.lower():
            failures.append(f"forbidden:{value}")
    return {"passed": not failures, "failures": failures}


def _slug(text: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "-" for ch in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "item"


def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    args = _parse_args()
    photos_root = Path(args.photos_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_files = _resolve_eval_files(args)
    if not eval_files:
        raise SystemExit("No eval JSON files matched.")
    prompt_variants = _resolve_prompt_variants(list(args.prompt_file or []))
    matrix_variants = _resolve_matrix_variants(str(args.matrix_file or ""))

    run_started = datetime.now().astimezone()
    run_rows: list[dict[str, Any]] = []
    print(f"Running {len(eval_files)} eval file(s), {len(prompt_variants)} prompt variant(s), {len(matrix_variants)} settings variant(s).")
    for eval_path in eval_files:
        eval_payload = dict(_load_json(eval_path) or {})
        eval_id = str(eval_payload.get("id") or eval_path.stem)
        photo_rel = str(eval_payload.get("photo") or "").strip()
        if not photo_rel:
            print(f"SKIP {eval_id}: missing photo field")
            continue
        image_path = (photos_root / photo_rel).resolve()
        if not image_path.is_file():
            print(f"SKIP {eval_id}: image not found: {image_path}")
            continue
        production_prompt = _build_production_prompt(eval_payload, image_path)
        for prompt_variant in prompt_variants:
            prompt_text = _render_prompt(
                str(prompt_variant.get("template") or ""),
                production_prompt=production_prompt,
                eval_payload=eval_payload,
                image_path=image_path,
            )
            for matrix_variant in matrix_variants:
                settings = _merge_run_config(eval_payload, matrix_variant)
                started = datetime.now().astimezone()
                row: dict[str, Any] = {
                    "eval_id": eval_id,
                    "eval_file": str(eval_path),
                    "photo": str(image_path),
                    "prompt_id": str(prompt_variant.get("id") or "production"),
                    "prompt_file": str(prompt_variant.get("path") or ""),
                    "prompt_hash": _prompt_hash(prompt_text),
                    "settings_id": str(matrix_variant.get("id") or "default"),
                    "settings": settings,
                    "started_at": started.isoformat(),
                    "prompt_text": prompt_text,
                }
                try:
                    call_result = _call_lmstudio(
                        image_path=image_path,
                        prompt_text=prompt_text,
                        settings=settings,
                        timeout_seconds=float(args.timeout_seconds),
                    )
                    row.update(call_result)
                    row["checks"] = _apply_checks(eval_payload, call_result)
                    status = "PASS" if row["checks"]["passed"] else "FAIL"
                    print(
                        f"{status} {eval_id} prompt={row['prompt_id']} settings={row['settings_id']} "
                        f"json={row['valid_json']} finish={row['finish_reason'] or '-'}"
                    )
                except Exception as exc:
                    row["valid_json"] = False
                    row["parse_error"] = ""
                    row["error"] = str(exc)
                    row["checks"] = {"passed": False, "failures": [f"exception:{exc}"]}
                    print(f"FAIL {eval_id} prompt={row['prompt_id']} settings={row['settings_id']} error={exc}")
                run_rows.append(row)

    finished = datetime.now().astimezone()
    output_path = results_dir / f"prompt-eval-{run_started.strftime('%Y%m%d-%H%M%S')}.json"
    output_payload = {
        "started_at": run_started.isoformat(),
        "finished_at": finished.isoformat(),
        "photos_root": str(photos_root),
        "results": run_rows,
    }
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for row in run_rows if bool(dict(row.get("checks") or {}).get("passed")))
    print(f"Wrote {len(run_rows)} result(s) to {output_path}")
    print(f"Passed {passed}/{len(run_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
