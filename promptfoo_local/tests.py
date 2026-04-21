from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.lib._caption_prompts import _build_local_prompt

DEFAULT_EVAL_DIR = REPO_ROOT / "photoalbums" / "evals"
DEFAULT_PHOTOS_ROOT = Path(r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _should_include_eval(payload: dict) -> bool:
    return bool(payload.get("photo")) and isinstance(payload.get("input_context"), dict)


def _resolve_image_paths(eval_payload: dict, photos_root: Path) -> list[str]:
    paths: list[str] = []
    for key in ("photo", "overlay_image"):
        value = str(eval_payload.get(key) or "").strip()
        if value:
            paths.append(str((photos_root / value).resolve()))
    for value in list(eval_payload.get("images") or []):
        text = str(value or "").strip()
        if text:
            paths.append(str((photos_root / text).resolve()))
    seen: set[str] = set()
    deduped: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def generate_tests():
    eval_filter = str(Path.cwd().joinpath(".").name)  # no-op, keeps signature simple for promptfoo
    del eval_filter
    filter_text = str(__import__("os").environ.get("PROMPTFOO_EVAL_FILTER", "") or "").strip().casefold()
    eval_dir = DEFAULT_EVAL_DIR
    photos_root = Path(__import__("os").environ.get("PROMPTFOO_PHOTOS_ROOT", str(DEFAULT_PHOTOS_ROOT))).resolve()
    tests: list[dict] = []
    for path in sorted(eval_dir.glob("*.json")):
        payload = _load_json(path)
        if not _should_include_eval(payload):
            continue
        eval_id = str(payload.get("id") or path.stem)
        if filter_text and filter_text not in eval_id.casefold() and filter_text not in path.name.casefold():
            continue
        image_paths = _resolve_image_paths(payload, photos_root)
        if not image_paths:
            continue
        context = dict(payload.get("input_context") or {})
        production_prompt = _build_local_prompt(
            people=list(context.get("people") or []),
            objects=list(context.get("objects") or []),
            ocr_text=str(context.get("ocr_text") or ""),
            source_path=image_paths[0],
            album_title=str(context.get("album_title") or ""),
            printed_album_title=str(context.get("printed_album_title") or context.get("album_title") or ""),
            context_ocr_text=str(context.get("context_ocr_text") or ""),
        )
        checks = dict(payload.get("checks") or {})
        pass_criteria = [str(item) for item in list(payload.get("pass_criteria") or [])]
        require_valid_json = bool(checks.get("require_valid_json"))
        if not require_valid_json:
            require_valid_json = any("valid json" in item.casefold() for item in pass_criteria)
        tests.append(
            {
                "vars": {
                    "eval_id": eval_id,
                    "eval_file": str(path.resolve()),
                    "image_paths": image_paths,
                    "photo_path": image_paths[0],
                    "album_title": str(context.get("album_title") or ""),
                    "printed_album_title": str(context.get("printed_album_title") or context.get("album_title") or ""),
                    "ocr_text": str(context.get("ocr_text") or ""),
                    "context_ocr_text": str(context.get("context_ocr_text") or ""),
                    "production_prompt": production_prompt,
                    "checks": checks,
                    "require_valid_json": require_valid_json,
                    "pass_criteria": pass_criteria,
                    "run_config": dict(payload.get("run_config") or {}),
                },
                "metadata": {
                    "eval_id": eval_id,
                    "eval_file": str(path.resolve()),
                    "photo_path": image_paths[0],
                },
            }
        )
    return tests
