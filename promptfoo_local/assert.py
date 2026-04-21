from __future__ import annotations

import json


def get_assert(output: str, context: dict) -> dict:
    vars_payload = dict(context.get("vars") or {})
    checks = dict(vars_payload.get("checks") or {})
    require_valid_json = bool(vars_payload.get("require_valid_json"))
    field_name = str(checks.get("field") or "")
    failures: list[str] = []

    parsed = None
    if require_valid_json or field_name:
        try:
            parsed = json.loads(str(output or ""))
        except json.JSONDecodeError as exc:
            if require_valid_json:
                failures.append(f"invalid_json:{exc}")

    field_text = str(output or "")
    if field_name and isinstance(parsed, dict):
        field_text = str(parsed.get(field_name) or "")

    for value in list(checks.get("must_contain") or []):
        if str(value).lower() not in field_text.lower():
            failures.append(f"missing:{value}")
    for value in list(checks.get("must_not_contain") or []):
        if str(value).lower() in field_text.lower():
            failures.append(f"forbidden:{value}")

    return {
        "pass": not failures,
        "score": 1.0 if not failures else 0.0,
        "reason": ", ".join(failures) if failures else "passed",
    }
