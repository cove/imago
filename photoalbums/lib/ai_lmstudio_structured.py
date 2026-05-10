from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ._caption_lmstudio import (
    _build_data_url,
    _format_lmstudio_debug_response,
)
from ._caption_lmstudio import (
    _lmstudio_request_json as _default_lmstudio_request_json,
)
from .ai_prompt_assets import load_params, load_prompt, load_schema, params_metadata, prompt_metadata


def schema_response_format(*, schema_name: str, schema_path: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": str(schema_name),
            "strict": True,
            "schema": load_schema(schema_path).values,
        },
    }


def prompt_assets_metadata(
    *,
    system_path: str,
    user_path: str,
    schema_path: str,
    params_path: str,
    resolved_params: dict,
) -> dict:
    metadata: dict = {}
    metadata.update(
        prompt_metadata(
            load_prompt(system_path),
            load_prompt(user_path),
            load_schema(schema_path),
        )
    )
    metadata.update(params_metadata(load_params(params_path), resolved_params))
    return metadata


def step_prompt_assets_metadata(*, step: str, resolved_params: dict) -> dict:
    base = f"ai-index/{step}"
    return prompt_assets_metadata(
        system_path=f"{base}/system.md",
        user_path=f"{base}/user.md",
        schema_path=f"{base}/schema.json",
        params_path=f"{base}/params.toml",
        resolved_params=resolved_params,
    )


def run_vision_json_request(
    *,
    base_url: str,
    timeout_seconds: float,
    resolve_model_name: Callable[[], str],
    image_path: Path | str,
    max_image_edge: int,
    system_prompt: str,
    user_prompt: str,
    response_format: dict,
    max_tokens: int,
    temperature: float,
    parse_fn: Callable,
    request_json: Callable | None = None,
    build_data_url: Callable | None = None,
):
    image_url_fn = build_data_url or _build_data_url
    image_url = image_url_fn(image_path, max_image_edge)
    payload = {
        "model": resolve_model_name(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "response_format": response_format,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_fn = request_json or _default_lmstudio_request_json
    response_payload = request_fn(
        f"{base_url}/chat/completions",
        payload=payload,
        timeout=timeout_seconds,
    )
    choices = list(response_payload.get("choices") or [])
    if not choices:
        raise RuntimeError("LM Studio returned no choices.")
    message = dict(choices[0].get("message") or {})
    finish_reason = str(choices[0].get("finish_reason") or "")
    response_text = _format_lmstudio_debug_response(message.get("content"))
    if not response_text:
        response_text = _format_lmstudio_debug_response(message)
    result = parse_fn(message.get("content"), finish_reason=finish_reason)
    return result, response_text, finish_reason


def run_engine_vision_json(
    engine,
    *,
    image_path: Path | str,
    system_prompt: str,
    user_prompt: str,
    response_format: dict,
    parse_fn: Callable,
    request_json: Callable | None = None,
    build_data_url: Callable | None = None,
):
    result, response_text, finish_reason = run_vision_json_request(
        base_url=engine.base_url,
        timeout_seconds=engine.timeout_seconds,
        resolve_model_name=engine._resolve_model_name,
        image_path=image_path,
        max_image_edge=engine.max_image_edge,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=response_format,
        max_tokens=engine.max_tokens,
        temperature=engine.temperature,
        parse_fn=parse_fn,
        request_json=request_json,
        build_data_url=build_data_url,
    )
    engine.last_response_text = response_text
    engine.last_finish_reason = finish_reason
    result.engine = engine.engine
    return result


def run_engine_with_model_fallback(
    engine,
    *,
    image_path: Path | str,
    system_prompt: str,
    user_prompt: str,
    response_format: dict,
    parse_fn: Callable,
    request_json: Callable | None = None,
    build_data_url: Callable | None = None,
):
    return engine._run_with_model_fallback(
        lambda: run_engine_vision_json(
            engine,
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            parse_fn=parse_fn,
            request_json=request_json,
            build_data_url=build_data_url,
        )
    )
