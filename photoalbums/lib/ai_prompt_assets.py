from __future__ import annotations

import hashlib
import json
import string
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


class PromptAssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptAsset:
    path: Path
    text: str
    rendered: str
    hash: str

    def metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "hash": self.hash,
        }


@dataclass(frozen=True)
class ParamsAsset:
    path: Path
    values: dict[str, Any]
    hash: str

    def metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "hash": self.hash,
        }


@dataclass(frozen=True)
class SchemaAsset:
    path: Path
    values: dict[str, Any]
    hash: str


def prompt_root() -> Path:
    return PROMPT_ROOT


def asset_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return prompt_root() / candidate


def content_hash(value: object) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptAssetError(f"Could not read prompt asset {path}: {exc}") from exc


def _stat_mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError as exc:
        raise PromptAssetError(f"Could not stat prompt asset {path}: {exc}") from exc


@lru_cache(maxsize=128)
def _cached_text(path: Path, mtime_ns: int) -> str:
    del mtime_ns
    return _read_text(path)


def load_text(relative_path: str | Path) -> tuple[Path, str]:
    path = asset_path(relative_path)
    return path, _cached_text(path, _stat_mtime(path))


class _DropLineFormatter(string.Formatter):
    def __init__(self, variables: dict[str, Any]) -> None:
        super().__init__()
        self.variables = variables
        self.drop = False

    def get_value(self, key: object, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if isinstance(key, str):
            value = self.variables.get(key, "")
            if value in (None, ""):
                self.drop = True
                return ""
            return value
        return super().get_value(key, args, kwargs)


def render_template(text: str, variables: dict[str, Any] | None = None) -> str:
    if not variables:
        return str(text or "").strip()
    rendered_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        formatter = _DropLineFormatter(dict(variables or {}))
        try:
            rendered = formatter.format(raw_line)
        except Exception as exc:
            raise PromptAssetError(f"Could not render prompt template line {raw_line!r}: {exc}") from exc
        if not formatter.drop:
            rendered_lines.append(rendered)
    return "\n".join(rendered_lines).strip()


def load_prompt(relative_path: str | Path, variables: dict[str, Any] | None = None) -> PromptAsset:
    path, text = load_text(relative_path)
    try:
        rendered = render_template(text, variables)
    except PromptAssetError as exc:
        raise PromptAssetError(f"Could not render prompt asset {path}: {exc}") from exc
    return PromptAsset(
        path=path,
        text=text,
        rendered=rendered,
        hash=content_hash(text),
    )


@lru_cache(maxsize=64)
def _cached_toml(path: Path, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    try:
        return dict(tomllib.loads(_read_text(path)))
    except tomllib.TOMLDecodeError as exc:
        raise PromptAssetError(f"Could not parse TOML params {path}: {exc}") from exc


def load_params(relative_path: str | Path) -> ParamsAsset:
    path = asset_path(relative_path)
    values = _cached_toml(path, _stat_mtime(path))
    return ParamsAsset(path=path, values=dict(values), hash=content_hash(values))


@lru_cache(maxsize=64)
def _cached_json(path: Path, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    try:
        payload = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise PromptAssetError(f"Could not parse JSON schema {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromptAssetError(f"Could not parse JSON schema {path}: expected object, got {type(payload).__name__}")
    return payload


def load_schema(relative_path: str | Path) -> SchemaAsset:
    path = asset_path(relative_path)
    values = _cached_json(path, _stat_mtime(path))
    return SchemaAsset(path=path, values=dict(values), hash=content_hash(values))


def prompt_metadata(*assets: PromptAsset) -> dict[str, object]:
    rows = [asset.metadata() for asset in assets if asset.path]
    return {
        "prompt_paths": [str(row["path"]) for row in rows],
        "prompt_hashes": [str(row["hash"]) for row in rows],
    }


def params_metadata(asset: ParamsAsset | None, resolved: dict[str, Any], overrides: dict[str, str] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "resolved_params": dict(resolved),
        "override_sources": dict(overrides or {}),
    }
    if asset is not None:
        metadata["params_path"] = str(asset.path)
        metadata["params_hash"] = asset.hash
    return metadata


def asset_hashes(*relative_paths: str | Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in relative_paths:
        path, text = load_text(relative_path)
        hashes[str(relative_path).replace("\\", "/")] = content_hash(text)
        if path.suffix.lower() == ".toml":
            hashes[str(relative_path).replace("\\", "/")] = load_params(relative_path).hash
    return hashes
