from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable


def read_tag(file_path: str | Path, tag: str) -> str | None:
    try:
        result = subprocess.run(
            ["exiftool", f"-{tag}", "-s3", str(file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def write_tags(
    file_path: str | Path,
    *,
    set_tags: dict[str, str] | None = None,
    clear_tags: Iterable[str] | None = None,
    overwrite_original: bool = True,
) -> None:
    command = ["exiftool"]
    if overwrite_original:
        command.append("-overwrite_original")
    for tag in list(clear_tags or []):
        clean_tag = str(tag or "").strip()
        if clean_tag:
            command.append(f"-{clean_tag}=")
    for tag, value in dict(set_tags or {}).items():
        clean_tag = str(tag or "").strip()
        if not clean_tag:
            continue
        command.append(f"-{clean_tag}={value}")
    command.append(str(file_path))
    subprocess.run(command, check=True)


def read_json_tags(file_path: str | Path, tags: list[str]) -> dict:
    result = subprocess.run(
        [
            "exiftool",
            "-json",
            "-a",
            "-G1",
            "-struct",
            "-charset",
            "UTF8",
            *[f"-{tag}" for tag in list(tags or [])],
            str(file_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    if not payload:
        return {}
    first = payload[0]
    if isinstance(first, dict):
        return first
    return {}
