from __future__ import annotations

import re
from pathlib import Path
from typing import Final

PAGE_REFERENCE_VIEW_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<prefix>(?:\.\.[/\\])?[^<>\r\n\"']+?)_View(?P<sep>[/\\])(?P<name>[^<>\r\n\"']+?_V\.(?:jpg|xmp))",
    re.IGNORECASE,
)


def rewrite_page_reference_text(text: str) -> tuple[str, int]:
    replacement_count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal replacement_count
        replacement_count += 1
        return f"{match.group('prefix')}_Pages{match.group('sep')}{match.group('name')}"

    return PAGE_REFERENCE_VIEW_RE.sub(_replace, text), replacement_count


def migrate_sidecar_page_references(sidecar_path: str | Path) -> int:
    path = Path(sidecar_path)
    original = path.read_text(encoding="utf-8")
    updated, replacements = rewrite_page_reference_text(original)
    if replacements and updated != original:
        path.write_text(updated, encoding="utf-8")
    return replacements


def migrate_album_page_references(photos_root: str | Path) -> dict[str, int]:
    root = Path(photos_root)
    if not root.exists():
        raise FileNotFoundError(f"Photo albums root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Photo albums root is not a directory: {root}")

    files_scanned = 0
    files_changed = 0
    replacements = 0
    for sidecar_path in sorted(root.rglob("*.xmp")):
        files_scanned += 1
        file_replacements = migrate_sidecar_page_references(sidecar_path)
        if file_replacements:
            files_changed += 1
            replacements += file_replacements
    return {
        "files_scanned": files_scanned,
        "files_changed": files_changed,
        "replacements": replacements,
    }


def find_sidecars_with_view_references(photos_root: str | Path) -> list[Path]:
    root = Path(photos_root)
    matches: list[Path] = []
    for sidecar_path in sorted(root.rglob("*.xmp")):
        if "_View" not in sidecar_path.read_text(encoding="utf-8"):
            continue
        matches.append(sidecar_path)
    return matches
