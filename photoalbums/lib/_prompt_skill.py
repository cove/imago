from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

BASE_SKILL_FILE = Path(__file__).parent.parent.parent / "skills" / "CORDELL_PHOTO_ALBUMS" / "SKILL.md"
TRAVEL_SKILL_FILE = Path(__file__).parent.parent.parent / "skills" / "CORDELL_PHOTO_ALBUMS_TRAVEL" / "SKILL.md"
FAMILY_SKILL_FILE = Path(__file__).parent.parent.parent / "skills" / "CORDELL_PHOTO_ALBUMS_FAMILY" / "SKILL.md"


def parse_skill(path: Path) -> dict[str, list[str]]:
    """Parse a skill file into {section_name: [lines]}."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.rstrip()
            if stripped.startswith("## ") and not stripped.startswith("### "):
                current = stripped[3:].strip()
                sections.setdefault(current, [])
            elif current is not None and stripped.strip() and not stripped.strip().startswith("#"):
                sections[current].append(stripped.strip())
    return sections


@lru_cache(maxsize=1)
def base_skill() -> dict[str, list[str]]:
    return parse_skill(BASE_SKILL_FILE)


@lru_cache(maxsize=1)
def travel_skill() -> dict[str, list[str]]:
    return parse_skill(TRAVEL_SKILL_FILE)


@lru_cache(maxsize=1)
def family_skill() -> dict[str, list[str]]:
    return parse_skill(FAMILY_SKILL_FILE)


def render_line(text: str, variables: dict[str, str]) -> tuple[str, bool]:
    """Substitute {var} patterns; drop the line if any variable is empty."""
    drop = False

    def replace(match: re.Match) -> str:
        nonlocal drop
        value = variables.get(match.group(1), "")
        if not value:
            drop = True
        return value

    return re.sub(r"\{(\w+)\}", replace, text), drop


def section(*names: str, **kwargs: str) -> list[str]:
    return section_from(base_skill(), *names, **kwargs)


def section_from(skill: dict[str, list[str]], *names: str, **kwargs: str) -> list[str]:
    lines: list[str] = []
    for name in names:
        for raw in skill.get(name, []):
            rendered, drop = render_line(raw, kwargs)
            if not drop:
                lines.append(rendered)
    return lines


def section_text(*names: str, **kwargs: str) -> str:
    return "\n".join(section(*names, **kwargs))


def required_section_text(*names: str, **kwargs: str) -> str:
    text = section_text(*names, **kwargs).strip()
    if text:
        return text
    joined = ", ".join(names) or "<unnamed>"
    raise RuntimeError(f"Missing required skill section text: {joined}")
