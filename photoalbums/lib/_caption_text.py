from __future__ import annotations


def clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def clean_lines(value: str) -> str:
    lines = str(value or "").splitlines()
    cleaned = [" ".join(line.split()) for line in lines]
    return "\n".join(line for line in cleaned if line)


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
