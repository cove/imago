from __future__ import annotations


def clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def clean_lines(value: str) -> str:
    lines = str(value or "").splitlines()
    cleaned = [" ".join(line.split()) for line in lines]
    return "\n".join(line for line in cleaned if line)


def join_human(values: list[str]) -> str:
    clean = [str(item or "").strip() for item in values if str(item or "").strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


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
