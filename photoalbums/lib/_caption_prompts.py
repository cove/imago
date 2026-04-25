from __future__ import annotations

from pathlib import Path

from .ai_prompt_assets import PromptAsset, load_prompt, prompt_metadata


def _position_label(cx: float, cy: float) -> str:
    v = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "centre")
    h = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
    if v == "centre" and h == "centre":
        return "centre"
    if v == "centre":
        return h
    if h == "centre":
        return v
    return f"{v}-{h}"


def _join_assets(assets: list[PromptAsset]) -> str:
    return "\n".join(asset.rendered for asset in assets if asset.rendered).strip()


def _load_caption_assets(*, source_path: str | Path | None, context_ocr_text: str = "") -> list[PromptAsset]:
    del source_path, context_ocr_text
    return []


def caption_prompt_metadata(*, source_path: str | Path | None, context_ocr_text: str = "") -> dict[str, object]:
    return prompt_metadata(*_load_caption_assets(source_path=source_path, context_ocr_text=context_ocr_text))


def people_count_prompt_metadata() -> dict[str, object]:
    return prompt_metadata(
        load_prompt("ai-index/people-count/user.md"),
        load_prompt("ai-index/people-count/output.md"),
    )


def location_prompt_metadata() -> dict[str, object]:
    return {}


def location_shown_prompt_metadata() -> dict[str, object]:
    return {}


def location_queries_prompt_metadata() -> dict[str, object]:
    return {}


def _build_local_prompt(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    people_positions: dict[str, str] | None = None,
    context_ocr_text: str = "",
) -> str:
    del people, objects, ocr_text, album_title, printed_album_title, people_positions
    return _join_assets(_load_caption_assets(source_path=source_path, context_ocr_text=context_ocr_text))


def _build_people_count_prompt(
    *,
    people: list[str],
    objects: list[str],
    ocr_text: str,
    source_path: str | Path | None = None,
    album_title: str = "",
    printed_album_title: str = "",
    people_positions: dict[str, str] | None = None,
) -> str:
    del people, objects, ocr_text, source_path, album_title, printed_album_title, people_positions
    return _join_assets(
        [
            load_prompt("ai-index/people-count/user.md"),
            load_prompt("ai-index/people-count/output.md"),
        ]
    )


def _build_location_prompt_for_output(
    output_path: str,
    *,
    ocr_text: str = "",
    album_title: str = "",
    printed_album_title: str = "",
) -> str:
    del output_path, ocr_text, album_title, printed_album_title
    return ""


def _build_location_prompt(*, ocr_text: str = "", album_title: str = "", printed_album_title: str = "") -> str:
    return _build_location_prompt_for_output(
        "ai-index/locations/output-location.md",
        ocr_text=ocr_text,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )


def _build_location_shown_prompt(*, ocr_text: str = "", album_title: str = "", printed_album_title: str = "") -> str:
    return _build_location_prompt_for_output(
        "ai-index/locations/output-shown.md",
        ocr_text=ocr_text,
        album_title=album_title,
        printed_album_title=printed_album_title,
    )


def _build_location_queries_prompt(
    *,
    caption_text: str = "",
    ocr_text: str = "",
    album_title: str = "",
    printed_album_title: str = "",
) -> str:
    del ocr_text, album_title, printed_album_title
    if caption_text:
        return f"Caption: {caption_text.strip()}"
    return ""
