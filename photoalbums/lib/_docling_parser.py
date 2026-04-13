"""Parse <doctag> XML responses from the granite-docling model into RegionResult objects.

The granite-docling model returns a <doctag>…</doctag> XML document where each
<picture> element represents a detected photo region.  Coordinates are given as
four consecutive <loc_N> child tags in order: top, left, bottom, right on a
0–500 normalised scale (500 = full image dimension).

After parsing, overlapping pictures (>15% of the smaller area) are merged
iteratively.  Then <paragraph> elements are examined to fill in caption_hint
for any region that has no embedded <caption>.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

log = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_RegionResult = None


def _get_region_result():
    global _RegionResult
    if _RegionResult is None:
        from .ai_view_regions import RegionResult  # pylint: disable=import-outside-toplevel
        _RegionResult = RegionResult
    return _RegionResult


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_LOC_RE = re.compile(r"<loc_(\d+)>")


def _parse_loc_tags(element_text: str) -> list[int]:
    """Return the integer values of all <loc_N> tags found in element_text."""
    return [int(m.group(1)) for m in _LOC_RE.finditer(element_text)]


def _inner_xml(element_text: str) -> str:
    """Return the content between the first opening and last closing tag."""
    start = element_text.find(">")
    end = element_text.rfind("<")
    if start < 0 or end <= start:
        return ""
    return element_text[start + 1 : end]


def _child_text(element_text: str, tag: str) -> str | None:
    """Return the text content of the first <tag>…</tag> child, or None."""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = element_text.find(open_tag)
    if start < 0:
        return None
    end = element_text.find(close_tag, start)
    if end < 0:
        return None
    return element_text[start + len(open_tag) : end].strip()


def _find_elements(xml_text: str, tag: str) -> list[str]:
    """Return a list of raw strings for each <tag>…</tag> block in xml_text."""
    elements = []
    close_tag = f"</{tag}>"
    search_start = 0
    while True:
        open_start = xml_text.find(f"<{tag}", search_start)
        if open_start < 0:
            break
        close_start = xml_text.find(close_tag, open_start)
        if close_start < 0:
            break
        elements.append(xml_text[open_start : close_start + len(close_tag)])
        search_start = close_start + len(close_tag)
    return elements


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def _loc_to_pixel(loc: int, dimension: int) -> int:
    return round(loc / 500 * dimension)


# ---------------------------------------------------------------------------
# Overlap / merge helpers
# ---------------------------------------------------------------------------


def _intersection_area(ax: int, ay: int, aw: int, ah: int, bx: int, by: int, bw: int, bh: int) -> int:
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _union_box(ax: int, ay: int, aw: int, ah: int, bx: int, by: int, bw: int, bh: int) -> tuple[int, int, int, int]:
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def _merge_overlapping(regions) -> list:
    """Iteratively merge region pairs that overlap by >15% of the smaller area.

    Returns a new list of RegionResult objects with consecutive indices.
    Merged regions carry no caption_hint.
    """
    RegionResult = _get_region_result()
    working = list(regions)
    changed = True
    while changed:
        changed = False
        merged: list = []
        used = [False] * len(working)
        for i in range(len(working)):
            if used[i]:
                continue
            current = working[i]
            for j in range(i + 1, len(working)):
                if used[j]:
                    continue
                other = working[j]
                inter = _intersection_area(
                    current.x, current.y, current.width, current.height,
                    other.x, other.y, other.width, other.height,
                )
                smaller_area = min(current.width * current.height, other.width * other.height)
                if smaller_area > 0 and inter / smaller_area > 0.15:
                    ux, uy, uw, uh = _union_box(
                        current.x, current.y, current.width, current.height,
                        other.x, other.y, other.width, other.height,
                    )
                    current = RegionResult(
                        index=current.index,
                        x=ux,
                        y=uy,
                        width=uw,
                        height=uh,
                        confidence=max(current.confidence, other.confidence),
                        caption_hint="",
                    )
                    used[j] = True
                    changed = True
            merged.append(current)
            used[i] = True
        working = merged

    # Re-index
    return [replace(r, index=i) for i, r in enumerate(working)]


# ---------------------------------------------------------------------------
# Paragraph caption association
# ---------------------------------------------------------------------------


def _associate_paragraphs(regions, paragraphs: list[dict]) -> list:
    """Associate <paragraph> elements with regions as caption_hint fallback.

    paragraphs is a list of dicts: {x, y, width, height, text}.

    Rules (in order):
    - Regions that already have caption_hint (from embedded <caption>) are skipped.
    - A paragraph is "adjacent" to a region if its horizontal span overlaps the
      region AND it sits within one text-line height (paragraph height) of the
      region boundary.
    - If a paragraph is adjacent to exactly one region with empty caption_hint,
      assign it to that region.
    - If a paragraph is adjacent to two or more regions equally (or centred in the
      middle third of the page and adjacent to none), broadcast to all with
      caption_ambiguous=True.
    - A centred paragraph (horizontal centre within the middle third of the page)
      that is not adjacent to any single region is broadcast to all with
      caption_ambiguous=True.
    """
    if not paragraphs or not regions:
        return list(regions)

    RegionResult = _get_region_result()

    # Determine page width from region extents (approximate)
    page_width = max((r.x + r.width) for r in regions) if regions else 1
    middle_third_left = page_width / 3
    middle_third_right = 2 * page_width / 3

    # Work with a dict for easy mutation of caption_hint
    captions: dict[int, str] = {r.index: r.caption_hint for r in regions}
    ambiguous: set[int] = set()

    def _horiz_overlap(rx: int, rw: int, px: int, pw: int) -> bool:
        return rx < px + pw and rx + rw > px

    def _vertically_adjacent(ry: int, rh: int, py: int, ph: int) -> bool:
        """True if paragraph is within one text-line (ph) of the region boundary."""
        region_bottom = ry + rh
        para_bottom = py + ph
        line_h = max(ph, 1)
        # Above region: paragraph bottom is close to region top
        if py < ry and region_bottom > py:
            # overlaps vertically — not "below/above"
            pass
        # Below region
        gap_below = py - region_bottom
        # Above region
        gap_above = ry - para_bottom
        return gap_below >= 0 and gap_below <= line_h or gap_above >= 0 and gap_above <= line_h

    for para in paragraphs:
        px = int(para.get("x", 0))
        py = int(para.get("y", 0))
        pw = int(para.get("width", 0))
        ph = int(para.get("height", 0))
        text = str(para.get("text", "")).strip()
        if not text:
            continue

        para_centre_x = px + pw / 2
        is_centred = middle_third_left <= para_centre_x <= middle_third_right

        # Find adjacent regions with empty caption_hint
        adjacent: list[int] = []
        for r in regions:
            if captions.get(r.index):
                continue  # already has caption
            if _horiz_overlap(r.x, r.width, px, pw) and _vertically_adjacent(r.y, r.height, py, ph):
                adjacent.append(r.index)

        if len(adjacent) == 1:
            captions[adjacent[0]] = text
        elif len(adjacent) > 1:
            for idx in adjacent:
                captions[idx] = text
                ambiguous.add(idx)
        elif is_centred and len(adjacent) == 0:
            # Broadcast to all regions
            for r in regions:
                if not captions.get(r.index):
                    captions[r.index] = text
                    ambiguous.add(r.index)

    return [replace(r, caption_hint=captions.get(r.index, r.caption_hint)) for r in regions]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_doctag_response(content: str, img_w: int, img_h: int) -> list:
    """Parse a <doctag> XML response into a list of RegionResult objects.

    Steps:
    1. Find all <picture> elements → initial region list
    2. Merge overlapping regions (>15% of smaller area)
    3. Associate <paragraph> elements as caption fallback

    Returns [] and logs a WARNING if <doctag> is missing or contains no <picture>.
    """
    RegionResult = _get_region_result()

    text = str(content or "").strip()

    if "<doctag>" not in text and "<doctag " not in text:
        log.warning("parse_doctag_response: no <doctag> element in response")
        return []

    # Extract doctag content
    doctag_elements = _find_elements(text, "doctag")
    if not doctag_elements:
        log.warning("parse_doctag_response: could not parse <doctag> element")
        return []
    doctag_body = doctag_elements[0]

    # Parse <picture> elements
    picture_elements = _find_elements(doctag_body, "picture")
    if not picture_elements:
        log.warning("parse_doctag_response: no <picture> elements found in <doctag>")
        return []

    regions: list = []
    for pic_xml in picture_elements:
        inner = _inner_xml(pic_xml)
        locs = _parse_loc_tags(inner)
        if len(locs) < 4:
            log.warning("parse_doctag_response: <picture> has fewer than 4 <loc_N> tags, skipping: %r", pic_xml[:120])
            continue

        loc_top, loc_left, loc_bottom, loc_right = locs[0], locs[1], locs[2], locs[3]

        left_px = _loc_to_pixel(loc_left, img_w)
        top_px = _loc_to_pixel(loc_top, img_h)
        right_px = _loc_to_pixel(loc_right, img_w)
        bottom_px = _loc_to_pixel(loc_bottom, img_h)

        x = left_px
        y = top_px
        width = max(1, right_px - left_px)
        height = max(1, bottom_px - top_px)

        caption_hint = _child_text(inner, "caption") or ""

        regions.append(RegionResult(
            index=len(regions),
            x=x,
            y=y,
            width=width,
            height=height,
            caption_hint=caption_hint,
        ))

    if not regions:
        log.warning("parse_doctag_response: no valid regions parsed from <picture> elements")
        return []

    # Merge overlapping regions
    regions = _merge_overlapping(regions)

    # Parse <paragraph> elements for caption fallback
    paragraph_elements = _find_elements(doctag_body, "paragraph")
    paragraphs: list[dict] = []
    for para_xml in paragraph_elements:
        inner = _inner_xml(para_xml)
        locs = _parse_loc_tags(inner)
        if len(locs) < 4:
            continue
        loc_top_p, loc_left_p, loc_bottom_p, loc_right_p = locs[0], locs[1], locs[2], locs[3]
        px = _loc_to_pixel(loc_left_p, img_w)
        py = _loc_to_pixel(loc_top_p, img_h)
        pw = max(1, _loc_to_pixel(loc_right_p, img_w) - px)
        ph = max(1, _loc_to_pixel(loc_bottom_p, img_h) - py)
        # Text content: everything after the loc tags
        inner_stripped = _LOC_RE.sub("", inner).strip()
        if inner_stripped:
            paragraphs.append({"x": px, "y": py, "width": pw, "height": ph, "text": inner_stripped})

    if paragraphs:
        regions = _associate_paragraphs(regions, paragraphs)

    return regions
