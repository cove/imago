"""Tests for _docling_parser: parse_doctag_response, merge logic, paragraph association."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib._docling_parser import (
    _associate_paragraphs,
    _merge_overlapping,
    parse_doctag_response,
)
from photoalbums.lib.ai_view_regions import RegionResult


IMG_W = 1000
IMG_H = 800


def _make_region(index: int, x: int, y: int, w: int, h: int, caption_hint: str = "") -> RegionResult:
    return RegionResult(index=index, x=x, y=y, width=w, height=h, caption_hint=caption_hint)


class TestParseDoctag(unittest.TestCase):
    """9a.1: parse_doctag_response with four <picture> elements, one with <caption>."""

    def _doctag(self, pictures: list[str]) -> str:
        body = "\n".join(pictures)
        return f"<doctag>{body}</doctag>"

    def _picture(self, top: int, left: int, bottom: int, right: int, caption: str | None = None) -> str:
        cap_xml = f"<caption>{caption}</caption>" if caption is not None else ""
        return f"<picture><loc_{top}><loc_{left}><loc_{bottom}><loc_{right}>{cap_xml}</picture>"

    def test_four_pictures_one_with_caption(self):
        xml = self._doctag([
            self._picture(68, 21, 250, 199),
            self._picture(10, 300, 200, 490, caption="Summer 1962"),
            self._picture(270, 21, 450, 199),
            self._picture(270, 300, 450, 490),
        ])
        regions = parse_doctag_response(xml, IMG_W, IMG_H)
        # Should produce 4 regions (no overlaps to merge)
        self.assertEqual(len(regions), 4)
        captions = [r.caption_hint for r in regions]
        # Exactly one region has "Summer 1962"
        self.assertEqual(captions.count("Summer 1962"), 1)
        # The rest have empty caption_hint
        empty_count = sum(1 for c in captions if not c)
        self.assertEqual(empty_count, 3)

    def test_pixel_conversion(self):
        # top=100, left=50, bottom=300, right=250  on 500-scale
        # IMG_W=1000, IMG_H=800
        # left_px = round(50/500*1000) = 100
        # top_px  = round(100/500*800) = 160
        # right_px = round(250/500*1000) = 500
        # bottom_px = round(300/500*800) = 480
        xml = self._doctag([self._picture(100, 50, 300, 250)])
        regions = parse_doctag_response(xml, IMG_W, IMG_H)
        self.assertEqual(len(regions), 1)
        r = regions[0]
        self.assertEqual(r.x, 100)
        self.assertEqual(r.y, 160)
        self.assertEqual(r.width, 400)   # 500 - 100
        self.assertEqual(r.height, 320)  # 480 - 160

    def test_missing_doctag(self):
        regions = parse_doctag_response("hello world no tags here", IMG_W, IMG_H)
        self.assertEqual(regions, [])

    def test_no_pictures(self):
        xml = "<doctag><paragraph><loc_10><loc_10><loc_50><loc_200>some text</paragraph></doctag>"
        regions = parse_doctag_response(xml, IMG_W, IMG_H)
        self.assertEqual(regions, [])

    def test_caption_hint_extracted(self):
        xml = self._doctag([self._picture(0, 0, 250, 500, caption="Family portrait")])
        regions = parse_doctag_response(xml, IMG_W, IMG_H)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].caption_hint, "Family portrait")

    def test_indices_sequential(self):
        xml = self._doctag([
            self._picture(0, 0, 100, 200),
            self._picture(0, 300, 100, 500),
            self._picture(200, 0, 300, 200),
        ])
        regions = parse_doctag_response(xml, IMG_W, IMG_H)
        indices = [r.index for r in regions]
        self.assertEqual(indices, list(range(len(regions))))


class TestMergeOverlapping(unittest.TestCase):
    """9a.2: merge logic — >15% overlap, ≤5% unchanged, three-way chain."""

    def test_two_regions_high_overlap_merged(self):
        # Region A: (0, 0, 100, 100) area=10000
        # Region B: (50, 50, 100, 100) — intersection (50,50,50,50)=2500, smaller_area=10000, frac=0.25 > 0.15
        a = _make_region(0, 0, 0, 100, 100)
        b = _make_region(1, 50, 50, 100, 100)
        merged = _merge_overlapping([a, b])
        self.assertEqual(len(merged), 1)
        r = merged[0]
        # Union: x=0, y=0, w=150, h=150
        self.assertEqual(r.x, 0)
        self.assertEqual(r.y, 0)
        self.assertEqual(r.width, 150)
        self.assertEqual(r.height, 150)
        self.assertEqual(r.caption_hint, "")

    def test_two_regions_low_overlap_unchanged(self):
        # A: (0, 0, 100, 100), B: (98, 0, 100, 100)
        # intersection: (98,0,2,100)=200, smaller_area=10000, frac=0.02 ≤ 0.05
        a = _make_region(0, 0, 0, 100, 100)
        b = _make_region(1, 98, 0, 100, 100)
        merged = _merge_overlapping([a, b])
        self.assertEqual(len(merged), 2)

    def test_three_way_chain_all_merged(self):
        # A overlaps B, B overlaps C — all three should merge to one
        # A: (0, 0, 100, 100)
        # B: (60, 0, 100, 100) — A∩B=(60,0,40,100)=4000, smaller=10000, 0.4>0.15 → merge A+B → (0,0,160,100)
        # C: (120, 0, 100, 100) — merged(0,0,160,100)∩C=(120,0,40,100)=4000, smaller(C)=10000, 0.4>0.15 → merge
        a = _make_region(0, 0, 0, 100, 100)
        b = _make_region(1, 60, 0, 100, 100)
        c = _make_region(2, 120, 0, 100, 100)
        merged = _merge_overlapping([a, b, c])
        self.assertEqual(len(merged), 1)
        r = merged[0]
        self.assertEqual(r.x, 0)
        self.assertEqual(r.y, 0)
        self.assertEqual(r.width, 220)
        self.assertEqual(r.height, 100)

    def test_merged_region_loses_caption_hint(self):
        a = _make_region(0, 0, 0, 100, 100, caption_hint="photo A")
        b = _make_region(1, 50, 50, 100, 100, caption_hint="photo B")
        merged = _merge_overlapping([a, b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].caption_hint, "")

    def test_reindexed_after_merge(self):
        a = _make_region(0, 0, 0, 100, 100)
        b = _make_region(1, 50, 50, 100, 100)
        c = _make_region(2, 400, 0, 100, 100)
        # A and B merge, C stays. Result should have indices 0 and 1.
        merged = _merge_overlapping([a, b, c])
        self.assertEqual(len(merged), 2)
        self.assertEqual([r.index for r in merged], [0, 1])


class TestAssociateParagraphs(unittest.TestCase):
    """9a.3: paragraph caption association rules."""

    def test_embedded_caption_takes_priority(self):
        # Region has caption_hint already — paragraph nearby should NOT override it
        region = _make_region(0, 0, 0, 200, 200, caption_hint="Embedded caption")
        para = {"x": 0, "y": 205, "width": 200, "height": 20, "text": "Paragraph text"}
        result = _associate_paragraphs([region], [para])
        self.assertEqual(result[0].caption_hint, "Embedded caption")

    def test_nearby_paragraph_fills_empty_caption(self):
        # Region at y=0..200, width=200. Paragraph just below at y=205..225
        region = _make_region(0, 0, 0, 200, 200)
        para = {"x": 10, "y": 205, "width": 180, "height": 20, "text": "Below photo"}
        result = _associate_paragraphs([region], [para])
        self.assertEqual(result[0].caption_hint, "Below photo")

    def test_centered_paragraph_broadcasts_to_all(self):
        # Two regions side by side. Paragraph centered, not adjacent to either.
        # Page width ~ 600 (0+300 + 300+300), middle third = 200..400
        # Paragraph at x=250, y=500 (far below both regions at y=0..200)
        r1 = _make_region(0, 0, 0, 280, 200)
        r2 = _make_region(1, 310, 0, 280, 200)
        # Centred paragraph, no adjacency (regions end at y=200, para at y=500 — too far)
        para = {"x": 250, "y": 500, "width": 100, "height": 20, "text": "Center caption"}
        result = _associate_paragraphs([r1, r2], [para])
        # Because the paragraph is far (>1 line height = 20px) from regions, it's not adjacent
        # But it's centred → broadcast
        captions = [r.caption_hint for r in result]
        self.assertTrue(all(c == "Center caption" for c in captions))

    def test_paragraph_outside_threshold_leaves_empty(self):
        # Region at y=0..100, x=0..200.
        # Paragraph is NOT adjacent (too far: gap=200 >> line_h=20) and NOT centred
        # (para centre_x = 10+90/2=55, page_width≈200, middle third ≈ 66..133 — 55 < 66).
        region = _make_region(0, 0, 0, 200, 100)
        para = {"x": 0, "y": 300, "width": 90, "height": 20, "text": "Distant text"}
        result = _associate_paragraphs([region], [para])
        self.assertEqual(result[0].caption_hint, "")

    def test_paragraph_adjacent_to_multiple_sets_ambiguous(self):
        # Two regions side by side. Paragraph spans both horizontally and is adjacent to both.
        r1 = _make_region(0, 0, 0, 200, 200)
        r2 = _make_region(1, 210, 0, 200, 200)
        # Paragraph spans from x=0..410, just below at y=205..225
        para = {"x": 0, "y": 205, "width": 410, "height": 20, "text": "Shared caption"}
        result = _associate_paragraphs([r1, r2], [para])
        # Both get the caption
        self.assertTrue(all(r.caption_hint == "Shared caption" for r in result))


if __name__ == "__main__":
    unittest.main()
