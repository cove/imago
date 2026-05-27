"""Tests for face_region_reconciler."""

from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.face_region_reconciler import (
    FaceCluster,
    PendingWrite,
    ProjectedFaceBox,
    ReconcileResult,
    SourceFaceBox,
    _archive_max_derived,
    _crop_face_to_page,
    _crop_region_in_page,
    _derived_number_from_xmp,
    _iou,
    _page_face_to_crop,
    _page_prefix,
    cluster_face_boxes,
    find_archive_scans_for_page,
    find_crop_xmps_for_page,
    merge_iptc_face_box,
    plan_backfill,
    project_sources_to_page,
    read_iptc_face_boxes,
    read_page_photo_regions,
    reconcile_page,
)

# ---------------------------------------------------------------------------
# Helpers for building test XMP content
# ---------------------------------------------------------------------------

_IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_MWGRS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
_STAREA_NS = "http://ns.adobe.com/xap/1.0/sType/Area#"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _write_xmp(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _face_xmp(faces: list[dict]) -> str:
    """Build a minimal XMP with IPTC face regions."""
    lines = [
        "<?xml version='1.0' encoding='utf-8'?>",
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>",
        "  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>",
        "    <rdf:Description rdf:about=''",
        "      xmlns:Iptc4xmpExt='http://iptc.org/std/Iptc4xmpExt/2008-02-29/'>",
        "      <Iptc4xmpExt:ImageRegion>",
        "        <rdf:Bag>",
    ]
    for i, f in enumerate(faces, 1):
        name = f.get("name", "")
        rx, ry, rw, rh = f["rx"], f["ry"], f["rw"], f["rh"]
        lines += [
            "          <rdf:li rdf:parseType='Resource'>",
            "            <Iptc4xmpExt:RegionBoundary rdf:parseType='Resource'>",
            "              <Iptc4xmpExt:rbShape>rectangle</Iptc4xmpExt:rbShape>",
            "              <Iptc4xmpExt:rbUnit>relative</Iptc4xmpExt:rbUnit>",
            f"              <Iptc4xmpExt:rbX>{rx:.6f}</Iptc4xmpExt:rbX>",
            f"              <Iptc4xmpExt:rbY>{ry:.6f}</Iptc4xmpExt:rbY>",
            f"              <Iptc4xmpExt:rbW>{rw:.6f}</Iptc4xmpExt:rbW>",
            f"              <Iptc4xmpExt:rbH>{rh:.6f}</Iptc4xmpExt:rbH>",
            "            </Iptc4xmpExt:RegionBoundary>",
            "            <Iptc4xmpExt:RCtype>face-identified</Iptc4xmpExt:RCtype>",
            f"            <Iptc4xmpExt:rId>face-{i}</Iptc4xmpExt:rId>",
        ]
        if name:
            lines += [
                "            <Iptc4xmpExt:Name>",
                "              <rdf:Alt>",
                f"                <rdf:li xml:lang='x-default'>{name}</rdf:li>",
                "              </rdf:Alt>",
                "            </Iptc4xmpExt:Name>",
            ]
        lines.append("          </rdf:li>")
    lines += [
        "        </rdf:Bag>",
        "      </Iptc4xmpExt:ImageRegion>",
        "    </rdf:Description>",
        "  </rdf:RDF>",
        "</x:xmpmeta>",
    ]
    return "\n".join(lines)


def _page_region_xmp(regions: list[dict]) -> str:
    """Build a minimal page XMP with mwg-rs:RegionList Photo regions."""
    lines = [
        "<?xml version='1.0' encoding='utf-8'?>",
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>",
        "  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>",
        "    <rdf:Description rdf:about=''",
        "      xmlns:mwg-rs='http://www.metadataworkinggroup.com/schemas/regions/'",
        "      xmlns:stArea='http://ns.adobe.com/xap/1.0/sType/Area#'>",
        "      <mwg-rs:Regions rdf:parseType='Resource'>",
        "        <mwg-rs:AppliedToDimensions rdf:parseType='Resource'",
        "          stArea:w='1000' stArea:h='800' stArea:unit='pixel'/>",
        "        <mwg-rs:RegionList>",
        "          <rdf:Bag>",
    ]
    for r in regions:
        cx, cy, nw, nh = r["cx"], r["cy"], r["nw"], r["nh"]
        lines += [
            "            <rdf:li rdf:parseType='Resource'",
            "              mwg-rs:Type='Photo'",
            f"              stArea:x='{cx}' stArea:y='{cy}'",
            f"              stArea:w='{nw}' stArea:h='{nh}' stArea:unit='normalized'",
            "              mwg-rs:Name='photo_1'/>",
        ]
    lines += [
        "          </rdf:Bag>",
        "        </mwg-rs:RegionList>",
        "      </mwg-rs:Regions>",
        "    </rdf:Description>",
        "  </rdf:RDF>",
        "</x:xmpmeta>",
    ]
    return "\n".join(lines)


def _empty_xmp() -> str:
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<rdf:Description rdf:about=''/>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
    )


# ---------------------------------------------------------------------------
# Unit tests: geometry helpers
# ---------------------------------------------------------------------------


class TestIou(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(_iou((0.1, 0.1, 0.2, 0.2), (0.1, 0.1, 0.2, 0.2)), 1.0)

    def test_no_overlap(self):
        self.assertEqual(_iou((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)), 0.0)

    def test_partial_overlap(self):
        # Two squares offset by half their size → overlap = 0.5*0.5 each quadrant
        iou = _iou((0.0, 0.0, 0.4, 0.4), (0.2, 0.2, 0.4, 0.4))
        # intersection = 0.2*0.2=0.04, union = 0.16+0.16-0.04=0.28
        self.assertAlmostEqual(iou, 0.04 / 0.28, places=5)

    def test_touching_edges(self):
        self.assertEqual(_iou((0.0, 0.0, 0.2, 0.2), (0.2, 0.0, 0.2, 0.2)), 0.0)


class TestCropProjection(unittest.TestCase):
    def test_full_crop_face_is_full_page_region(self):
        region = {"cx": 0.5, "cy": 0.5, "nw": 0.4, "nh": 0.6}
        face = {"rx": 0.0, "ry": 0.0, "rw": 1.0, "rh": 1.0}
        px, py, pw, ph = _crop_face_to_page(face, region)
        # crop origin: ox=0.3, oy=0.2; size 0.4×0.6
        self.assertAlmostEqual(px, 0.3)
        self.assertAlmostEqual(py, 0.2)
        self.assertAlmostEqual(pw, 0.4)
        self.assertAlmostEqual(ph, 0.6)

    def test_centered_face_maps_correctly(self):
        region = {"cx": 0.5, "cy": 0.5, "nw": 0.4, "nh": 0.4}
        # face at center of crop, half the size
        face = {"rx": 0.25, "ry": 0.25, "rw": 0.5, "rh": 0.5}
        px, py, pw, ph = _crop_face_to_page(face, region)
        # ox=0.3, oy=0.3; face: px=0.3+0.25*0.4=0.4, py=0.4, pw=0.2, ph=0.2
        self.assertAlmostEqual(px, 0.4)
        self.assertAlmostEqual(py, 0.4)
        self.assertAlmostEqual(pw, 0.2)
        self.assertAlmostEqual(ph, 0.2)

    def test_roundtrip(self):
        region = {"cx": 0.6, "cy": 0.4, "nw": 0.3, "nh": 0.5}
        face = {"rx": 0.1, "ry": 0.2, "rw": 0.4, "rh": 0.3}
        px, py, pw, ph = _crop_face_to_page(face, region)
        crx, cry, crw, crh = _page_face_to_crop(px, py, pw, ph, region)
        self.assertAlmostEqual(crx, face["rx"], places=10)
        self.assertAlmostEqual(cry, face["ry"], places=10)
        self.assertAlmostEqual(crw, face["rw"], places=10)
        self.assertAlmostEqual(crh, face["rh"], places=10)


# ---------------------------------------------------------------------------
# Unit tests: IPTC face reading
# ---------------------------------------------------------------------------


class TestReadIptcFaceBoxes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_single_face(self):
        xmp = self.tmpdir / "test.xmp"
        _write_xmp(xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.2, "rw": 0.3, "rh": 0.4}]))
        boxes = read_iptc_face_boxes(xmp)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["name"], "Alice")
        self.assertAlmostEqual(boxes[0]["rx"], 0.1)
        self.assertAlmostEqual(boxes[0]["ry"], 0.2)
        self.assertAlmostEqual(boxes[0]["rw"], 0.3)
        self.assertAlmostEqual(boxes[0]["rh"], 0.4)

    def test_reads_multiple_faces(self):
        xmp = self.tmpdir / "test.xmp"
        faces = [
            {"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.1, "rh": 0.1},
            {"name": "Bob", "rx": 0.5, "ry": 0.5, "rw": 0.1, "rh": 0.1},
        ]
        _write_xmp(xmp, _face_xmp(faces))
        boxes = read_iptc_face_boxes(xmp)
        self.assertEqual(len(boxes), 2)
        self.assertEqual({b["name"] for b in boxes}, {"Alice", "Bob"})

    def test_missing_file_returns_empty(self):
        self.assertEqual(read_iptc_face_boxes(self.tmpdir / "missing.xmp"), [])

    def test_empty_xmp_returns_empty(self):
        xmp = self.tmpdir / "empty.xmp"
        _write_xmp(xmp, _empty_xmp())
        self.assertEqual(read_iptc_face_boxes(xmp), [])

    def test_face_without_name(self):
        xmp = self.tmpdir / "test.xmp"
        _write_xmp(xmp, _face_xmp([{"rx": 0.1, "ry": 0.2, "rw": 0.3, "rh": 0.4}]))
        boxes = read_iptc_face_boxes(xmp)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["name"], "")


# ---------------------------------------------------------------------------
# Unit tests: page photo regions reading
# ---------------------------------------------------------------------------


class TestReadPagePhotoRegions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_single_region(self):
        xmp = self.tmpdir / "page.xmp"
        _write_xmp(xmp, _page_region_xmp([{"cx": 0.5, "cy": 0.5, "nw": 0.4, "nh": 0.6}]))
        regions = read_page_photo_regions(xmp)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0]["cx"], 0.5)
        self.assertAlmostEqual(regions[0]["nw"], 0.4)

    def test_reads_multiple_regions_in_order(self):
        xmp = self.tmpdir / "page.xmp"
        _write_xmp(xmp, _page_region_xmp([
            {"cx": 0.3, "cy": 0.3, "nw": 0.2, "nh": 0.2},
            {"cx": 0.7, "cy": 0.7, "nw": 0.2, "nh": 0.2},
        ]))
        regions = read_page_photo_regions(xmp)
        self.assertEqual(len(regions), 2)
        self.assertAlmostEqual(regions[0]["cx"], 0.3)
        self.assertAlmostEqual(regions[1]["cx"], 0.7)

    def test_reads_exiftool_canonical_regions_from_writer(self):
        from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
        from photoalbums.lib.xmp_sidecar import write_region_list

        xmp = self.tmpdir / "page.xmp"
        write_region_list(
            xmp,
            [RegionWithCaption(RegionResult(index=0, x=0, y=0, width=400, height=300), "Caption")],
            800,
            600,
        )

        regions = read_page_photo_regions(xmp)

        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0]["cx"], 0.25)
        self.assertAlmostEqual(regions[0]["cy"], 0.25)
        self.assertAlmostEqual(regions[0]["nw"], 0.5)
        self.assertAlmostEqual(regions[0]["nh"], 0.5)

    def test_missing_file_returns_empty(self):
        self.assertEqual(read_page_photo_regions(self.tmpdir / "missing.xmp"), [])


# ---------------------------------------------------------------------------
# Unit tests: cluster_face_boxes
# ---------------------------------------------------------------------------


def _make_projected(name: str, rx: float, ry: float, rw: float, rh: float, kind: str = "page", xmp: Path | None = None) -> ProjectedFaceBox:
    src = SourceFaceBox(name=name, rx=rx, ry=ry, rw=rw, rh=rh, source_kind=kind, source_xmp=xmp or Path("/dummy.xmp"))
    return ProjectedFaceBox(name=name, page_rx=rx, page_ry=ry, page_rw=rw, page_rh=rh, confidence="high", origin=src)


class TestClusterFaceBoxes(unittest.TestCase):
    def test_identical_boxes_merge(self):
        pf1 = _make_projected("Alice", 0.1, 0.1, 0.2, 0.2)
        pf2 = _make_projected("Alice", 0.1, 0.1, 0.2, 0.2)
        clusters = cluster_face_boxes([pf1, pf2])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].names, ["Alice"])

    def test_non_overlapping_boxes_separate(self):
        pf1 = _make_projected("Alice", 0.0, 0.0, 0.2, 0.2)
        pf2 = _make_projected("Bob", 0.8, 0.8, 0.1, 0.1)
        clusters = cluster_face_boxes([pf1, pf2])
        self.assertEqual(len(clusters), 2)

    def test_conflict_detected(self):
        pf1 = _make_projected("Alice", 0.1, 0.1, 0.2, 0.2)
        pf2 = _make_projected("Bob", 0.1, 0.1, 0.2, 0.2)
        clusters = cluster_face_boxes([pf1, pf2])
        self.assertEqual(len(clusters), 1)
        self.assertTrue(clusters[0].has_conflict)
        self.assertIn("Alice", clusters[0].names)
        self.assertIn("Bob", clusters[0].names)

    def test_empty_name_does_not_conflict(self):
        pf1 = _make_projected("Alice", 0.1, 0.1, 0.2, 0.2)
        pf2 = _make_projected("", 0.1, 0.1, 0.2, 0.2)
        clusters = cluster_face_boxes([pf1, pf2])
        self.assertEqual(len(clusters), 1)
        self.assertFalse(clusters[0].has_conflict)
        self.assertEqual(clusters[0].names, ["Alice"])

    def test_empty_input(self):
        self.assertEqual(cluster_face_boxes([]), [])


# ---------------------------------------------------------------------------
# Unit tests: derived number parsing
# ---------------------------------------------------------------------------


class TestDerivedNumber(unittest.TestCase):
    def test_standard_crop_name(self):
        self.assertEqual(_derived_number_from_xmp(Path("Egypt_1975_B00_P26_D03-00_V.xmp")), 3)

    def test_two_digit_derived(self):
        self.assertEqual(_derived_number_from_xmp(Path("Family_2000_B01_P12_D12-01_V.xmp")), 12)

    def test_non_crop_returns_none(self):
        self.assertIsNone(_derived_number_from_xmp(Path("Egypt_1975_B00_P26_V.xmp")))


# ---------------------------------------------------------------------------
# Integration tests: merge_iptc_face_box
# ---------------------------------------------------------------------------


class TestMergeIptcFaceBox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_to_empty_xmp(self):
        xmp = self.tmpdir / "test.xmp"
        _write_xmp(xmp, _empty_xmp())
        wrote = merge_iptc_face_box(xmp, "Alice", 0.1, 0.2, 0.3, 0.4)
        self.assertTrue(wrote)
        boxes = read_iptc_face_boxes(xmp)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["name"], "Alice")

    def test_skips_duplicate(self):
        xmp = self.tmpdir / "test.xmp"
        _write_xmp(xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.2, "rw": 0.3, "rh": 0.4}]))
        wrote = merge_iptc_face_box(xmp, "Alice", 0.11, 0.21, 0.3, 0.4)
        self.assertFalse(wrote)  # high IoU with existing box

    def test_creates_new_file(self):
        xmp = self.tmpdir / "new.xmp"
        self.assertFalse(xmp.exists())
        wrote = merge_iptc_face_box(xmp, "Bob", 0.5, 0.5, 0.1, 0.1)
        self.assertTrue(wrote)
        self.assertTrue(xmp.exists())

    def test_adds_to_existing_faces(self):
        xmp = self.tmpdir / "test.xmp"
        _write_xmp(xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.1, "rh": 0.1}]))
        wrote = merge_iptc_face_box(xmp, "Bob", 0.8, 0.8, 0.1, 0.1)
        self.assertTrue(wrote)
        boxes = read_iptc_face_boxes(xmp)
        self.assertEqual(len(boxes), 2)


# ---------------------------------------------------------------------------
# Integration tests: source discovery
# ---------------------------------------------------------------------------


class TestSourceDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Create album directory structure
        self.pages_dir = self.root / "Family_2020_B01_Pages"
        self.photos_dir = self.root / "Family_2020_B01_Photos"
        self.archive_dir = self.root / "Family_2020_B01_Archive"
        self.pages_dir.mkdir()
        self.photos_dir.mkdir()
        self.archive_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _page_xmp(self) -> Path:
        return self.pages_dir / "Family_2020_B01_P05_V.xmp"

    def test_find_archive_scans(self):
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _empty_xmp())
        # Create scan TIF + XMP
        (self.archive_dir / "Family_2020_B01_P05_S01.tif").touch()
        _write_xmp(self.archive_dir / "Family_2020_B01_P05_S01.xmp", _empty_xmp())
        # Create a scan for a different page (should not be included)
        (self.archive_dir / "Family_2020_B01_P06_S01.tif").touch()
        _write_xmp(self.archive_dir / "Family_2020_B01_P06_S01.xmp", _empty_xmp())

        scans = find_archive_scans_for_page(page_xmp)
        self.assertEqual(len(scans), 1)
        self.assertIn("P05_S01.xmp", scans[0].name)

    def test_find_crop_xmps(self):
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _empty_xmp())
        # Create crop XMPs
        _write_xmp(self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp", _empty_xmp())
        _write_xmp(self.photos_dir / "Family_2020_B01_P05_D02-00_V.xmp", _empty_xmp())
        # Different page crop (should not be included)
        _write_xmp(self.photos_dir / "Family_2020_B01_P06_D01-00_V.xmp", _empty_xmp())

        crops = find_crop_xmps_for_page(page_xmp)
        self.assertEqual(len(crops), 2)

    def test_missing_dirs_return_empty(self):
        # page_xmp not in a pages dir
        xmp = self.root / "orphan.xmp"
        _write_xmp(xmp, _empty_xmp())
        self.assertEqual(find_archive_scans_for_page(xmp), [])
        self.assertEqual(find_crop_xmps_for_page(xmp), [])


# ---------------------------------------------------------------------------
# Integration tests: full reconcile_page
# ---------------------------------------------------------------------------


class TestReconcilePage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pages_dir = self.root / "Family_2020_B01_Pages"
        self.photos_dir = self.root / "Family_2020_B01_Photos"
        self.archive_dir = self.root / "Family_2020_B01_Archive"
        self.pages_dir.mkdir()
        self.photos_dir.mkdir()
        self.archive_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _page_xmp(self) -> Path:
        return self.pages_dir / "Family_2020_B01_P05_V.xmp"

    # ------------------------------------------------------------------
    # OpenCV visual-validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_patch(img_path: Path, box: dict, size: tuple[int, int] = (32, 32)):
        """Extract and resize a normalised-coordinate face box from an image.

        Returns a 32×32 float32 greyscale array, or None if the image can't be
        opened or the box falls outside the image.
        """
        import cv2

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        h, w = img.shape
        x0 = max(0, int(box["rx"] * w))
        y0 = max(0, int(box["ry"] * h))
        x1 = min(w, int((box["rx"] + box["rw"]) * w))
        y1 = min(h, int((box["ry"] + box["rh"]) * h))
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        return cv2.resize(patch, size).astype("float32")

    @staticmethod
    def _ncc(patch_a, patch_b) -> float:
        """Normalised cross-correlation in [-1, 1]; 1.0 = identical content."""
        import cv2

        result = cv2.matchTemplate(patch_a, patch_b, cv2.TM_CCOEFF_NORMED)
        return float(result[0, 0])

    def _assert_same_region(self, img_a: Path, box_a: dict, img_b: Path, box_b: dict,
                             *, msg: str = "", threshold: float = 0.90) -> None:
        """Assert the two face-box patches contain the same visual content (NCC ≥ threshold)."""
        pa = self._extract_patch(img_a, box_a)
        pb = self._extract_patch(img_b, box_b)
        self.assertIsNotNone(pa, f"could not extract patch from {img_a.name}")
        self.assertIsNotNone(pb, f"could not extract patch from {img_b.name}")
        score = self._ncc(pa, pb)
        label = msg or f"{img_a.name}@{box_a} vs {img_b.name}@{box_b}"
        self.assertGreater(score, threshold,
                           f"Face patches should show the same scene content "
                           f"(NCC={score:.3f} < {threshold}): {label}")

    # ------------------------------------------------------------------

    def _write_alignment_images(self, page_xmp: Path, scan_boxes: dict[str, tuple[int, int, int, int]]) -> None:
        import cv2
        import numpy as np

        rng = np.random.default_rng(12345)
        page = rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
        for i in range(16):
            x = 20 + (i * 17) % 270
            y = 20 + (i * 29) % 130
            cv2.circle(page, (x, y), 5 + (i % 5), (255 - i * 8, 30 + i * 11, 80 + i * 7), -1)
            cv2.putText(page, str(i), (x + 4, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        assert cv2.imwrite(str(page_xmp.with_suffix(".png")), page)
        for scan_name, (x, y, w, h) in scan_boxes.items():
            scan_path = self.archive_dir / scan_name
            assert cv2.imwrite(str(scan_path), page[y : y + h, x : x + w])

    def test_dry_run_no_writes(self):
        """dry_run=True should not modify any files."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]))
        result = reconcile_page(page_xmp, dry_run=True)
        self.assertIsInstance(result, ReconcileResult)
        # File should not be modified (Alice already in page, no other sources)
        self.assertEqual(len(result.pending_writes), 0)

    def test_face_in_page_only_no_crops(self):
        """A page with a face but no crops/scans should produce one cluster, no writes needed."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]))
        result = reconcile_page(page_xmp, dry_run=True)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].names, ["Alice"])
        self.assertFalse(result.clusters[0].has_conflict)
        self.assertEqual(len(result.pending_writes), 0)

    def test_face_in_page_backfills_to_crop_and_scan(self):
        """A page face should be written to missing crop and scan peers."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        self._write_alignment_images(page_xmp, {"Family_2020_B01_P05_S01.tif": (16, 9, 128, 72)})
        merge_iptc_face_box(page_xmp, "Alice", 0.09, 0.09, 0.12, 0.12)
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _empty_xmp())
        scan_xmp = self.archive_dir / "Family_2020_B01_P05_S01.xmp"
        _write_xmp(scan_xmp, _empty_xmp())

        result = reconcile_page(page_xmp, dry_run=True)

        writes_by_kind = {w.source_kind: w for w in result.pending_writes}
        self.assertEqual(set(writes_by_kind), {"crop", "scan"})
        self.assertEqual(writes_by_kind["crop"].xmp_path, crop_xmp)
        self.assertAlmostEqual(writes_by_kind["crop"].rx, 0.1)
        self.assertAlmostEqual(writes_by_kind["crop"].rw, 0.3)
        self.assertEqual(writes_by_kind["scan"].xmp_path, scan_xmp)
        self.assertAlmostEqual(writes_by_kind["scan"].rx, 0.1, places=2)
        self.assertAlmostEqual(writes_by_kind["scan"].rw, 0.3, places=2)

    def test_face_in_crop_backfills_to_page(self):
        """A face only in a crop should be backfilled to the page XMP."""
        page_xmp = self._page_xmp()
        # Page has a mwg-rs region with one photo crop, no face regions
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        # Crop has a face (full image = full crop = entire region)
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _face_xmp([{"name": "Alice", "rx": 0.0, "ry": 0.0, "rw": 1.0, "rh": 1.0}]))

        result = reconcile_page(page_xmp, dry_run=True)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(len(result.pending_writes), 1)
        self.assertEqual(result.pending_writes[0].source_kind, "page")
        self.assertEqual(result.pending_writes[0].name, "Alice")

    def test_face_in_crop_backfills_to_page_and_scan(self):
        """A crop face should be projected to page space and shared with the scan."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        self._write_alignment_images(page_xmp, {"Family_2020_B01_P05_S01.tif": (16, 9, 128, 72)})
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]))
        scan_xmp = self.archive_dir / "Family_2020_B01_P05_S01.xmp"
        _write_xmp(scan_xmp, _empty_xmp())

        result = reconcile_page(page_xmp, dry_run=True)

        writes_by_kind = {w.source_kind: w for w in result.pending_writes}
        self.assertEqual(set(writes_by_kind), {"page", "scan"})
        self.assertEqual(writes_by_kind["page"].xmp_path, page_xmp)
        self.assertEqual(writes_by_kind["scan"].xmp_path, scan_xmp)
        self.assertAlmostEqual(writes_by_kind["page"].rx, 0.09)
        self.assertAlmostEqual(writes_by_kind["page"].rw, 0.08)
        self.assertAlmostEqual(writes_by_kind["scan"].rx, 0.1, places=2)
        self.assertAlmostEqual(writes_by_kind["scan"].rw, 0.2, places=2)

    def test_face_in_crop_write_applied(self):
        """dry_run=False should write the face to the page XMP."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _face_xmp([{"name": "Alice", "rx": 0.0, "ry": 0.0, "rw": 1.0, "rh": 1.0}]))

        reconcile_page(page_xmp, dry_run=False)
        boxes = read_iptc_face_boxes(page_xmp)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["name"], "Alice")

    def test_face_id_in_scan_propagates_to_photo_and_page(self):
        """Face identity in scan is written to both photo (crop) and page after full reconcile.

        Geometry (scan covers page pixels x=16,y=9,w=128,h=72 in a 320×180 page):
          scan (0.1,0.1,0.3,0.3) → page ≈ (0.09,0.09,0.12,0.12)
                                 → crop (cx=0.25,cy=0.25,nw=0.4) ≈ (0.10,0.10,0.30,0.30)

        OpenCV patch comparison verifies the bounding-box pixels are the same scene
        content across all three files.
        """
        import cv2

        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        self._write_alignment_images(page_xmp, {"Family_2020_B01_P05_S01.tif": (16, 9, 128, 72)})
        # Build crop image: same 128×72 cut that the crop XMP represents
        page_arr = cv2.imread(str(page_xmp.with_suffix(".png")))
        crop_img_path = self.photos_dir / "Family_2020_B01_P05_D01-00_V.jpg"
        cv2.imwrite(str(crop_img_path), page_arr[9:81, 16:144])
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _empty_xmp())
        scan_xmp = self.archive_dir / "Family_2020_B01_P05_S01.xmp"
        _write_xmp(scan_xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.3, "rh": 0.3}]))

        reconcile_page(page_xmp, dry_run=False)

        page_faces = read_iptc_face_boxes(page_xmp)
        crop_faces = read_iptc_face_boxes(crop_xmp)
        scan_faces = read_iptc_face_boxes(scan_xmp)

        # Name propagates to all three files
        self.assertIn("Alice", {f["name"] for f in page_faces}, "face from scan must appear in page")
        self.assertIn("Alice", {f["name"] for f in crop_faces}, "face from scan must appear in crop")
        self.assertIn("Alice", {f["name"] for f in scan_faces}, "face must remain in originating scan")

        page_alice = next(f for f in page_faces if f["name"] == "Alice")
        crop_alice = next(f for f in crop_faces if f["name"] == "Alice")
        scan_alice = next(f for f in scan_faces if f["name"] == "Alice")

        # Coordinates are geometrically correct after OpenCV-aligned projection
        self.assertAlmostEqual(page_alice["rx"], 0.09, places=2, msg="page rx from scan")
        self.assertAlmostEqual(page_alice["ry"], 0.09, places=2, msg="page ry from scan")
        self.assertAlmostEqual(page_alice["rw"], 0.12, places=2, msg="page rw from scan")
        self.assertAlmostEqual(page_alice["rh"], 0.12, places=2, msg="page rh from scan")
        self.assertAlmostEqual(crop_alice["rx"], 0.10, places=2, msg="crop rx back-projected from scan")
        self.assertAlmostEqual(crop_alice["ry"], 0.10, places=2, msg="crop ry back-projected from scan")
        self.assertAlmostEqual(crop_alice["rw"], 0.30, places=2, msg="crop rw back-projected from scan")
        self.assertAlmostEqual(crop_alice["rh"], 0.30, places=2, msg="crop rh back-projected from scan")

        # Visual validation: bounding boxes in all three images enclose the same pixels
        scan_img = self.archive_dir / "Family_2020_B01_P05_S01.tif"
        page_img = page_xmp.with_suffix(".png")
        self._assert_same_region(scan_img, scan_alice, page_img, page_alice,
                                  msg="scan face patch vs page face patch")
        self._assert_same_region(scan_img, scan_alice, crop_img_path, crop_alice,
                                  msg="scan face patch vs crop face patch")

    def test_face_id_in_photo_propagates_to_scan_and_page(self):
        """Face identity in photo (crop) is written to both scan and page after full reconcile.

        Geometry (crop region cx=0.25,cy=0.25,nw=0.4; scan covers page pixels 16,9,128,72):
          crop (0.1,0.1,0.2,0.2) → page ≈ (0.09,0.09,0.08,0.08)
                                 → scan ≈ (0.10,0.10,0.20,0.20)

        OpenCV patch comparison verifies the bounding-box pixels are the same scene
        content across all three files.
        """
        import cv2

        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        self._write_alignment_images(page_xmp, {"Family_2020_B01_P05_S01.tif": (16, 9, 128, 72)})
        page_arr = cv2.imread(str(page_xmp.with_suffix(".png")))
        crop_img_path = self.photos_dir / "Family_2020_B01_P05_D01-00_V.jpg"
        cv2.imwrite(str(crop_img_path), page_arr[9:81, 16:144])
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]))
        scan_xmp = self.archive_dir / "Family_2020_B01_P05_S01.xmp"
        _write_xmp(scan_xmp, _empty_xmp())

        reconcile_page(page_xmp, dry_run=False)

        page_faces = read_iptc_face_boxes(page_xmp)
        crop_faces = read_iptc_face_boxes(crop_xmp)
        scan_faces = read_iptc_face_boxes(scan_xmp)

        # Name propagates to all three files
        self.assertIn("Alice", {f["name"] for f in page_faces}, "face from crop must appear in page")
        self.assertIn("Alice", {f["name"] for f in crop_faces}, "face must remain in originating crop")
        self.assertIn("Alice", {f["name"] for f in scan_faces}, "face from crop must appear in scan")

        page_alice = next(f for f in page_faces if f["name"] == "Alice")
        crop_alice = next(f for f in crop_faces if f["name"] == "Alice")
        scan_alice = next(f for f in scan_faces if f["name"] == "Alice")

        # Coordinates are geometrically correct
        self.assertAlmostEqual(page_alice["rx"], 0.09, places=2, msg="page rx from crop")
        self.assertAlmostEqual(page_alice["ry"], 0.09, places=2, msg="page ry from crop")
        self.assertAlmostEqual(page_alice["rw"], 0.08, places=2, msg="page rw from crop")
        self.assertAlmostEqual(page_alice["rh"], 0.08, places=2, msg="page rh from crop")
        self.assertAlmostEqual(scan_alice["rx"], 0.10, places=2, msg="scan rx back-projected from crop")
        self.assertAlmostEqual(scan_alice["ry"], 0.10, places=2, msg="scan ry back-projected from crop")
        self.assertAlmostEqual(scan_alice["rw"], 0.20, places=2, msg="scan rw back-projected from crop")
        self.assertAlmostEqual(scan_alice["rh"], 0.20, places=2, msg="scan rh back-projected from crop")

        # Visual validation: bounding boxes in all three images enclose the same pixels
        scan_img = self.archive_dir / "Family_2020_B01_P05_S01.tif"
        page_img = page_xmp.with_suffix(".png")
        self._assert_same_region(crop_img_path, crop_alice, page_img, page_alice,
                                  msg="crop face patch vs page face patch")
        self._assert_same_region(crop_img_path, crop_alice, scan_img, scan_alice,
                                  msg="crop face patch vs scan face patch")

    def test_face_id_in_page_propagates_to_photo_and_scan(self):
        """Face identity in page is written to both photo (crop) and scan after full reconcile.

        Geometry (crop region cx=0.25,cy=0.25,nw=0.4; scan covers page pixels 16,9,128,72):
          page (0.09,0.09,0.12,0.12) → crop ≈ (0.10,0.10,0.30,0.30)
                                     → scan ≈ (0.10,0.10,0.30,0.30)

        OpenCV patch comparison verifies the bounding-box pixels are the same scene
        content across all three files.
        """
        import cv2

        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        self._write_alignment_images(page_xmp, {"Family_2020_B01_P05_S01.tif": (16, 9, 128, 72)})
        page_arr = cv2.imread(str(page_xmp.with_suffix(".png")))
        crop_img_path = self.photos_dir / "Family_2020_B01_P05_D01-00_V.jpg"
        cv2.imwrite(str(crop_img_path), page_arr[9:81, 16:144])
        # Place face in page at a position inside the crop region
        # (region spans page 0.05–0.45 in both axes; face at 0.09,0.09 is inside it)
        merge_iptc_face_box(page_xmp, "Alice", 0.09, 0.09, 0.12, 0.12)
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _empty_xmp())
        scan_xmp = self.archive_dir / "Family_2020_B01_P05_S01.xmp"
        _write_xmp(scan_xmp, _empty_xmp())

        reconcile_page(page_xmp, dry_run=False)

        page_faces = read_iptc_face_boxes(page_xmp)
        crop_faces = read_iptc_face_boxes(crop_xmp)
        scan_faces = read_iptc_face_boxes(scan_xmp)

        # Name propagates to all three files
        self.assertIn("Alice", {f["name"] for f in page_faces}, "face must remain in originating page")
        self.assertIn("Alice", {f["name"] for f in crop_faces}, "face from page must appear in crop")
        self.assertIn("Alice", {f["name"] for f in scan_faces}, "face from page must appear in scan")

        page_alice = next(f for f in page_faces if f["name"] == "Alice")
        crop_alice = next(f for f in crop_faces if f["name"] == "Alice")
        scan_alice = next(f for f in scan_faces if f["name"] == "Alice")

        # Coordinates are geometrically correct
        self.assertAlmostEqual(crop_alice["rx"], 0.10, places=2, msg="crop rx from page")
        self.assertAlmostEqual(crop_alice["ry"], 0.10, places=2, msg="crop ry from page")
        self.assertAlmostEqual(crop_alice["rw"], 0.30, places=2, msg="crop rw from page")
        self.assertAlmostEqual(crop_alice["rh"], 0.30, places=2, msg="crop rh from page")
        self.assertAlmostEqual(scan_alice["rx"], 0.10, places=2, msg="scan rx from page via OpenCV inverse")
        self.assertAlmostEqual(scan_alice["ry"], 0.10, places=2, msg="scan ry from page via OpenCV inverse")
        self.assertAlmostEqual(scan_alice["rw"], 0.30, places=2, msg="scan rw from page via OpenCV inverse")
        self.assertAlmostEqual(scan_alice["rh"], 0.30, places=2, msg="scan rh from page via OpenCV inverse")

        # Visual validation: bounding boxes in all three images enclose the same pixels
        scan_img = self.archive_dir / "Family_2020_B01_P05_S01.tif"
        page_img = page_xmp.with_suffix(".png")
        self._assert_same_region(page_img, page_alice, crop_img_path, crop_alice,
                                  msg="page face patch vs crop face patch")
        self._assert_same_region(page_img, page_alice, scan_img, scan_alice,
                                  msg="page face patch vs scan face patch")

    def test_face_in_scan_backfills_to_page_and_crop(self):
        """A single-scan face should be reconciled into page space and crop space."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        self._write_alignment_images(page_xmp, {"Family_2020_B01_P05_S01.tif": (16, 9, 128, 72)})
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _empty_xmp())
        scan_xmp = self.archive_dir / "Family_2020_B01_P05_S01.xmp"
        _write_xmp(scan_xmp, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.3, "rh": 0.3}]))

        result = reconcile_page(page_xmp, dry_run=True)

        writes_by_kind = {w.source_kind: w for w in result.pending_writes}
        self.assertEqual(set(writes_by_kind), {"page", "crop"})
        self.assertEqual(writes_by_kind["page"].xmp_path, page_xmp)
        self.assertEqual(writes_by_kind["crop"].xmp_path, crop_xmp)
        self.assertAlmostEqual(writes_by_kind["page"].rx, 0.09, places=2)
        self.assertAlmostEqual(writes_by_kind["page"].rw, 0.12, places=2)
        self.assertAlmostEqual(writes_by_kind["crop"].rx, 0.1, places=2)
        self.assertAlmostEqual(writes_by_kind["crop"].rw, 0.3, places=2)

    def test_face_in_stitched_scan_uses_opencv_alignment_to_backfill_page_and_crop(self):
        """A multi-scan face should use image alignment instead of normalized scan coordinates."""
        page_xmp = self._page_xmp()
        scan_box = (120, 45, 140, 100)
        cx = (scan_box[0] + scan_box[2] / 2) / 320
        cy = (scan_box[1] + scan_box[3] / 2) / 180
        nw = scan_box[2] / 320
        nh = scan_box[3] / 180
        _write_xmp(page_xmp, _page_region_xmp([{"cx": cx, "cy": cy, "nw": nw, "nh": nh}]))
        self._write_alignment_images(
            page_xmp,
            {
                "Family_2020_B01_P05_S01.tif": (0, 25, 140, 100),
                "Family_2020_B01_P05_S02.tif": scan_box,
            },
        )
        _write_xmp(self.archive_dir / "Family_2020_B01_P05_S01.xmp", _empty_xmp())
        scan_xmp = self.archive_dir / "Family_2020_B01_P05_S02.xmp"
        _write_xmp(scan_xmp, _face_xmp([{"name": "Alice", "rx": 0.25, "ry": 0.3, "rw": 0.35, "rh": 0.35}]))
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _empty_xmp())

        result = reconcile_page(page_xmp, dry_run=True)

        writes_by_kind = {w.source_kind: w for w in result.pending_writes}
        self.assertEqual(set(writes_by_kind), {"page", "crop"})
        self.assertAlmostEqual(writes_by_kind["page"].rx, (scan_box[0] + 0.25 * scan_box[2]) / 320, places=2)
        self.assertAlmostEqual(writes_by_kind["page"].ry, (scan_box[1] + 0.3 * scan_box[3]) / 180, places=2)
        self.assertAlmostEqual(writes_by_kind["crop"].rx, 0.25, places=2)
        self.assertAlmostEqual(writes_by_kind["crop"].ry, 0.3, places=2)

    def test_page_face_uses_opencv_alignment_to_backfill_matching_scan_only(self):
        """A page face should be projected into the stitched scan that contains it."""
        page_xmp = self._page_xmp()
        scan_box = (120, 45, 140, 100)
        cx = (scan_box[0] + scan_box[2] / 2) / 320
        cy = (scan_box[1] + scan_box[3] / 2) / 180
        nw = scan_box[2] / 320
        nh = scan_box[3] / 180
        _write_xmp(page_xmp, _page_region_xmp([{"cx": cx, "cy": cy, "nw": nw, "nh": nh}]))
        self._write_alignment_images(
            page_xmp,
            {
                "Family_2020_B01_P05_S01.tif": (0, 25, 140, 100),
                "Family_2020_B01_P05_S02.tif": scan_box,
            },
        )
        scan1_xmp = self.archive_dir / "Family_2020_B01_P05_S01.xmp"
        scan2_xmp = self.archive_dir / "Family_2020_B01_P05_S02.xmp"
        _write_xmp(scan1_xmp, _empty_xmp())
        _write_xmp(scan2_xmp, _empty_xmp())
        crop_xmp = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop_xmp, _empty_xmp())
        merge_iptc_face_box(
            page_xmp,
            "Alice",
            (scan_box[0] + 0.25 * scan_box[2]) / 320,
            (scan_box[1] + 0.3 * scan_box[3]) / 180,
            0.35 * scan_box[2] / 320,
            0.35 * scan_box[3] / 180,
        )

        result = reconcile_page(page_xmp, dry_run=True)

        scan_writes = [w for w in result.pending_writes if w.source_kind == "scan"]
        self.assertEqual([w.xmp_path for w in scan_writes], [scan2_xmp])
        self.assertAlmostEqual(scan_writes[0].rx, 0.25, places=2)
        self.assertAlmostEqual(scan_writes[0].ry, 0.3, places=2)

    def test_conflict_skipped_in_backfill(self):
        """Clusters with multiple distinct names are skipped during backfill by default."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]))
        # Two crops at same location with different names
        crop1 = self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp"
        _write_xmp(crop1, _face_xmp([{"name": "Alice", "rx": 0.1, "ry": 0.1, "rw": 0.2, "rh": 0.2}]))
        # Face also in page with different name for same region (conflict)
        _write_xmp(
            page_xmp,
            _page_region_xmp([{"cx": 0.25, "cy": 0.25, "nw": 0.4, "nh": 0.4}]),
        )
        # Manually add a conflicting face to the page XMP
        # region ox=0.05, oy=0.05; Alice face at crop (0.1,0.1,0.2,0.2) →
        # page_rx = 0.05 + 0.1*0.4 = 0.09; same for all (project Alice's face)
        # Now add Bob at the same page location
        page_tree = ET.parse(str(page_xmp))
        desc = page_tree.getroot().find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
        from photoalbums.lib.face_region_reconciler import (
            _IPTC_IMAGE_REGION,
            _RDF_BAG,
            _RDF_LI,
            _add_iptc_face_li,
        )
        ir = ET.SubElement(desc, _IPTC_IMAGE_REGION)
        bag = ET.SubElement(ir, _RDF_BAG)
        # Bob at same projected location as Alice's face
        _add_iptc_face_li(bag, 1, "Bob", 0.09, 0.09, 0.08, 0.08)
        page_tree.write(str(page_xmp), encoding="utf-8", xml_declaration=True)

        result = reconcile_page(page_xmp, dry_run=True, skip_conflicts=True)
        conflict_clusters = [c for c in result.clusters if c.has_conflict]
        self.assertGreater(len(conflict_clusters), 0)
        conflict_writes = [w for w in result.pending_writes if w.xmp_path == crop1]
        self.assertEqual(len(conflict_writes), 0)  # conflict skipped

    def test_no_faces_returns_empty_result(self):
        """No faces anywhere should produce no clusters."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _empty_xmp())
        result = reconcile_page(page_xmp, dry_run=True)
        self.assertEqual(len(result.clusters), 0)
        self.assertEqual(len(result.pending_writes), 0)

    def test_sources_scanned_includes_all(self):
        """sources_scanned should include page + scan + crop XMPs."""
        page_xmp = self._page_xmp()
        _write_xmp(page_xmp, _empty_xmp())
        (self.archive_dir / "Family_2020_B01_P05_S01.tif").touch()
        _write_xmp(self.archive_dir / "Family_2020_B01_P05_S01.xmp", _empty_xmp())
        _write_xmp(self.photos_dir / "Family_2020_B01_P05_D01-00_V.xmp", _empty_xmp())

        result = reconcile_page(page_xmp, dry_run=True)
        names = {p.name for p in result.sources_scanned}
        self.assertIn("Family_2020_B01_P05_V.xmp", names)
        self.assertIn("Family_2020_B01_P05_S01.xmp", names)
        self.assertIn("Family_2020_B01_P05_D01-00_V.xmp", names)


if __name__ == "__main__":
    unittest.main()
