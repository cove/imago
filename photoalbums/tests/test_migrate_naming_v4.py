from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from photoalbums.lib.xmp_sidecar import DC_NS, _get_rdf_desc, _read_processing_history

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    module_path = REPO_ROOT / "photoalbums" / "scripts" / "migrate_naming_v4.py"
    spec = importlib.util.spec_from_file_location("migrate_naming_v4", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency optional in some environments
        raise RuntimeError(f"cv2 and numpy are required for this test: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 32, 3), color, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not write test jpeg: {path}")


def _write_xmp(path: Path, source_text: str = "legacy-source") -> None:
    _write_file(
        path,
        f"""<?xml version="1.0" encoding="utf-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <dc:source>{source_text}</dc:source>
      <xmp:CreatorTool>imago-test</xmp:CreatorTool>
      <xmp:CreateDate>2026-03-29T00:00:00Z</xmp:CreateDate>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
    )


class TestMigrateNamingV4(unittest.TestCase):
    def test_build_plan_assigns_next_free_archive_iteration_and_view_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_1907-1946_B01_Archive"
            view = root / "Family_1907-1946_B01_View"

            _write_bytes(archive / "Family_1907-1946_B01_P03_S01.tif", b"s01")
            _write_bytes(archive / "Family_1907-1946_B01_P03_S02.tif", b"s02")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01.tif", b"crop")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01_C.png", b"colorized")
            _write_bytes(view / "Family_1907-1946_B01_P03_D01-01_C.jpg", b"view")

            plan = MODULE.build_plan(root)

            self.assertEqual(len(plan), 1)
            entry = plan[0]
            self.assertEqual(
                Path(entry["archive_image_new"]).name,
                "Family_1907-1946_B01_P03_D01-02.png",
            )
            self.assertEqual(
                Path(entry["view_image_new"]).name,
                "Family_1907-1946_B01_P03_D01-02_V.jpg",
            )
            self.assertEqual(
                entry["xmp_source_text"],
                "Family_1907-1946_B01_P03_S01.tif Family_1907-1946_B01_P03_S02.tif",
            )
            self.assertEqual(
                entry["xmp_source_detail"],
                "Family_1907-1946_B01_P03_D01-01.tif",
            )
            self.assertFalse(entry["archive_target_exists"])
            self.assertEqual(entry["view_collision_status"], "none")

    def test_build_plan_preserves_existing_non_one_iteration_when_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_1907-1946_B01_Archive"

            _write_bytes(archive / "Family_1907-1946_B01_P03_S01.tif", b"s01")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01.tif", b"crop")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-02_C.png", b"colorized")

            plan = MODULE.build_plan(root)

            self.assertEqual(len(plan), 1)
            self.assertEqual(
                Path(plan[0]["archive_image_new"]).name,
                "Family_1907-1946_B01_P03_D01-02.png",
            )
            self.assertEqual(
                Path(plan[0]["view_image_new"]).name,
                "Family_1907-1946_B01_P03_D01-02_V.jpg",
            )

    def test_build_plan_skips_to_next_free_iteration_on_archive_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_1907-1946_B01_Archive"

            _write_bytes(archive / "Family_1907-1946_B01_P03_S01.tif", b"s01")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01.tif", b"crop")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-02.png", b"occupied")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01_C.png", b"colorized")

            plan = MODULE.build_plan(root)

            self.assertEqual(len(plan), 1)
            self.assertEqual(
                Path(plan[0]["archive_image_new"]).name,
                "Family_1907-1946_B01_P03_D01-03.png",
            )
            self.assertFalse(plan[0]["archive_target_exists"])

    def test_build_plan_marks_duplicate_view_collision(self):
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"opencv/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_1907-1946_B01_Archive"
            view = root / "Family_1907-1946_B01_View"

            _write_bytes(archive / "Family_1907-1946_B01_P03_S01.tif", b"s01")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01.tif", b"crop")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01_C.png", b"colorized")
            _write_jpeg(view / "Family_1907-1946_B01_P03_D01-01_C.jpg", (10, 40, 90))
            _write_jpeg(view / "Family_1907-1946_B01_P03_D01-02_V.jpg", (10, 40, 90))

            plan = MODULE.build_plan(root)

            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["view_collision_status"], "duplicate")

    def test_build_plan_marks_different_view_collision(self):
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency optional
            self.skipTest(f"opencv/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_1907-1946_B01_Archive"
            view = root / "Family_1907-1946_B01_View"

            _write_bytes(archive / "Family_1907-1946_B01_P03_S01.tif", b"s01")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01.tif", b"crop")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01_C.png", b"colorized")
            _write_jpeg(view / "Family_1907-1946_B01_P03_D01-01_C.jpg", (10, 40, 90))
            _write_jpeg(view / "Family_1907-1946_B01_P03_D01-02_V.jpg", (90, 10, 40))

            plan = MODULE.build_plan(root)

            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["view_collision_status"], "different")

    def test_execute_plan_renames_and_patches_xmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Family_1907-1946_B01_Archive"
            view = root / "Family_1907-1946_B01_View"

            _write_bytes(archive / "Family_1907-1946_B01_P03_S01.tif", b"s01")
            _write_bytes(archive / "Family_1907-1946_B01_P03_S02.tif", b"s02")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01.tif", b"crop")
            _write_bytes(archive / "Family_1907-1946_B01_P03_D01-01_C.png", b"colorized-png")
            _write_bytes(view / "Family_1907-1946_B01_P03_D01-01_C.jpg", b"colorized-jpg")
            _write_xmp(archive / "Family_1907-1946_B01_P03_D01-01_C.xmp")
            _write_xmp(view / "Family_1907-1946_B01_P03_D01-01_C.xmp")

            plan = MODULE.build_plan(root)
            hashes_before = MODULE.compute_hashes(plan)
            results = MODULE.execute_plan(plan)
            report = MODULE.verify_results(results, hashes_before, plan)

            self.assertEqual(report["errors"], [])

            archive_xmp = archive / "Family_1907-1946_B01_P03_D01-02.xmp"
            view_xmp = view / "Family_1907-1946_B01_P03_D01-02_V.xmp"
            self.assertTrue(archive_xmp.is_file())
            self.assertTrue(view_xmp.is_file())

            tree = ET.parse(archive_xmp)
            desc = _get_rdf_desc(tree)
            assert desc is not None
            self.assertEqual(
                str(desc.findtext(f"{{{DC_NS}}}source", default="") or "").strip(),
                "Family_1907-1946_B01_P03_S01.tif Family_1907-1946_B01_P03_S02.tif",
            )
            history = _read_processing_history(desc)
            self.assertTrue(
                any(
                    str(item.get("action") or "").strip() == "colorized"
                    and isinstance(item.get("parameters"), dict)
                    and str(item["parameters"].get("source_detail") or "").strip()
                    == "Family_1907-1946_B01_P03_D01-01.tif"
                    for item in history
                )
            )


if __name__ == "__main__":
    unittest.main()
