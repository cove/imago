"""migrate_naming_v4.py - Migrate `_C` colorized detail names to variant numbering.

Changes applied
---------------
1. Archive colorized detail PNGs:
   ``..._D##-##_C.png`` -> ``..._D##-NN.png`` using the next free archive slot
2. View colorized detail JPGs:
   ``..._D##-##_C.jpg`` -> ``..._D##-##_V.jpg``
3. Matching XMP sidecars move with their companion image.

Metadata patches
----------------
- ``dc:source`` is rewritten to the full page scan basenames, space-separated.
- XMP processing history gains a ``colorized`` event pointing back to the source
  detail crop basename (the ``-01.tif`` archive crop for that D-number family).

Archive slot assignment
-----------------------
- Archive colorized PNGs are reassigned to the next free ``D##-##`` slot in the
  same ``_Archive`` directory.
- This avoids basename collisions with the source crop TIFFs, for example
  ``..._D01-01_C.png`` -> ``..._D01-02.png`` when ``..._D01-01.tif`` exists.

View collision analysis
-----------------------
- If a target ``..._V.jpg`` already exists, OpenCV compares the colliding images.
- Collisions are classified as ``duplicate`` when the decoded pixels match within a
  small JPEG tolerance, otherwise ``different``.
- The migration reports these view collisions in the dry-run manifest and overwrites
  the existing ``_View`` targets during execution.

Usage
-----
  # Dry run:
  uv run python photoalbums/scripts/migrate_naming_v4.py

  # Execute:
  uv run python photoalbums/scripts/migrate_naming_v4.py --run
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photoalbums.naming import SCAN_TIFF_RE, parse_album_filename
from photoalbums.lib.xmp_sidecar import (
    DC_NS,
    _get_rdf_desc,
    _read_processing_history,
    _set_processing_history,
    _set_simple_text,
)

PHOTO_ALBUMS_ROOT = Path(r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums")
MANIFEST_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v4_manifest.json"
HASHES_BEFORE_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v4_hashes_before.json"
VERIFY_REPORT_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v4_verify.json"

COLORIZED_STEM_RE = re.compile(
    r"^(?P<prefix>.+_D(?P<derived>\d{1,2})-(?P<iter>\d{1,2}))_C$",
    re.IGNORECASE,
)
VARIANT_STEM_RE = re.compile(
    r"^(?P<base>.+_D(?P<derived>\d{1,2}))-(?P<iter>\d{1,2})$",
    re.IGNORECASE,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_VISUAL_DUPLICATE_MAX_DIFF = 2
_VISUAL_DUPLICATE_MEAN_DIFF = 0.25


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_archive_colorized_pngs(root: Path) -> list[Path]:
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        parent = Path(dirpath)
        if not parent.name.endswith("_Archive"):
            continue
        for name in sorted(filenames):
            path = parent / name
            if path.suffix.lower() != ".png":
                continue
            if COLORIZED_STEM_RE.match(path.stem):
                matches.append(path)
    return matches


def _family_base_from_stem(stem: str) -> tuple[str, int, int]:
    match = COLORIZED_STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"Not a colorized detail stem: {stem}")
    prefix = str(match.group("prefix"))
    current_iter = int(match.group("iter"))
    derived = int(match.group("derived"))
    family_base = prefix.rsplit("-", 1)[0]
    return family_base, derived, current_iter


def _iter_variant_stems(directory: Path, family_base: str) -> set[int]:
    occupied: set[int] = set()
    prefix = f"{family_base}-"
    for child in directory.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() not in IMAGE_EXTENSIONS and child.suffix.lower() != ".xmp":
            continue
        stem = child.stem
        if stem.endswith("_C"):
            continue
        if not stem.startswith(prefix):
            continue
        match = VARIANT_STEM_RE.match(stem)
        if match is None:
            continue
        occupied.add(int(match.group("iter")))
    return occupied


def _next_free_variant(current_iter: int, occupied: set[int], planned: set[int]) -> int:
    candidate = current_iter
    while candidate in occupied or candidate in planned:
        candidate += 1
    return candidate


def _view_dir_for_archive_dir(archive_dir: Path) -> Path:
    if not archive_dir.name.endswith("_Archive"):
        raise ValueError(f"Not an archive directory: {archive_dir}")
    return archive_dir.with_name(archive_dir.name[:-8] + "_View")


def _full_page_scan_names(archive_dir: Path, image_name: str) -> list[str]:
    _collection, _year, _book, page = parse_album_filename(image_name)
    if not str(page).isdigit() or int(page) <= 0:
        raise ValueError(f"Could not parse page number from {image_name}")
    page_number = int(page)
    scans: list[str] = []
    for child in sorted(archive_dir.iterdir()):
        match = SCAN_TIFF_RE.match(child.name)
        if match is None:
            continue
        if int(match.group("page")) != page_number:
            continue
        scans.append(child.name)
    if not scans:
        raise ValueError(f"No page scan TIFFs found for {image_name} in {archive_dir}")
    return scans


def _source_detail_basename(target_stem: str) -> str:
    match = VARIANT_STEM_RE.match(target_stem)
    if match is None:
        raise ValueError(f"Not a variant detail stem: {target_stem}")
    return f"{match.group('base')}-01.tif"


def _paired_path(image_path: Path, new_path: Path, suffix: str) -> tuple[Path, Path] | None:
    old_sidecar = image_path.with_suffix(suffix)
    if not old_sidecar.exists():
        return None
    return old_sidecar, new_path.with_suffix(suffix)


def _require_cv2():
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError("opencv-python and numpy are required for visual collision checks.") from exc
    return cv2, np


def _classify_visual_collision(old_path: Path, new_path: Path) -> str:
    if not old_path.exists() or not new_path.exists():
        return "missing"
    cv2, np = _require_cv2()
    old_image = cv2.imread(str(old_path), cv2.IMREAD_COLOR)
    new_image = cv2.imread(str(new_path), cv2.IMREAD_COLOR)
    if old_image is None or new_image is None:
        raise RuntimeError(f"Could not read collision images: {old_path} | {new_path}")
    if old_image.shape != new_image.shape:
        return "different"
    diff = cv2.absdiff(old_image, new_image)
    max_diff = int(diff.max())
    mean_diff = float(np.mean(diff))
    if max_diff <= _VISUAL_DUPLICATE_MAX_DIFF and mean_diff <= _VISUAL_DUPLICATE_MEAN_DIFF:
        return "duplicate"
    return "different"


def _patch_xmp(path: Path, *, source_text: str, source_detail: str) -> None:
    tree = ET.parse(path)
    desc = _get_rdf_desc(tree)
    if desc is None:
        raise ValueError(f"rdf:Description missing from XMP sidecar: {path}")
    _set_simple_text(desc, f"{{{DC_NS}}}source", source_text)

    history = list(_read_processing_history(desc))
    already_present = False
    for entry in history:
        if str(entry.get("action") or "").strip() != "colorized":
            continue
        parameters = entry.get("parameters")
        if not isinstance(parameters, dict):
            continue
        if str(parameters.get("source_detail") or "").strip() == source_detail:
            already_present = True
            break
    if not already_present:
        creator = ""
        existing_creator = desc.findtext("{http://ns.adobe.com/xap/1.0/}CreatorTool", default="")
        if existing_creator is not None:
            creator = str(existing_creator or "").strip()
        when_text = str(desc.findtext("{http://ns.adobe.com/xap/1.0/}CreateDate", default="") or "").strip()
        history.append(
            {
                "action": "colorized",
                "when": when_text or _utc_now(),
                "software_agent": creator or "https://github.com/cove/imago",
                "parameters": {
                    "stage": "colorize",
                    "source_detail": source_detail,
                },
            }
        )
        _set_processing_history(desc, history)

    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_plan(root: Path) -> list[dict]:
    archives = _iter_archive_colorized_pngs(root)
    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for archive_png in archives:
        family_base, _derived, _current_iter = _family_base_from_stem(archive_png.stem)
        groups[(archive_png.parent, family_base)].append(archive_png)

    plan: list[dict] = []
    for (archive_dir, family_base), items in sorted(groups.items(), key=lambda row: (str(row[0][0]), row[0][1])):
        occupied = _iter_variant_stems(archive_dir, family_base)
        planned: set[int] = set()
        for archive_png in sorted(items, key=lambda path: path.name.lower()):
            _family_base, derived, current_iter = _family_base_from_stem(archive_png.stem)
            assigned_iter = _next_free_variant(current_iter, occupied, planned)
            planned.add(assigned_iter)

            target_stem = f"{family_base}-{assigned_iter:02d}"
            target_archive_png = archive_dir / f"{target_stem}.png"
            view_dir = _view_dir_for_archive_dir(archive_dir)
            old_view_jpg = view_dir / f"{archive_png.stem}.jpg"
            new_view_jpg = view_dir / f"{target_stem}_V.jpg"
            source_text = " ".join(_full_page_scan_names(archive_dir, archive_png.name))
            source_detail = _source_detail_basename(target_stem)

            entry = {
                "archive_image_old": str(archive_png),
                "archive_image_new": str(target_archive_png),
                "view_image_old": str(old_view_jpg),
                "view_image_new": str(new_view_jpg),
                "xmp_source_text": source_text,
                "xmp_source_detail": source_detail,
                "family_base": family_base,
                "derived": derived,
                "from_iter": current_iter,
                "to_iter": assigned_iter,
                "archive_target_exists": bool(target_archive_png.exists()),
                "view_collision_status": "none",
            }

            if old_view_jpg.exists() and new_view_jpg.exists() and old_view_jpg != new_view_jpg:
                entry["view_collision_status"] = _classify_visual_collision(old_view_jpg, new_view_jpg)

            archive_xmp = _paired_path(archive_png, target_archive_png, ".xmp")
            if archive_xmp is not None:
                entry["archive_xmp_old"] = str(archive_xmp[0])
                entry["archive_xmp_new"] = str(archive_xmp[1])
            view_xmp = _paired_path(old_view_jpg, new_view_jpg, ".xmp")
            if view_xmp is not None:
                entry["view_xmp_old"] = str(view_xmp[0])
                entry["view_xmp_new"] = str(view_xmp[1])
            plan.append(entry)
    return plan


def compute_hashes(plan: list[dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for entry in plan:
        for key in (
            "archive_image_old",
            "archive_xmp_old",
            "view_image_old",
            "view_xmp_old",
        ):
            raw = str(entry.get(key) or "").strip()
            if not raw:
                continue
            path = Path(raw)
            if path.is_file():
                hashes[raw] = _sha256(path)
    return hashes


def _planned_paths(entry: dict) -> list[tuple[Path, Path]]:
    return [
        (Path(entry["archive_image_old"]), Path(entry["archive_image_new"])),
        (Path(entry["view_image_old"]), Path(entry["view_image_new"])),
        *(
            (Path(str(entry[key_old])), Path(str(entry[key_new])))
            for key_old, key_new in (
                ("archive_xmp_old", "archive_xmp_new"),
                ("view_xmp_old", "view_xmp_new"),
            )
            if str(entry.get(key_old) or "").strip()
        ),
    ]


def _ensure_targets_available(entry: dict) -> None:
    if bool(entry.get("archive_target_exists")):
        raise FileExistsError(f"Archive target already exists: {entry['archive_image_new']}")
    for old_path, new_path in _planned_paths(entry):
        if new_path.parent.name.endswith("_View"):
            continue
        if old_path.exists() and new_path.exists() and old_path != new_path:
            raise FileExistsError(f"Target already exists: {new_path}")


def _patch_entry_xmps(entry: dict) -> list[str]:
    patched: list[str] = []
    for key in ("archive_xmp_old", "view_xmp_old"):
        raw = str(entry.get(key) or "").strip()
        if not raw:
            continue
        xmp_path = Path(raw)
        _patch_xmp(
            xmp_path,
            source_text=str(entry["xmp_source_text"]),
            source_detail=str(entry["xmp_source_detail"]),
        )
        patched.append(str(xmp_path))
    return patched


def _rename_existing(old_path: Path, new_path: Path, *, overwrite: bool = False) -> None:
    if old_path.exists():
        if overwrite and new_path.exists() and new_path != old_path:
            new_path.unlink()
        os.rename(old_path, new_path)


def _rename_entry_files(entry: dict) -> None:
    _rename_existing(Path(entry["archive_image_old"]), Path(entry["archive_image_new"]))
    _rename_existing(Path(entry["view_image_old"]), Path(entry["view_image_new"]), overwrite=True)
    for key_old, key_new in (
        ("archive_xmp_old", "archive_xmp_new"),
        ("view_xmp_old", "view_xmp_new"),
    ):
        old_raw = str(entry.get(key_old) or "").strip()
        if not old_raw:
            continue
        overwrite = Path(str(entry[key_new])).parent.name.endswith("_View")
        _rename_existing(Path(old_raw), Path(str(entry[key_new])), overwrite=overwrite)


def execute_plan(plan: list[dict]) -> list[dict]:
    results: list[dict] = []
    for entry in plan:
        result = {
            "archive_image_old": entry["archive_image_old"],
            "archive_image_new": entry["archive_image_new"],
            "status": "ok",
            "error": "",
            "patched_xmp": [],
        }
        try:
            _ensure_targets_available(entry)
            result["patched_xmp"] = _patch_entry_xmps(entry)
            _rename_entry_files(entry)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        results.append(result)
    return results


def _verify_paths(entry: dict) -> list[str]:
    errors: list[str] = []
    for key in ("archive_image_new", "view_image_new", "archive_xmp_new", "view_xmp_new"):
        raw = str(entry.get(key) or "").strip()
        if raw and not Path(raw).exists():
            errors.append(f"missing after rename: {raw}")
    for key in ("archive_image_old", "view_image_old", "archive_xmp_old", "view_xmp_old"):
        raw = str(entry.get(key) or "").strip()
        if raw and Path(raw).exists():
            errors.append(f"old path still exists: {raw}")
    return errors


def _verify_hashes(entry: dict, hashes_before: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for key in ("archive_image_old", "view_image_old"):
        raw = str(entry.get(key) or "").strip()
        if not raw or raw not in hashes_before:
            continue
        new_path = Path(str(entry[key.replace("_old", "_new")]))
        if new_path.is_file() and _sha256(new_path) != hashes_before[raw]:
            errors.append(f"hash mismatch after rename: {new_path}")
    return errors


def _has_colorized_history(desc: ET.Element, source_detail: str) -> bool:
    history = _read_processing_history(desc)
    return any(
        str(item.get("action") or "").strip() == "colorized"
        and isinstance(item.get("parameters"), dict)
        and str(item["parameters"].get("source_detail") or "").strip() == source_detail
        for item in history
    )


def _verify_xmp_files(entry: dict) -> list[str]:
    errors: list[str] = []
    for key in ("archive_xmp_new", "view_xmp_new"):
        raw = str(entry.get(key) or "").strip()
        if not raw:
            continue
        try:
            tree = ET.parse(raw)
        except ET.ParseError as exc:
            errors.append(f"invalid xmp xml: {raw}: {exc}")
            continue
        desc = _get_rdf_desc(tree)
        if desc is None:
            errors.append(f"missing rdf:Description: {raw}")
            continue
        source_text = str(desc.findtext(f"{{{DC_NS}}}source", default="") or "").strip()
        if source_text != str(entry["xmp_source_text"]):
            errors.append(f"dc:source mismatch: {raw}")
        if not _has_colorized_history(desc, str(entry["xmp_source_detail"])):
            errors.append(f"missing colorized history event: {raw}")
    return errors


def verify_results(results: list[dict], hashes_before: dict[str, str], plan: list[dict]) -> dict:
    errors: list[str] = []
    plan_by_old = {str(entry["archive_image_old"]): entry for entry in plan}

    for result in results:
        if result["status"] != "ok":
            errors.append(f"error: {result['archive_image_old']}: {result['error']}")
            continue

        entry = plan_by_old[result["archive_image_old"]]
        errors.extend(_verify_paths(entry))
        errors.extend(_verify_hashes(entry, hashes_before))
        errors.extend(_verify_xmp_files(entry))

    return {
        "total": len(results),
        "ok": sum(1 for row in results if row["status"] == "ok"),
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(PHOTO_ALBUMS_ROOT), help="Photo albums root directory")
    parser.add_argument("--run", action="store_true", help="Execute the migration in place")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        return 1

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    plan = build_plan(root)
    counts: dict[str, int] = defaultdict(int)
    for entry in plan:
        counts[str(Path(entry["archive_image_old"]).parent.name)] += 1
    archive_conflicts = sum(1 for entry in plan if bool(entry.get("archive_target_exists")))
    view_status_counts = Counter(str(entry.get("view_collision_status") or "none") for entry in plan)

    print(f"Found {len(plan)} colorized archive variants to migrate.")
    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")
    print(f"Archive target conflicts: {archive_conflicts}")
    print(
        "View collisions: "
        f"duplicate={view_status_counts.get('duplicate', 0)} "
        f"different={view_status_counts.get('different', 0)} "
        f"none={view_status_counts.get('none', 0)}"
    )

    manifest = {"root": str(root), "plan": plan}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not plan:
        print("Nothing to rename.")
        return 0

    hashes_before = compute_hashes(plan)
    HASHES_BEFORE_PATH.write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")

    if not args.run:
        print("DRY RUN -- pass --run to execute.")
        for entry in plan[:15]:
            print(
                "  "
                f"{Path(entry['archive_image_old']).name} -> {Path(entry['archive_image_new']).name}"
                f" | {Path(entry['view_image_new']).name}"
                f" | view_collision={entry['view_collision_status']}"
            )
        if len(plan) > 15:
            print(f"  ... and {len(plan) - 15} more")
        return 0

    results = execute_plan(plan)
    report = verify_results(results, hashes_before, plan)
    VERIFY_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verification: {report['ok']}/{report['total']} ok")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
