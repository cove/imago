"""migrate_naming_v3.py — Safe migration to the v3 archival naming convention.

Changes applied
---------------
1. ``_P##_VC.jpg``   → ``_P##_V.jpg``        (+ matching ``.xmp`` sidecar)
2. ``_D##_##.tif``   → ``_D##-##.tif``       (archive derived images, + ``.xmp``)
3. ``_D##_##.jpg``   → ``_D##-##_V.jpg``     (view derived images, + ``.xmp``)

XMP content patches
-------------------
- Archive derived image XMP: any ``dc:source`` value containing ``_D##_##.tif``
  is rewritten to ``_D##-##.tif`` so the self-reference stays current.

Invariants preserved
--------------------
- Archive scan TIF files (``_S##.tif``) are never touched.
- ``os.rename()`` is used throughout — preserves read-only attribute on Windows.
- SHA256 hash of every renamed file's content is verified post-rename (XMP files
  that were content-patched are excluded from the hash check).

Phases
------
  Phase 1 — Inventory (always runs; dry-run by default)
  Phase 2 — Pre-flight hash  (always runs with Phase 1)
  Phase 3 — Execute         (requires ``--run``)
  Phase 4 — Verify          (runs automatically after Phase 3)

Usage
-----
  # Dry run — print what would change, write manifest:
  uv run python photoalbums/scripts/migrate_naming_v3.py

  # Execute migration:
  uv run python photoalbums/scripts/migrate_naming_v3.py --run

  # Rollback (inverts the manifest):
  uv run python photoalbums/scripts/migrate_naming_v3.py --rollback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PHOTO_ALBUMS_ROOT = Path(r"C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums")

MANIFEST_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v3_manifest.json"
HASHES_BEFORE_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v3_hashes_before.json"
VERIFY_REPORT_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v3_verify.json"

# _P##_VC stem in a View directory
_UNKNOWN_BOOK_RE = r"(?:\.{3}|\N{HORIZONTAL ELLIPSIS})"

_VC_STEM_RE = re.compile(
    rf"^(?P<base>.+_B(?:\d{{2}}|{_UNKNOWN_BOOK_RE})_P\d+)_VC$",
    re.IGNORECASE,
)

# _D##_## stem anywhere (archive or view); captures everything before and after
_DERIVED_UNDER_RE = re.compile(
    r"^(?P<pre>.+_P\d+_D\d{1,2})_(?P<iter>\d{1,2})(?P<suf>.*)$",
    re.IGNORECASE,
)

# Pattern to patch inside XMP content: _D##_## followed by .tif
_XMP_DERIVED_PATCH_RE = re.compile(r"(_D\d{1,2})_(\d{1,2})(\.tif)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Phase 1: Inventory
# ---------------------------------------------------------------------------


def _iter_album_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dp = Path(dirpath)
        for fname in sorted(filenames):
            yield dp / fname


def build_rename_plan(root: Path) -> list[dict]:
    plan: list[dict] = []
    seen_new: set[Path] = set()

    for path in _iter_album_files(root):
        suffix_lower = path.suffix.lower()
        if suffix_lower not in {".jpg", ".jpeg", ".tif", ".tiff", ".xmp"}:
            continue

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        parent_name = parent.name

        in_view = parent_name.endswith("_View")
        in_archive = parent_name.endswith("_Archive")

        if not in_view and not in_archive:
            continue

        # --- _VC → _V (view only, jpg only) ---
        if in_view and suffix_lower in {".jpg", ".jpeg"}:
            m = _VC_STEM_RE.match(stem)
            if m:
                new_path = parent / f"{m.group('base')}_V{suffix}"
                if new_path != path:
                    _add(plan, seen_new, path, new_path, "vc_to_v")
                continue

        # --- _D##_## → _D##-## (archive tif) or _D##-##_V (view jpg) ---
        # Skip XMP here — handled by _pair_image_and_xmp
        if suffix_lower == ".xmp":
            continue
        # Skip archive scan TIFs
        if in_archive and suffix_lower in {".tif", ".tiff"}:
            if re.search(r"_S\d+$", stem, re.IGNORECASE):
                continue

        m = _DERIVED_UNDER_RE.match(stem)
        if m and not m.group("suf"):
            # No trailing suffix — could be archive tif or old-style view jpg
            iter_part = m.group("iter")
            pre = m.group("pre")
            if in_archive and suffix_lower in {".tif", ".tiff"}:
                new_path = parent / f"{pre}-{iter_part}{suffix}"
                _add(plan, seen_new, path, new_path, "derived_archive_hyphen")
            elif in_view and suffix_lower in {".jpg", ".jpeg"}:
                new_path = parent / f"{pre}-{iter_part}_V{suffix}"
                _add(plan, seen_new, path, new_path, "derived_view_hyphen_v")

    return plan


def _add(plan: list[dict], seen: set[Path], old: Path, new: Path, kind: str) -> None:
    if new in seen:
        print(f"  CONFLICT: {new} already in plan", file=sys.stderr)
        return
    plan.append({"old": str(old), "new": str(new), "kind": kind})
    seen.add(new)


def _pair_image_and_xmp(plan: list[dict]) -> list[dict]:
    xmp_entries: dict[str, str] = {e["old"]: e["new"] for e in plan if e["old"].lower().endswith(".xmp")}
    extra: list[dict] = []
    for entry in plan:
        old = entry["old"]
        if old.lower().endswith(".xmp"):
            continue
        xmp_old = Path(old).with_suffix(".xmp")
        xmp_new = Path(entry["new"]).with_suffix(".xmp")
        if xmp_old.exists() and str(xmp_old) not in xmp_entries:
            extra.append({"old": str(xmp_old), "new": str(xmp_new), "kind": entry["kind"] + "_xmp"})
    plan.extend(extra)
    return plan


# ---------------------------------------------------------------------------
# Phase 2: Pre-flight hash
# ---------------------------------------------------------------------------


def compute_hashes(plan: list[dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for entry in plan:
        p = Path(entry["old"])
        if p.is_file():
            hashes[entry["old"]] = _sha256(p)
    return hashes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Phase 3: Execute
# ---------------------------------------------------------------------------


def execute_plan(plan: list[dict], hashes_before: dict[str, str]) -> list[dict]:
    results: list[dict] = []
    for entry in plan:
        old = Path(entry["old"])
        new = Path(entry["new"])
        result = {
            "old": entry["old"],
            "new": entry["new"],
            "kind": entry["kind"],
            "status": "ok",
            "error": "",
            "xmp_patched": False,
        }
        try:
            if not old.exists():
                result["status"] = "skipped_missing"
            elif new.exists() and new != old:
                result["status"] = "skipped_conflict"
                result["error"] = f"Target already exists: {new}"
            else:
                if entry["kind"] == "derived_archive_hyphen_xmp":
                    _patch_derived_xmp(old)
                    result["xmp_patched"] = True
                os.rename(old, new)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        results.append(result)
    return results


def _patch_derived_xmp(xmp_path: Path) -> None:
    """Rewrite _D##_##.tif self-references in dc:source to _D##-##.tif."""
    text = xmp_path.read_text(encoding="utf-8")
    patched = _XMP_DERIVED_PATCH_RE.sub(r"\1-\2\3", text)
    if patched != text:
        xmp_path.write_text(patched, encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 4: Verify
# ---------------------------------------------------------------------------


def verify_results(results: list[dict], hashes_before: dict[str, str]) -> dict:
    errors: list[str] = []

    for entry in results:
        if entry["status"] in {"skipped_missing", "skipped_conflict", "error"}:
            errors.append(f"{entry['status']}: {entry['old']} → {entry.get('error', '')}")
            continue

        final = Path(entry["new"])
        if not final.exists():
            errors.append(f"Post-rename file missing: {final}")
            continue

        if final.suffix.lower() == ".xmp":
            try:
                ET.parse(final)
            except ET.ParseError as exc:
                errors.append(f"XMP not valid XML after rename: {final}: {exc}")

        if not entry.get("xmp_patched") and entry["old"] in hashes_before:
            if _sha256(final) != hashes_before[entry["old"]]:
                errors.append(f"Hash mismatch after rename: {final}")

    return {
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def rollback_plan(plan: list[dict]) -> list[dict]:
    inverted: list[dict] = []
    for entry in reversed(plan):
        inverted.append({"old": entry["new"], "new": entry["old"], "kind": entry["kind"] + "_rollback"})
    return inverted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(PHOTO_ALBUMS_ROOT), help="Photo albums root directory")
    parser.add_argument("--run", action="store_true", help="Execute the migration (default is dry-run)")
    parser.add_argument("--rollback", action="store_true", help="Invert and re-execute from saved manifest")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        return 1

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.rollback:
        if not MANIFEST_PATH.exists():
            print(f"ERROR: no manifest found at {MANIFEST_PATH}", file=sys.stderr)
            return 1
        plan = json.loads(MANIFEST_PATH.read_text())["plan"]
        plan = rollback_plan(plan)
        results = execute_plan(plan, {})
        ok = sum(1 for r in results if r["status"] == "ok")
        err = [r for r in results if r["status"] not in {"ok", "skipped_missing"}]
        print(f"Rollback: {ok} ok, {len(err)} errors")
        for e in err:
            print(f"  ERROR: {e}")
        return 0 if not err else 1

    print(f"Scanning {root} ...")
    plan = build_rename_plan(root)
    plan = _pair_image_and_xmp(plan)

    counts: dict[str, int] = {}
    for entry in plan:
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1

    print(f"\nRename plan ({len(plan)} operations):")
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}")

    if not plan:
        print("Nothing to rename.")
        return 0

    conflicts = [e["new"] for e in plan if Path(e["new"]).exists()]
    if conflicts:
        print(f"\nWARNING: {len(conflicts)} target paths already exist:")
        for c in conflicts[:10]:
            print(f"  {c}")

    print("\nComputing pre-flight hashes ...")
    hashes_before = compute_hashes(plan)
    print(f"Hashed {len(hashes_before)} files.")

    manifest = {"root": str(root), "plan": plan}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    HASHES_BEFORE_PATH.write_text(json.dumps(hashes_before, indent=2))
    print(f"Manifest written to {MANIFEST_PATH}")

    if not args.run:
        print("\nDRY RUN -- pass --run to execute.")
        for entry in plan[:15]:
            print(f"  [{entry['kind']}] {Path(entry['old']).name}  ->  {Path(entry['new']).name}")
        if len(plan) > 15:
            print(f"  ... and {len(plan) - 15} more")
        return 0

    print(f"\nExecuting {len(plan)} renames ...")
    results = execute_plan(plan, hashes_before)

    print("Verifying ...")
    report = verify_results(results, hashes_before)
    VERIFY_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nVerification: {report['ok']}/{report['total']} ok")

    if report["errors"]:
        print(f"ERRORS ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  {e}")
        return 1

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
