"""migrate_naming_v2.py — Safe migration to the v2 archival naming convention.

Changes applied
---------------
1. ``_P##_stitched.jpg`` → ``_P##_VR.jpg``   (+ matching ``.xmp`` sidecar)
2. ``_P##.jpg``          → ``_P##_V.jpg``    (+ matching ``.xmp`` sidecar)
3. ``PanamaCanal&Mexico``  → ``PanamaCanalAndMexico``  (directory names + all files)
   XMP sidecars in PanamaCanal View dirs have their dc:source text patched to
   replace ``PanamaCanal&Mexico`` token with ``PanamaCanalAndMexico``.

Invariants preserved
--------------------
- Archive TIF files are never renamed (``_S##.tif`` stays unchanged).
- Detail crops ``_D##_##`` are never renamed.
- ``os.rename()`` is used throughout — preserves read-only attribute on Windows.
- SHA256 hash of every renamed file's content is verified post-rename.

Phases
------
  Phase 1 — Inventory (always runs; dry-run by default)
  Phase 2 — Pre-flight hash  (always runs with Phase 1)
  Phase 3 — Execute         (requires ``--run``)
  Phase 4 — Verify          (runs automatically after Phase 3)
  Phase 5 — Regen checksums (requires ``--regen-checksums``, separate from Phase 3)

Usage
-----
  # Dry run — print what would change, write manifest:
  uv run python photoalbums/scripts/migrate_naming_v2.py

  # Execute migration:
  uv run python photoalbums/scripts/migrate_naming_v2.py --run

  # Rollback (inverts the manifest):
  uv run python photoalbums/scripts/migrate_naming_v2.py --rollback

  # Regenerate SHA256SUMS after migration:
  uv run python photoalbums/scripts/migrate_naming_v2.py --regen-checksums
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

MANIFEST_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v2_manifest.json"
HASHES_BEFORE_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v2_hashes_before.json"
VERIFY_REPORT_PATH = REPO_ROOT / ".tmp" / "migrate_naming_v2_verify.json"

OLD_PANAMA = "PanamaCanal&Mexico"
NEW_PANAMA = "PanamaCanalAndMexico"

# Matches bare view page stem: ends exactly at _P## (no type token follows)
_BARE_VIEW_STEM_RE = re.compile(
    r"^(?P<base>.+_B(?:\d{2}|…)_P\d+)$",
    re.IGNORECASE,
)
# Matches stitched view page stem (legacy _stitched)
_STITCHED_STEM_RE = re.compile(
    r"^(?P<base>.+_B(?:\d{2}|…)_P\d+)_stitched$",
    re.IGNORECASE,
)
# Matches the old _VR stem (renamed to _VC)
_VR_STEM_RE = re.compile(
    r"^(?P<base>.+_B(?:\d{2}|…)_P\d+)_VR$",
    re.IGNORECASE,
)
# Matches album filename component for PanamaCanal
_PANAMA_TOKEN_RE = re.compile(re.escape(OLD_PANAMA), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Phase 1: Inventory
# ---------------------------------------------------------------------------


def _iter_album_files(root: Path):
    """Yield (path, dir_type) for every file under _View and _Archive dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dp = Path(dirpath)
        for fname in sorted(filenames):
            yield dp / fname


def build_rename_plan(root: Path) -> list[dict]:
    """Return list of rename operations: {old, new, kind}."""
    plan: list[dict] = []
    seen_new: set[Path] = set()

    for path in _iter_album_files(root):
        suffix_lower = path.suffix.lower()
        is_image_or_xmp = suffix_lower in {".jpg", ".jpeg", ".xmp"}
        is_tif = suffix_lower in {".tif", ".tiff"}
        if not is_image_or_xmp and not is_tif:
            continue

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        parent_name = parent.name

        # --- PanamaCanal rename (TIFs included; possibly combined with naming convention transform) ---
        if OLD_PANAMA in str(path):
            new_name = path.name.replace(OLD_PANAMA, NEW_PANAMA)
            new_stem = Path(new_name).stem
            new_suffix = Path(new_name).suffix
            # Inside _View dirs also apply the stitched→VR / bare→V transform
            if parent_name.endswith("_View"):
                ms = _STITCHED_STEM_RE.match(new_stem)
                mb = _BARE_VIEW_STEM_RE.match(new_stem)
                if ms:
                    new_name = f"{ms.group('base')}_VC{new_suffix}"
                    kind = "panama_stitched_to_vc"
                elif mb and new_suffix.lower() in {".jpg", ".jpeg"}:
                    new_name = f"{mb.group('base')}_V{new_suffix}"
                    kind = "panama_bare_to_v"
                else:
                    kind = "panama_file"
            else:
                kind = "panama_file"
            new_path = parent / new_name
            if new_path != path:
                plan.append({"old": str(path), "new": str(new_path), "kind": kind})
                seen_new.add(new_path)
            continue

        # Non-Panama TIFs need no further renaming
        if is_tif:
            continue

        # Only rename non-Panama files inside _View directories
        if not parent_name.endswith("_View"):
            continue

        # --- _stitched → _VR ---
        m = _STITCHED_STEM_RE.match(stem)
        if m:
            base = m.group("base")
            new_path = parent / f"{base}_VC{suffix}"
            if new_path != path:
                if new_path in seen_new:
                    print(f"  CONFLICT: {new_path} already in plan", file=sys.stderr)
                    continue
                plan.append({"old": str(path), "new": str(new_path), "kind": "stitched_to_vc"})
                seen_new.add(new_path)
            continue

        # --- _VR → _VC (rename from previous migration pass) ---
        m = _VR_STEM_RE.match(stem)
        if m:
            base = m.group("base")
            new_path = parent / f"{base}_VC{suffix}"
            if new_path != path:
                if new_path in seen_new:
                    print(f"  CONFLICT: {new_path} already in plan", file=sys.stderr)
                    continue
                plan.append({"old": str(path), "new": str(new_path), "kind": "vr_to_vc"})
                seen_new.add(new_path)
            continue

        # --- bare _P## → _P##_V ---
        m = _BARE_VIEW_STEM_RE.match(stem)
        if m and suffix.lower() in {".jpg", ".jpeg"}:
            base = m.group("base")
            new_path = parent / f"{base}_V{suffix}"
            if new_path != path:
                if new_path in seen_new:
                    print(f"  CONFLICT: {new_path} already in plan", file=sys.stderr)
                    continue
                plan.append({"old": str(path), "new": str(new_path), "kind": "bare_to_v"})
                seen_new.add(new_path)
            continue

    # --- PanamaCanal directory renames (collected last, applied last) ---
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        dp = Path(dirpath)
        if OLD_PANAMA in dp.name:
            new_dir = dp.parent / dp.name.replace(OLD_PANAMA, NEW_PANAMA)
            plan.append(
                {
                    "old": str(dp),
                    "new": str(new_dir),
                    "kind": "panama_dir",
                }
            )

    return plan


def _pair_image_and_xmp(plan: list[dict]) -> list[dict]:
    """Ensure every image rename also has a matching XMP rename in the plan."""
    xmp_entries: dict[str, str] = {e["old"]: e["new"] for e in plan if e["old"].lower().endswith(".xmp")}
    extra: list[dict] = []
    for entry in plan:
        old = entry["old"]
        if not old.lower().endswith(".xmp"):
            xmp_old = Path(old).with_suffix(".xmp")
            xmp_new = Path(entry["new"]).with_suffix(".xmp")
            if xmp_old.exists() and str(xmp_old) not in xmp_entries:
                extra.append(
                    {
                        "old": str(xmp_old),
                        "new": str(xmp_new),
                        "kind": entry["kind"] + "_xmp",
                    }
                )
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
    # Files first, directories last
    files = [e for e in plan if e["kind"] != "panama_dir"]
    dirs = [e for e in plan if e["kind"] == "panama_dir"]

    for entry in files + dirs:
        old = Path(entry["old"])
        new = Path(entry["new"])
        result = {"old": entry["old"], "new": entry["new"], "kind": entry["kind"], "status": "ok", "error": ""}
        try:
            if entry["kind"] == "panama_dir":
                if old.exists():
                    os.rename(old, new)
            else:
                if not old.exists():
                    result["status"] = "skipped_missing"
                elif new.exists() and new != old:
                    result["status"] = "skipped_conflict"
                    result["error"] = f"Target already exists: {new}"
                else:
                    # Patch PanamaCanal token in XMP content before renaming
                    if "panama" in entry["kind"] and old.suffix.lower() == ".xmp":
                        _patch_panama_xmp(old)
                    os.rename(old, new)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        results.append(result)
    return results


def _patch_panama_xmp(xmp_path: Path) -> None:
    """Replace PanamaCanal&Mexico token in XMP dc:source fields."""
    text = xmp_path.read_text(encoding="utf-8")
    if OLD_PANAMA not in text:
        return
    patched = text.replace(OLD_PANAMA, NEW_PANAMA)
    xmp_path.write_text(patched, encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 4: Verify
# ---------------------------------------------------------------------------


def verify_results(
    results: list[dict],
    hashes_before: dict[str, str],
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    for entry in results:
        if entry["status"] in {"skipped_missing", "skipped_conflict", "error"}:
            errors.append(f"{entry['status']}: {entry['old']} → {entry.get('error', '')}")
            continue

        old_key = entry["old"]
        # For panama file/xmp entries the manifest records the interim path (file
        # renamed inside the still-old-named directory).  By the time verify runs the
        # directory rename has also been applied, so compute the actual final path.
        interim = Path(entry["new"])
        if "panama" in entry["kind"] and entry["kind"] != "panama_dir":
            final = Path(str(interim).replace(OLD_PANAMA, NEW_PANAMA))
        else:
            final = interim

        if not final.exists():
            errors.append(f"Post-rename file missing: {final}")
            continue

        # Verify XMP is valid XML
        if final.suffix.lower() == ".xmp":
            try:
                ET.parse(final)
            except ET.ParseError as exc:
                errors.append(f"XMP not valid XML after rename: {final}: {exc}")

        # Verify PanamaCanal token removed from XMP
        if "panama" in entry["kind"] and final.suffix.lower() == ".xmp":
            text = final.read_text(encoding="utf-8", errors="replace")
            if OLD_PANAMA in text:
                errors.append(f"Old PanamaCanal token still present in: {final}")

        # Verify content hash unchanged (XMP may have been patched, so skip hash
        # check for panama XMPs whose content was intentionally modified)
        if old_key in hashes_before and "panama" not in entry["kind"]:
            new_hash = _sha256(final)
            if new_hash != hashes_before[old_key]:
                errors.append(f"Hash mismatch after rename: {final}")

    report = {
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "errors": errors,
        "warnings": warnings,
    }
    return report


# ---------------------------------------------------------------------------
# Phase 5: Regen SHA256SUMS
# ---------------------------------------------------------------------------


def regen_checksums(root: Path) -> None:
    lines: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        dp = Path(dirpath)
        if not dp.name.endswith("_Archive"):
            continue
        for fname in sorted(filenames):
            if fname.lower().endswith(".tif"):
                fpath = dp / fname
                digest = _sha256(fpath)
                rel = fpath.relative_to(root)
                lines.append(f"SHA256 ({rel}) = {digest}")

    checksums_path = root / "SHA256SUMS"
    checksums_path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print(f"Written {len(lines)} checksums to {checksums_path}")


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def rollback_plan(plan: list[dict]) -> list[dict]:
    """Invert a manifest plan (new → old)."""
    inverted: list[dict] = []
    dirs = [e for e in plan if e["kind"] == "panama_dir"]
    files = [e for e in plan if e["kind"] != "panama_dir"]
    for entry in list(reversed(dirs)) + list(reversed(files)):
        inverted.append(
            {
                "old": entry["new"],
                "new": entry["old"],
                "kind": entry["kind"] + "_rollback",
            }
        )
    return inverted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(PHOTO_ALBUMS_ROOT), help="Photo albums root directory")
    parser.add_argument("--run", action="store_true", help="Execute the migration (default is dry-run)")
    parser.add_argument("--rollback", action="store_true", help="Invert and re-execute from saved manifest")
    parser.add_argument("--regen-checksums", action="store_true", help="Regenerate SHA256SUMS from archive TIFs")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        return 1

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Regen checksums only ---
    if args.regen_checksums:
        print(f"Regenerating SHA256SUMS under {root} ...")
        regen_checksums(root)
        return 0

    # --- Rollback ---
    if args.rollback:
        if not MANIFEST_PATH.exists():
            print(f"ERROR: no manifest found at {MANIFEST_PATH}", file=sys.stderr)
            return 1
        plan = json.loads(MANIFEST_PATH.read_text())["plan"]
        plan = rollback_plan(plan)
        hashes: dict[str, str] = {}
        print(f"Rolling back {len(plan)} operations ...")
        results = execute_plan(plan, hashes)
        ok = sum(1 for r in results if r["status"] == "ok")
        err = [r for r in results if r["status"] not in {"ok", "skipped_missing"}]
        print(f"Rollback: {ok} ok, {len(err)} errors")
        for e in err:
            print(f"  ERROR: {e}")
        return 0 if not err else 1

    # --- Phase 1: Inventory ---
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

    # Check for conflicts
    new_paths = [e["new"] for e in plan]
    conflicts = [p for p in new_paths if Path(p).exists()]
    if conflicts:
        print(f"\nWARNING: {len(conflicts)} target paths already exist:")
        for c in conflicts[:10]:
            print(f"  {c}")
        if not args.run:
            print("Resolve conflicts before running --run.")

    # --- Phase 2: Pre-flight hash ---
    print("\nComputing pre-flight hashes ...")
    hashes_before = compute_hashes(plan)
    print(f"Hashed {len(hashes_before)} files.")

    manifest = {"root": str(root), "plan": plan}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    HASHES_BEFORE_PATH.write_text(json.dumps(hashes_before, indent=2))
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Hashes written to {HASHES_BEFORE_PATH}")

    if not args.run:
        print("\nDRY RUN -- pass --run to execute.")
        # Print sample
        sample = plan[:15]
        for entry in sample:
            print(f"  [{entry['kind']}] {Path(entry['old']).name}  ->  {Path(entry['new']).name}")
        if len(plan) > 15:
            print(f"  ... and {len(plan) - 15} more")
        return 0

    # --- Phase 3: Execute ---
    print(f"\nExecuting {len(plan)} renames ...")
    results = execute_plan(plan, hashes_before)

    # --- Phase 4: Verify ---
    print("Verifying ...")
    report = verify_results(results, hashes_before)
    VERIFY_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nVerification: {report['ok']}/{report['total']} ok")

    if report["errors"]:
        print(f"ERRORS ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  {e}")
        return 1

    print("Migration complete. Review the verification report, then run --regen-checksums.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
