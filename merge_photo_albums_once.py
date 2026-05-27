from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import os
import re
import shutil
import subprocess


RESTORED = Path("/Users/cove/OneDrive/Cordell, Leslie & Audrey/Photo Albums")
BACKUP = Path("/Users/cove/OneDriveBackupBeforeRestore/Cordell, Leslie & Audrey/Photo Albums")
TARGET = Path("/Users/cove/Photo Albums Merged")
CONFLICT_ROOT = TARGET / "_merge_conflicts"
MEDIA_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".pdf"}
COPY_ANOMALIES: list[dict[str, object]] = []


def scan(root: Path) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            st = path.stat()
            entries[rel] = {"path": path, "size": st.st_size}
    return entries


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def clone_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    restored_source = src.is_relative_to(RESTORED)
    if not src.is_relative_to(RESTORED):
        try:
            subprocess.run(["ln", str(src), str(dst)], check=True, capture_output=True, text=True, timeout=0.5)
            return
        except subprocess.TimeoutExpired as error:
            COPY_ANOMALIES.append({"source": str(src), "target": str(dst), "operation": "link", "error": str(error)})
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or repr(error)
            COPY_ANOMALIES.append({"source": str(src), "target": str(dst), "operation": "link", "error": detail})

    if restored_source and src.stat().st_size < 1024 * 1024:
        try:
            shutil.copy2(src, dst)
            return
        except OSError as error:
            COPY_ANOMALIES.append({"source": str(src), "target": str(dst), "operation": "copy2", "error": repr(error)})

    if not restored_source or src.stat().st_size >= 1024 * 1024:
        try:
            subprocess.run(["cp", "-c", "-p", str(src), str(dst)], check=True, capture_output=True, text=True, timeout=5)
            return
        except subprocess.TimeoutExpired as error:
            COPY_ANOMALIES.append({"source": str(src), "target": str(dst), "operation": "clone", "error": repr(error)})
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or repr(error)
            COPY_ANOMALIES.append({"source": str(src), "target": str(dst), "operation": "clone", "error": detail})

    try:
        subprocess.run(["cp", "-p", str(src), str(dst)], check=True, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        COPY_ANOMALIES.append({"source": str(src), "target": str(dst), "operation": "cp", "error": repr(error)})
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or repr(error)
        COPY_ANOMALIES.append({"source": str(src), "target": str(dst), "operation": "cp", "error": detail})

def is_media(rel: str) -> bool:
    return Path(rel).suffix.lower() in MEDIA_EXTS


def album_key(rel: str) -> str:
    first = rel.split("/", 1)[0]
    for suffix in ("_Archive", "_Photos", "_Pages"):
        if first.endswith(suffix):
            return first[: -len(suffix)]
    return first


def page_number(rel: str) -> int | None:
    match = re.search(r"_P(\d{2,3})(?:_|\.)", Path(rel).name)
    if not match:
        return None
    return int(match.group(1))


def read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line for line in path.read_text(encoding="utf-8").splitlines() if line}


def read_photo_album_list(path: Path) -> set[str]:
    prefix = "Photo Albums/"
    return {line[len(prefix) :] if line.startswith(prefix) else line for line in read_list(path)}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def page_gaps(entries: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    by_folder: dict[str, set[int]] = defaultdict(set)
    for rel in entries:
        page = page_number(rel)
        if page is not None:
            by_folder[str(Path(rel).parent)].add(page)

    rows = []
    for folder, pages in sorted(by_folder.items()):
        if len(pages) < 2:
            continue
        missing = [page for page in range(min(pages), max(pages) + 1) if page not in pages]
        if missing:
            rows.append(
                {
                    "folder": folder,
                    "min_page": min(pages),
                    "max_page": max(pages),
                    "missing_pages": " ".join(f"{page:02d}" for page in missing),
                    "missing_count": len(missing),
                }
            )
    return rows


def main() -> None:
    if TARGET.exists() and any(TARGET.iterdir()):
        raise SystemExit(f"Refusing to merge into non-empty target: {TARGET}")
    TARGET.mkdir(parents=True, exist_ok=True)

    restored = scan(RESTORED)
    backup = scan(BACKUP)
    restored_paths = set(restored)
    backup_paths = set(backup)
    only_restored = restored_paths - backup_paths
    only_backup = backup_paths - restored_paths
    common = restored_paths & backup_paths

    size_mismatches = read_photo_album_list(Path("onedrive_common_size_mismatch.txt"))
    hash_mismatches = read_photo_album_list(Path("onedrive_common_hash_mismatch.txt"))
    common_conflicts = sorted((size_mismatches | hash_mismatches) & common)

    backup_only_media_by_hash: dict[str, list[str]] = defaultdict(list)
    for rel in sorted(path for path in only_backup if is_media(path)):
        backup_only_media_by_hash[sha256(backup[rel]["path"])].append(rel)  # type: ignore[index]

    renamed_media_rows: list[dict[str, object]] = []
    restored_media_to_backup: dict[str, str] = {}
    duplicate_match_anomalies: list[dict[str, object]] = []

    for rel in sorted(path for path in only_restored if is_media(path)):
        digest = sha256(restored[rel]["path"])  # type: ignore[index]
        matches = backup_only_media_by_hash.get(digest, [])
        same_album = [match for match in matches if album_key(match) == album_key(rel)]
        chosen = same_album[0] if same_album else (matches[0] if matches else None)
        if chosen is None:
            continue
        restored_media_to_backup[rel] = chosen
        restored_page = page_number(rel)
        backup_page = page_number(chosen)
        renamed_media_rows.append(
            {
                "restored_path": rel,
                "backup_path": chosen,
                "sha256": digest,
                "same_album": bool(same_album),
                "restored_page": restored_page or "",
                "backup_page": backup_page or "",
                "page_delta": (backup_page - restored_page) if restored_page is not None and backup_page is not None else "",
            }
        )
        if len(matches) > 1:
            duplicate_match_anomalies.append(
                {
                    "restored_path": rel,
                    "chosen_backup_path": chosen,
                    "candidate_count": len(matches),
                    "candidate_paths": " | ".join(matches),
                }
            )

    restored_sidecars_for_renamed_media: dict[str, str] = {}
    for restored_media, backup_media in restored_media_to_backup.items():
        restored_xmp = str(Path(restored_media).with_suffix(".xmp"))
        if restored_xmp in only_restored:
            backup_xmp = str(Path(backup_media).with_suffix(".xmp"))
            restored_sidecars_for_renamed_media[restored_xmp] = backup_xmp

    print(f"Copying backup base: {len(backup):,} files")
    for index, rel in enumerate(sorted(backup), 1):
        if index % 1000 == 0:
            print(f"  backup {index:,}/{len(backup):,}", flush=True)
        clone_copy(backup[rel]["path"], TARGET / rel)  # type: ignore[index]

    restored_unique_copied = []
    restored_renamed_sidecars = []
    restored_duplicate_media_skipped = set(restored_media_to_backup)

    print(f"Merging restored-only paths: {len(only_restored):,} files")
    for index, rel in enumerate(sorted(only_restored), 1):
        if index % 500 == 0:
            print(f"  restored-only {index:,}/{len(only_restored):,}", flush=True)
        src = restored[rel]["path"]  # type: ignore[index]
        if rel in restored_duplicate_media_skipped:
            continue
        if rel in restored_sidecars_for_renamed_media:
            dst = CONFLICT_ROOT / "restored_sidecars_for_renamed_media" / rel
            clone_copy(src, dst)
            restored_renamed_sidecars.append(
                {
                    "restored_xmp_path": rel,
                    "backup_xmp_path": restored_sidecars_for_renamed_media[rel],
                }
            )
            continue
        clone_copy(src, TARGET / rel)
        restored_unique_copied.append(rel)

    restored_gap_rows = page_gaps(restored)
    backup_gap_rows = page_gaps(backup)
    shift_counts = Counter(
        (row["restored_path"].split("/", 1)[0], row["backup_path"].split("/", 1)[0], row["page_delta"])
        for row in renamed_media_rows
    )
    shift_rows = [
        {
            "restored_folder": restored_folder,
            "backup_folder": backup_folder,
            "page_delta": page_delta,
            "count": count,
        }
        for (restored_folder, backup_folder, page_delta), count in sorted(shift_counts.items())
    ]

    write_csv(
        Path("photo_albums_merged_renamed_media_matches.csv"),
        renamed_media_rows,
        ["restored_path", "backup_path", "sha256", "same_album", "restored_page", "backup_page", "page_delta"],
    )
    write_csv(
        Path("photo_albums_merged_restored_sidecars_for_renamed_media.csv"),
        restored_renamed_sidecars,
        ["restored_xmp_path", "backup_xmp_path"],
    )
    write_csv(
        Path("photo_albums_merged_duplicate_hash_anomalies.csv"),
        duplicate_match_anomalies,
        ["restored_path", "chosen_backup_path", "candidate_count", "candidate_paths"],
    )
    write_csv(
        Path("photo_albums_merged_page_gap_report_restored.csv"),
        restored_gap_rows,
        ["folder", "min_page", "max_page", "missing_pages", "missing_count"],
    )
    write_csv(
        Path("photo_albums_merged_page_gap_report_backup.csv"),
        backup_gap_rows,
        ["folder", "min_page", "max_page", "missing_pages", "missing_count"],
    )
    write_csv(
        Path("photo_albums_merged_shift_summary.csv"),
        shift_rows,
        ["restored_folder", "backup_folder", "page_delta", "count"],
    )
    write_csv(
        Path("photo_albums_merged_copy_anomalies.csv"),
        COPY_ANOMALIES,
        ["source", "target", "operation", "error"],
    )
    Path("photo_albums_merged_restored_unique_copied.txt").write_text(
        "\n".join(restored_unique_copied) + ("\n" if restored_unique_copied else ""),
        encoding="utf-8",
    )
    Path("photo_albums_merged_same_path_conflicts.txt").write_text(
        "\n".join(common_conflicts) + ("\n" if common_conflicts else ""),
        encoding="utf-8",
    )

    target_count = sum(len(filenames) for _, _, filenames in os.walk(TARGET))
    lines = [
        "# Photo Albums Merge Report",
        "",
        f"- Target: `{TARGET}`",
        f"- Backup files used as canonical base: {len(backup):,}",
        f"- Restored-only files: {len(only_restored):,}",
        f"- Restored-only media matched to backup-only media by SHA-256 and treated as renames: {len(restored_media_to_backup):,}",
        f"- Restored-only files copied into canonical tree: {len(restored_unique_copied):,}",
        f"- Restored sidecars for renamed media copied to conflict area: {len(restored_renamed_sidecars):,}",
        f"- Same-path restored files with different content kept as backup canonical and listed only: {len(common_conflicts):,}",
        f"- Duplicate hash match anomalies: {len(duplicate_match_anomalies):,}",
        f"- Copy anomalies: {len(COPY_ANOMALIES):,}",
        f"- Target file count, including conflict copies: {target_count:,}",
        "- Copy method: backup-canonical files are hard links where possible; restored additions are copied or cloned as needed.",
        "",
        "## Files Written",
        "",
        "- `photo_albums_merged_renamed_media_matches.csv`",
        "- `photo_albums_merged_shift_summary.csv`",
        "- `photo_albums_merged_restored_sidecars_for_renamed_media.csv`",
        "- `photo_albums_merged_duplicate_hash_anomalies.csv`",
        "- `photo_albums_merged_copy_anomalies.csv`",
        "- `photo_albums_merged_page_gap_report_restored.csv`",
        "- `photo_albums_merged_page_gap_report_backup.csv`",
        "- `photo_albums_merged_restored_unique_copied.txt`",
        "- `photo_albums_merged_same_path_conflicts.txt`",
    ]
    Path("photo_albums_merged_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
