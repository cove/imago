from __future__ import annotations

import sys
from pathlib import Path

from common import (
    DRIVE_CHECKSUM_FILE,
    DRIVE_DIR,
    all_store_archive_dirs,
    archive_checksum_file_for,
    archive_dir_for,
    verify_manifest,
    write_sha3_manifest,
)


def _parse_verify_arg(arg, remaining, i):
    """Return (algo, archive, manifest, consumed_extra) for one argument."""
    if arg in ("--blake3", "--b3"):
        return "blake3", None, None, 0
    if arg in ("--sha3", "--sha3-256"):
        return "sha3", None, None, 0
    if arg == "--archive" and i + 1 < len(remaining):
        return None, remaining[i + 1], None, 1
    if arg.startswith("--archive="):
        return None, arg.split("=", 1)[1], None, 0
    return None, None, arg, 0


def parse_verify_args(argv):
    algo = "auto"
    manifest = None
    archive = None
    remaining = list(argv or [])
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        new_algo, new_archive, new_manifest, extra = _parse_verify_arg(arg, remaining, i)
        if new_algo:
            algo = new_algo
        if new_archive:
            archive = new_archive
        if new_manifest:
            manifest = new_manifest
        i += 1 + extra
    return manifest, algo, archive


def _verify_all_archives(algo):
    archive_dirs = all_store_archive_dirs()
    if not archive_dirs:
        raise SystemExit("No archives found. Use --archive <name> or pass a manifest path.")
    failed = 0
    for root_dir in archive_dirs:
        manifest_path = root_dir / "SHA256SUMS"
        if not manifest_path.exists():
            print(f"Skipping {root_dir} (no SHA256SUMS)\n")
            continue
        print(f"Verifying: {manifest_path}\n")
        rc = verify_manifest(root_dir, manifest_path, algo=algo)
        if rc:
            failed += 1
    return failed


def verify_archive(argv=None):
    manifest, algo, archive = parse_verify_args(argv)
    if manifest:
        root_dir = Path(manifest).parent
        manifest_path = Path(manifest)
    elif archive:
        root_dir = archive_dir_for(archive)
        manifest_path = archive_checksum_file_for(archive)
    else:
        return _verify_all_archives(algo)
    print(f"Verifying: {manifest_path}\n")
    return verify_manifest(root_dir, manifest_path, algo=algo)


def verify_drive(argv=None):
    manifest, algo, _archive = parse_verify_args(argv)
    if manifest:
        manifest_path = Path(manifest)
    else:
        manifest_path = DRIVE_CHECKSUM_FILE
    print(f"Verifying: {manifest_path}\n")
    return verify_manifest(DRIVE_DIR, manifest_path, algo=algo)


def should_ignore_drive_path(path: Path) -> bool:
    name = path.name
    exact = {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        ".Spotlight-V100",
        ".Trashes",
        ".fseventsd",
        ".TemporaryItems",
        ".VolumeIcon.icns",
        ".AppleDouble",
        ".AppleDesktop",
        ".android_secure",
        "LOST.DIR",
        DRIVE_CHECKSUM_FILE.name,
        "venv-mac",
        ".venv",
        ".git",
        ".gitignore",
        "__pycache__",
    }
    dirs = {
        "$RECYCLE.BIN",
        "System Volume Information",
        "Android",
        ".thumbnails",
    }
    prefixes = ("._", ".Trash")
    if name in exact:
        return True
    if name in dirs:
        return True
    return any(name.startswith(prefix) for prefix in prefixes)


def generate_drive_checksum():
    write_sha3_manifest(
        DRIVE_DIR,
        DRIVE_CHECKSUM_FILE,
        relative_base=DRIVE_DIR,
        ignore_fn=should_ignore_drive_path,
    )
    print(f"Checksum manifest: {DRIVE_CHECKSUM_FILE}")
    print("All done.")
    return 0


def main(argv=None):
    _ = argv
    return generate_drive_checksum()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
