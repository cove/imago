from __future__ import annotations

import sys
from pathlib import Path

from common import (
    DRIVE_CHECKSUM_FILE,
    DRIVE_DIR,
    all_store_archive_dirs,
    archive_checksum_file_for,
    archive_dir_for,
    drive_checksum_file_for,
    verify_manifest,
    write_sha3_manifest,
)


def parse_verify_args(argv):
    algo = "auto"
    manifest = None
    archive = None
    remaining = list(argv or [])
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg in ("--blake3", "--b3"):
            algo = "blake3"
        elif arg in ("--sha3", "--sha3-256"):
            algo = "sha3"
        elif arg == "--archive" and i + 1 < len(remaining):
            i += 1
            archive = remaining[i]
        elif arg.startswith("--archive="):
            archive = arg.split("=", 1)[1]
        else:
            manifest = arg
        i += 1
    return manifest, algo, archive


def verify_archive(argv=None):
    manifest, algo, archive = parse_verify_args(argv)
    if manifest:
        root_dir = Path(manifest).parent
        manifest_path = Path(manifest)
    elif archive:
        root_dir = archive_dir_for(archive)
        manifest_path = archive_checksum_file_for(archive)
    else:
        raise SystemExit("verify archive requires --archive <name> or a manifest path argument")
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
