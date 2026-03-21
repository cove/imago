from __future__ import annotations

import sys
from pathlib import Path

from common import (
    ARCHIVE_CHECKSUM_FILE,
    ARCHIVE_DIR,
    DRIVE_CHECKSUM_FILE,
    DRIVE_DIR,
    LEGACY_ARCHIVE_CHECKSUM_FILE,
    LEGACY_DRIVE_CHECKSUM_FILE,
    verify_manifest,
    write_sha3_manifest,
)


def parse_verify_args(argv):
    algo = "auto"
    manifest = None
    for arg in list(argv or []):
        if arg in ("--blake3", "--b3"):
            algo = "blake3"
        elif arg in ("--sha3", "--sha3-256"):
            algo = "sha3"
        else:
            manifest = arg
    return manifest, algo


def _resolve_manifest(manifest, algo, primary_file, legacy_file):
    if manifest:
        return Path(manifest), algo
    if primary_file.exists():
        return primary_file, algo
    if legacy_file.exists():
        return legacy_file, "blake3" if algo == "auto" else algo
    return primary_file, algo


def resolve_archive_manifest(manifest, algo):
    return _resolve_manifest(
        manifest, algo, ARCHIVE_CHECKSUM_FILE, LEGACY_ARCHIVE_CHECKSUM_FILE
    )


def resolve_drive_manifest(manifest, algo):
    return _resolve_manifest(
        manifest, algo, DRIVE_CHECKSUM_FILE, LEGACY_DRIVE_CHECKSUM_FILE
    )


def verify_archive(argv=None):
    manifest, algo = parse_verify_args(argv)
    manifest, algo = resolve_archive_manifest(manifest, algo)
    print(f"Verifying: {manifest}\n")
    return verify_manifest(ARCHIVE_DIR, manifest, algo=algo)


def verify_drive(argv=None):
    manifest, algo = parse_verify_args(argv)
    manifest, algo = resolve_drive_manifest(manifest, algo)
    print(f"Verifying: {manifest}\n")
    return verify_manifest(DRIVE_DIR, manifest, algo=algo)


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
        ARCHIVE_CHECKSUM_FILE.name,
        DRIVE_CHECKSUM_FILE.name,
        LEGACY_ARCHIVE_CHECKSUM_FILE.name,
        LEGACY_DRIVE_CHECKSUM_FILE.name,
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
