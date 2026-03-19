#!/usr/bin/env python
"""Extract bundled runtime binaries needed by platform-specific workflows."""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
VHS_BIN_DIR = BASE / "vhs" / "bin"
LINUX_FFMPEG_ARCHIVE = VHS_BIN_DIR / "ffmpeg-release-amd64-static.tar.xz"


def _needs_refresh(target_path: Path, source_paths: list[Path]) -> bool:
    if not target_path.exists():
        return True
    target_mtime = target_path.stat().st_mtime
    return any(src.exists() and src.stat().st_mtime > target_mtime for src in source_paths)


def _safe_chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass


def _extract_tar_member(
    archive_path: Path,
    member_suffix: str,
    output_path: Path,
    *,
    executable: bool = False,
) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tarfile.open(archive_path, mode="r:*") as archive:
        member = next(
            (item for item in archive.getmembers() if item.isfile() and item.name.endswith(member_suffix)),
            None,
        )
        if member is None:
            raise RuntimeError(f"Member '{member_suffix}' not found in {archive_path.name}")
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"Unable to extract '{member.name}' from {archive_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    _safe_chmod(output_path, 0o755 if executable else 0o644)


def install_windows_binary_archives() -> None:
    if sys.platform != "win32":
        return

    ffmpeg_dir = BASE / "vhs" / "software" / "Windows" / "FFmpeg-QTGMC Easy 2025.01.11"
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        executable_path = ffmpeg_dir / name
        archive_path = ffmpeg_dir / f"{name}.zip"
        if executable_path.exists():
            continue
        if not archive_path.exists():
            print(f"WARNING: {archive_path.name} not found in {ffmpeg_dir}; skipping.")
            continue
        print(f"Extracting {archive_path.name} ...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extract(name, ffmpeg_dir)
        print(f"  -> {executable_path}")


def install_linux_binary_archives() -> None:
    if not sys.platform.startswith("linux"):
        return

    if not LINUX_FFMPEG_ARCHIVE.exists():
        print(f"Linux ffmpeg archive missing from vhs/bin/: {LINUX_FFMPEG_ARCHIVE.name}")
        print("Skipping binary extraction.")
        return

    ffmpeg_bin = VHS_BIN_DIR / "ffmpeg"
    ffprobe_bin = VHS_BIN_DIR / "ffprobe"

    if _needs_refresh(ffmpeg_bin, [LINUX_FFMPEG_ARCHIVE]):
        print(f"Extracting ffmpeg from {LINUX_FFMPEG_ARCHIVE.name} ...")
        _extract_tar_member(
            LINUX_FFMPEG_ARCHIVE,
            "/ffmpeg",
            ffmpeg_bin,
            executable=True,
        )
    if _needs_refresh(ffprobe_bin, [LINUX_FFMPEG_ARCHIVE]):
        print(f"Extracting ffprobe from {LINUX_FFMPEG_ARCHIVE.name} ...")
        _extract_tar_member(
            LINUX_FFMPEG_ARCHIVE,
            "/ffprobe",
            ffprobe_bin,
            executable=True,
        )

    print("Linux ffmpeg binaries extracted to vhs/bin/.")
    if shutil.which("mediainfo") is None:
        print(
            "WARNING: 'mediainfo' not found on PATH. "
            "Install it (for example: sudo apt-get install mediainfo) or set MEDIAINFO_BIN."
        )
    else:
        print("Found system 'mediainfo' on PATH.")


def main() -> None:
    install_windows_binary_archives()
    install_linux_binary_archives()
    print("Runtime bootstrap complete.")


if __name__ == "__main__":
    main()
