#!/usr/bin/env python3
#
# Sets up a shared Python virtual environment for the imago monorepo
# (vhs + photoalbums), based on the host OS. Run from the repo root.
#
import subprocess
import sys
import venv
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

BASE = Path(__file__).parent.resolve()

if sys.platform == "win32":
    VENV_DIR = BASE / ".venv"
elif sys.platform == "darwin":
    VENV_DIR = BASE / "venv-mac"
elif sys.platform.startswith("linux"):
    VENV_DIR = BASE / "venv-linux"
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

REQ_FILE = BASE / "requirements.txt"
BIN_DIR = BASE / "vhs" / "bin"
LINUX_FFMPEG_ARCHIVE = BIN_DIR / "ffmpeg-release-amd64-static.tar.xz"


def create_venv():
    if VENV_DIR.exists():
        print("Virtual environment already exists:", VENV_DIR)
        return
    print("Creating virtual environment:", VENV_DIR)
    venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    print("Done.")


def get_python_bin():
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_requirements():
    if not REQ_FILE.exists():
        print("No requirements.txt found - skipping install.")
        return
    python_bin = get_python_bin()
    print("Installing requirements from requirements.txt ...")
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", str(REQ_FILE)])
    print("Requirements installed.")


def _needs_refresh(target_path, source_paths):
    target = Path(target_path)
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    return any(
        Path(src).exists() and Path(src).stat().st_mtime > target_mtime
        for src in source_paths
    )


def _safe_chmod(path, mode):
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass


def _extract_tar_member(archive_path, member_suffix, output_path, executable=False):
    archive = Path(archive_path)
    out = Path(output_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    with tarfile.open(archive, mode="r:*") as tf:
        member = next(
            (m for m in tf.getmembers() if m.isfile() and m.name.endswith(member_suffix)),
            None,
        )
        if member is None:
            raise RuntimeError(f"Member '{member_suffix}' not found in {archive.name}")
        src = tf.extractfile(member)
        if src is None:
            raise RuntimeError(f"Unable to extract '{member.name}' from {archive.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    _safe_chmod(out, 0o755 if executable else 0o644)


def install_windows_binary_archives():
    if sys.platform != "win32":
        return
    ffmpeg_dir = BASE / "vhs" / "software" / "Windows" / "FFmpeg-QTGMC Easy 2025.01.11"
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        exe = ffmpeg_dir / name
        zip_file = ffmpeg_dir / f"{name}.zip"
        if exe.exists():
            continue
        if not zip_file.exists():
            print(f"WARNING: {zip_file.name} not found in {ffmpeg_dir} — skipping.")
            continue
        print(f"Extracting {zip_file.name} ...")
        with zipfile.ZipFile(zip_file) as zf:
            zf.extract(name, ffmpeg_dir)
        print(f"  → {exe}")


def install_linux_binary_archives():
    if not sys.platform.startswith("linux"):
        return
    if not LINUX_FFMPEG_ARCHIVE.exists():
        print(f"Linux ffmpeg archive missing from vhs/bin/: {LINUX_FFMPEG_ARCHIVE.name}")
        print("Skipping binary extraction.")
        return
    ffmpeg_bin = BIN_DIR / "ffmpeg"
    ffprobe_bin = BIN_DIR / "ffprobe"
    if _needs_refresh(ffmpeg_bin, [LINUX_FFMPEG_ARCHIVE]):
        print(f"Extracting ffmpeg from {LINUX_FFMPEG_ARCHIVE.name} ...")
        _extract_tar_member(LINUX_FFMPEG_ARCHIVE, "/ffmpeg", ffmpeg_bin, executable=True)
    if _needs_refresh(ffprobe_bin, [LINUX_FFMPEG_ARCHIVE]):
        print(f"Extracting ffprobe from {LINUX_FFMPEG_ARCHIVE.name} ...")
        _extract_tar_member(LINUX_FFMPEG_ARCHIVE, "/ffprobe", ffprobe_bin, executable=True)
    print("Linux ffmpeg binaries extracted to vhs/bin/.")
    if shutil.which("mediainfo") is None:
        print(
            "WARNING: 'mediainfo' not found on PATH. "
            "Install it (e.g. sudo apt-get install mediainfo) or set MEDIAINFO_BIN."
        )
    else:
        print("Found system 'mediainfo' on PATH.")


def main():
    create_venv()
    install_windows_binary_archives()
    install_linux_binary_archives()
    install_requirements()
    print("Environment setup complete.")
    print(f"Venv: {VENV_DIR}")


if __name__ == "__main__":
    main()
