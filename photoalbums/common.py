import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

CREATOR = "Audrey D. Cordell"
INCOMING_NAME = "incoming_scan.tif"

FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)_P(?P<page>\d{2})_S(?P<scan>\d{2})\.tif$",
    re.IGNORECASE,
)
PAGE_SCAN_RE = re.compile(r"_P(?P<page>\d+)_S(?P<scan>\d+)", re.IGNORECASE)


def get_photo_albums_dir() -> Path:
    if sys.platform.startswith("darwin"):
        home = Path(os.environ["HOME"])
        return home / "Library/CloudStorage/OneDrive-Personal/Cordell, Leslie & Audrey/Photo Albums"
    if sys.platform.startswith("win"):
        return Path("C:/Users/covec/OneDrive/Cordell, Leslie & Audrey/Photo Albums")
    raise NotImplementedError("Unsupported platform")


PHOTO_ALBUMS_DIR = get_photo_albums_dir()


def configure_imagemagick() -> None:
    if sys.platform.startswith("win"):
        magick_path = Path(r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI")
        os.environ["PATH"] = str(magick_path) + os.pathsep + os.environ["PATH"]


def list_archive_dirs(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("*_Archive") if p.is_dir()])


def parse_filename(
    filename: str,
    *patterns: re.Pattern,
    default: Tuple[str, str, str, str] = ("Unknown", "Unknown", "00", "00"),
) -> Tuple[str, str, str, str]:
    for pattern in patterns:
        m = pattern.search(filename)
        if m:
            return m.group("collection"), m.group("year"), m.group("book"), m.group("page")
    return default


def count_totals(
    archive_dirs: Iterable[Path],
    new_name_re: re.Pattern,
    parse_fn: Callable[[str], Tuple[str, str, str, str]],
) -> dict:
    totals = {}

    for archive in archive_dirs:
        for entry in archive.iterdir():
            if not entry.is_file():
                continue
            if not new_name_re.fullmatch(entry.name):
                continue

            collection, year, book, page = parse_fn(entry.name)
            key = f"{collection}_{year}_B{book}"

            if key not in totals:
                totals[key] = {"pages": set(), "page_scans": {}, "max_page": 0}

            m = PAGE_SCAN_RE.search(entry.name)
            if m:
                page_num = int(m.group("page"))
                scan_num = int(m.group("scan"))

                totals[key]["pages"].add(page_num)
                totals[key]["max_page"] = max(totals[key]["max_page"], page_num)

                page_scans = totals[key]["page_scans"]
                page_scans[page_num] = max(page_scans.get(page_num, 0), scan_num)

    for key, data in totals.items():
        unique_count = len(data["pages"])
        highest_page = data["max_page"]
        data["total_pages"] = highest_page if unique_count != highest_page else unique_count
        del data["pages"]
        del data["max_page"]

    return totals


def file_modified_ts(path: Path) -> float:
    return float(path.stat().st_mtime)


def dir_created_ts(path: str | Path) -> float:
    """
    Return a sortable timestamp for "created".
    - macOS: st_birthtime if available
    - Windows: st_ctime is creation time
    - Linux: no true creation time; fall back to mtime
    """
    st = Path(path).stat()
    if hasattr(st, "st_birthtime"):
        return float(st.st_birthtime)
    if sys.platform.startswith("win"):
        return float(st.st_ctime)
    return float(st.st_mtime)


def file_created_ts(path: str | Path) -> float:
    return dir_created_ts(path)


def derive_prefix(dir_path: str | Path) -> str:
    base = Path(dir_path).name
    if base.lower().endswith("_archive"):
        base = base[: -len("_archive")]
    return base


def get_next_filename(watch_dir: str | Path, filename_pattern: re.Pattern = FILENAME_PATTERN) -> str:
    watch_path = Path(watch_dir)
    files = [f.name for f in watch_path.iterdir() if f.suffix.lower() == ".tif"]
    valid_files = [f for f in files if filename_pattern.match(f)]

    if not valid_files:
        prefix = derive_prefix(watch_path)
        return f"{prefix}_P01_S01.tif"

    valid_files.sort()
    last_file = valid_files[-1]

    match = filename_pattern.match(last_file)
    prefix = match.group("prefix")
    page = int(match.group("page"))
    scan = int(match.group("scan"))

    if page == 1:
        page = 2
        scan = 1
    else:
        if scan < 2:
            scan += 1
        else:
            page += 1
            scan = 1

    return f"{prefix}_P{page:02d}_S{scan:02d}.tif"


def list_page_scan_groups(directory: str | Path, name_re: re.Pattern) -> List[List[str]]:
    dir_path = Path(directory)
    files = [f.name for f in dir_path.iterdir() if f.is_file() and name_re.fullmatch(f.name)]

    def key(name: str) -> Tuple[int, int]:
        m = PAGE_SCAN_RE.search(name)
        if not m:
            return 0, 0
        return int(m.group("page")), int(m.group("scan"))

    files.sort(key=key)

    pages: dict[int, List[str]] = {}
    for name in files:
        page = key(name)[0]
        pages.setdefault(page, []).append(str(dir_path / name))

    return list(pages.values())


def list_page_scans_for_page(directory: str | Path, page_num: int) -> List[str]:
    dir_path = Path(directory)
    files = []
    for entry in dir_path.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".tif", ".tiff"}:
            continue
        m = PAGE_SCAN_RE.search(entry.name)
        if not m or int(m.group("page")) != page_num:
            continue
        files.append(str(entry))

    def key(path: str) -> int:
        m = PAGE_SCAN_RE.search(path)
        return int(m.group("scan")) if m else 0

    files.sort(key=key)
    return files


def open_image_fullscreen(path: str, fallback_to_default: bool = False):
    xnview = Path(r"C:\Program Files\XnViewMP\xnviewmp.exe")
    if xnview.exists():
        return subprocess.Popen([str(xnview), path])
    if fallback_to_default and sys.platform.startswith("win"):
        os.startfile(path)
    return None


def rename_with_retry(
    old_path: str | Path,
    new_path: str | Path,
    *,
    attempts: int = 30,
    delay: float = 1.0,
    log_error: Callable[[str], None] | None = None,
) -> bool:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            os.rename(str(old_path), str(new_path))
            return True
        except (PermissionError, FileNotFoundError) as exc:
            last_error = exc
            time.sleep(delay)
        except Exception as exc:
            last_error = exc
            break

    if last_error is not None and log_error is not None:
        log_error(f"Rename failed: {last_error}")
    return False


def tiff_needs_conversion(tiff_path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-ExtraSamples",
                "-Compression",
                "-Predictor",
                str(tiff_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout.lower()
        has_alpha = bool(re.search(r"(alpha|\b[1-9]\b)", output))
        compression_ok = bool(re.search(r"\blzw\b", output))
        predictor_ok = bool(re.search(r"horizontal differencing", output))

        return has_alpha or (not compression_ok) or (not predictor_ok)
    except Exception:
        return True


def validate_pixels(original_path: Path, converted_path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "magick",
                "compare",
                "-metric",
                "AE",
                str(original_path),
                str(converted_path),
                "null:",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        diff_count = int(result.stderr.strip().split()[0])
        return diff_count == 0
    except Exception:
        return False


def process_tiff_in_place(
    tiff_path: Path,
    *,
    replace_attempts: int = 10,
    replace_delay: float = 0.5,
    log_error: Callable[[str], None] | None = None,
) -> bool:
    if not tiff_needs_conversion(tiff_path):
        return True

    fd, temp_name = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    tmp_path = Path(temp_name)

    def emit(message: str) -> None:
        if log_error is not None:
            log_error(message)

    try:
        subprocess.run(
            [
                "magick",
                str(tiff_path),
                "-alpha",
                "off",
                "-compress",
                "lzw",
                "-define",
                "tiff:predictor=2",
                str(tmp_path),
            ],
            check=True,
        )

        if not validate_pixels(tiff_path, tmp_path):
            emit(f"{tiff_path.name} pixel mismatch; not replacing")
            tmp_path.unlink(missing_ok=True)
            return False

        for _ in range(replace_attempts):
            try:
                tmp_path.replace(tiff_path)
                return True
            except PermissionError:
                time.sleep(replace_delay)
            except Exception as e:
                emit(f"{tiff_path.name} replace error: {e}")
                break

        emit(f"{tiff_path.name} replace permission denied")
        return False

    except subprocess.CalledProcessError as e:
        emit(f"{tiff_path.name} processing error: {e}")
        tmp_path.unlink(missing_ok=True)
        return False
