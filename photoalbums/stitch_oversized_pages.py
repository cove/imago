import os
import re
import warnings
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    cv2 = None
    np = None
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from stitching import AffineStitcher
except Exception:
    AffineStitcher = None

from common import (
    CREATOR,
    PHOTO_ALBUMS_DIR,
    dir_created_ts,
    list_archive_dirs,
    list_page_scan_groups,
)
from exiftool_utils import write_tags
from naming import (
    BASE_PAGE_NAME_RE,
    SCAN_NAME_RE,
    SCAN_TIFF_RE,
    parse_album_filename,
)

MIN_OUTPUT_SIZE = 100 * 1024

NEW_NAME_RE = SCAN_TIFF_RE
DERIVED_RE = re.compile(r"_D(?P<d1>\d{2})_(?P<d2>\d{2})", re.IGNORECASE)
FILENAME_RE = SCAN_NAME_RE
FILENAME_RE_NO_SCAN = BASE_PAGE_NAME_RE

IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")


def _require_image_modules() -> None:
    if cv2 is None or np is None or Image is None:
        raise RuntimeError("cv2, numpy, and pillow are required for image processing.")


def _require_stitcher() -> None:
    if AffineStitcher is None:
        raise RuntimeError("stitching package is required for stitching.")


def build_scans_text(scan_nums: list[int]) -> str:
    return " ".join(f"S{s:02d}" for s in scan_nums)


def build_scan_header(
    collection: str,
    year: str,
    book: str,
    page: int,
    scan_nums: list[int],
) -> str:
    book_display = f"{int(book):02d}"
    scans_text = build_scans_text(scan_nums)
    return f"{collection} ({year}) - Book {book_display}, Page {page:02d}, Scans {scans_text}"


def extract_scan_numbers(files: list[str]) -> list[int]:
    scan_nums = []
    for file_path in files:
        match = re.search(r"_S(\d+)", file_path)
        if match:
            scan_nums.append(int(match.group(1)))
    return scan_nums


def build_source_filenames_text(files: list[str]) -> str:
    names = [os.path.basename(str(path)) for path in files if str(path).strip()]
    return "; ".join(names)


def build_detail_description(
    collection: str,
    year: str,
    book: str,
    page: int,
    d1: str,
    d2: str,
) -> str:
    book_display = f"{int(book):02d}"
    return (
        f"{collection} ({year}) - Book {book_display}, "
        f"Page {page:02d}, Detail D{d1}_{d2}"
    )


def build_derived_output_name(base: str) -> str:
    collection, year, book, page = parse_album_filename(base)
    m_d = DERIVED_RE.search(base)
    d1 = m_d.group("d1") if m_d else "00"
    d2 = m_d.group("d2") if m_d else "00"

    if collection != "Unknown":
        return f"{collection}_{year}_B{book}_P{int(page):02d}_D{d1}_{d2}.jpg"

    stem, _ = os.path.splitext(base)
    m_view = re.match(
        r"^(?P<collection>[A-Za-z]+)_(?P<year>\d{4}(?:-\d{4})?)_(?P<rest>.+)$",
        stem,
    )
    if m_view:
        return (
            f"{m_view.group('collection')}_{m_view.group('year')}_"
            f"{m_view.group('rest')}_D{d1}_{d2}.jpg"
        )
    return f"{stem}_D{d1}_{d2}.jpg"


def output_is_valid(path: str | Path, min_size: int = MIN_OUTPUT_SIZE) -> bool:
    path = Path(path)
    return path.exists() and path.stat().st_size > min_size


def get_view_dirname(path: str | Path) -> str:
    base = Path(path).name
    base_no_archive = base.replace("_Archive", "")
    match = re.match(
        r"^(?P<collection>[A-Za-z]+)_(?P<year>\d{4}(?:-\d{4})?)_(?P<rest>.+)$",
        base_no_archive,
    )
    if match:
        collection = match.group("collection")
        year = match.group("year")
        rest = match.group("rest")
        return str(Path(path).parent / f"{collection}_{year}_{rest}_View")
    return str(Path(path).parent / f"{base_no_archive}_View")


def add_bottom_header(image, date_text: str, header_text: str, margin: int = 15):
    _require_image_modules()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    width, height = pil_image.size

    font_size = 60
    small_font_size = 48

    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Apple Symbols.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    font = None
    small_font = None
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                small_font = ImageFont.truetype(font_path, small_font_size)
                break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    temp_img = Image.new("RGB", (1, 1))
    draw_temp = ImageDraw.Draw(temp_img)

    bbox = draw_temp.textbbox((0, 0), header_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    while text_width > width - 40 and font_size > 20:
        font_size = int(font_size * 0.9)
        small_font_size = int(font_size * 0.8)
        try:
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    small_font = ImageFont.truetype(font_path, small_font_size)
                    break
        except Exception:
            pass
        bbox = draw_temp.textbbox((0, 0), header_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    date_bbox = draw_temp.textbbox((0, 0), date_text, font=small_font)
    small_text_height = date_bbox[3] - date_bbox[1]

    line_spacing = margin * 2
    footer_height = text_height + small_text_height + (margin * 4) + line_spacing
    new_height = height + footer_height

    new_image = Image.new("RGB", (width, new_height), color="black")
    new_image.paste(pil_image, (0, 0))

    draw = ImageDraw.Draw(new_image)

    y1 = height + margin * 2
    x1 = (width - text_width) // 2
    draw.text((x1, y1), header_text, fill="white", font=font)

    y2 = y1 + text_height + line_spacing
    date_width = date_bbox[2] - date_bbox[0]
    draw.text((width - date_width - margin, y2), date_text, fill="white", font=small_font)

    result = np.array(new_image)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def list_page_scans(directory: str | Path):
    return list_page_scan_groups(directory, NEW_NAME_RE)


def list_derived_images(directory: str | Path) -> list[str]:
    files = []
    for name in os.listdir(directory):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        if not DERIVED_RE.search(name):
            continue
        files.append(os.path.join(directory, name))

    def key(path: str):
        base = os.path.basename(path)
        m_page = FILENAME_RE.search(base) or FILENAME_RE_NO_SCAN.search(base)
        m_d = DERIVED_RE.search(base)
        page = int(m_page.group("page")) if m_page else 0
        d1 = int(m_d.group("d1")) if m_d else 0
        d2 = int(m_d.group("d2")) if m_d else 0
        return page, d1, d2, base.lower()

    files.sort(key=key)
    return files


def write_jpeg(
    image,
    path: str | Path,
    header_text: str,
    quality: int = 95,
    extra_tags: dict[str, str] | None = None,
) -> None:
    _require_image_modules()
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    tags = {
        "XMP-dc:Creator": CREATOR,
        "XMP-dc:Description": header_text,
    }
    for tag, value in dict(extra_tags or {}).items():
        if str(tag or "").strip() and str(value or "").strip():
            tags[str(tag)] = str(value)
    write_tags(
        path,
        set_tags=tags,
    )


def tif_to_jpg(tif_path: str, output_dir: str) -> None:
    _require_image_modules()
    os.makedirs(output_dir, exist_ok=True)
    collection, year, book, page = parse_album_filename(os.path.basename(tif_path))
    out = os.path.join(
        output_dir,
        f"{collection}_{year}_B{book}_P{int(page):02d}.jpg",
    )

    scan_nums = extract_scan_numbers([tif_path]) or [1]
    jpg_header = build_scan_header(collection, year, book, int(page), scan_nums)

    if output_is_valid(out):
        print(f"{collection} B{book} P{int(page):02d} OK")
        return

    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Could not read image")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = add_bottom_header(
        img,
        f"Stitched: {datetime.now():%Y-%m-%d %H:%M:%S}",
        jpg_header,
    )

    write_jpeg(img, out, jpg_header)

    print(f"{collection} B{book} P{int(page):02d} OK")


def derived_to_jpg(src_path: str, output_dir: str) -> None:
    _require_image_modules()
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(src_path)
    collection, year, book, page = parse_album_filename(base)
    m_d = DERIVED_RE.search(base)
    d1 = m_d.group("d1") if m_d else "00"
    d2 = m_d.group("d2") if m_d else "00"

    out_name = build_derived_output_name(base)
    out = os.path.join(output_dir, out_name)

    if output_is_valid(out, min_size=1):
        if collection != "Unknown":
            print(f"{collection} B{book} P{int(page):02d} D{d1}_{d2} OK")
        else:
            print(f"{out_name} OK")
        return

    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    desc = ""
    if collection != "Unknown":
        desc = build_detail_description(collection, year, book, int(page), d1, d2)

    original_size = os.path.getsize(src_path)
    quality = 80
    write_jpeg(img, out, desc, quality=quality)

    while os.path.exists(out) and os.path.getsize(out) >= original_size and quality > 40:
        quality -= 10
        write_jpeg(img, out, desc, quality=quality)

    if collection != "Unknown":
        print(f"{collection} B{book} P{int(page):02d} D{d1}_{d2} OK")
    else:
        print(f"{out_name} OK")


def stitch(files, output_dir: str) -> None:
    _require_stitcher()
    _require_image_modules()
    os.makedirs(output_dir, exist_ok=True)

    collection, year, book, page = parse_album_filename(os.path.basename(files[0]))

    out = os.path.join(
        output_dir,
        f"{collection}_{year}_B{book}_P{int(page):02d}_stitched.jpg",
    )

    scan_nums = extract_scan_numbers(files)
    header = build_scan_header(collection, year, book, int(page), scan_nums)
    source_text = build_source_filenames_text(files)

    if output_is_valid(out):
        print(f"{collection} B{book} P{int(page):02d} OK")
        return

    attempts = [
       # {"detector": "sift", "confidence_threshold": 0.5},
        {"detector": "sift", "confidence_threshold": 0.3},
        {"detector": "sift", "confidence_threshold": 0.1},
        {"detector": "brisk", "confidence_threshold": 0.1},
    ]

    result = None
    partial_warning = None
    for cfg in attempts:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = AffineStitcher(**cfg).stitch(files)
            partial_warning = next(
                (
                    w
                    for w in caught
                    if "not all images are included in the final panorama"
                    in str(w.message).lower()
                ),
                None,
            )
            if partial_warning is not None:
                result = None
                continue
            if result is not None and result.size:
                break
        except Exception:
            pass

    if result is None:
        if partial_warning is not None:
            raise RuntimeError(
                "Stitching produced a partial panorama (not all scans were included)",
            )
        raise RuntimeError("All stitching attempts failed")

    write_jpeg(
        result,
        out,
        header,
        extra_tags={"XMP-dc:Source": source_text} if source_text else None,
    )

    print(f"{collection} B{book} P{int(page):02d} OK")


def main() -> None:
    success = failures = 0
    failed = []

    archive_dirs = list_archive_dirs(PHOTO_ALBUMS_DIR)
    archive_dirs.sort(key=dir_created_ts, reverse=True)

    for archive in archive_dirs:
        view = get_view_dirname(archive)

        for group in list_page_scans(archive):
            try:
                if len(group) > 1:
                    stitch(group, view)
                else:
                    tif_to_jpg(group[0], view)
                success += 1
            except Exception as exc:
                failures += 1
                failed.append(group)
                print("Error:", exc)

        for derived in list_derived_images(archive):
            try:
                derived_to_jpg(derived, view)
                success += 1
            except Exception as exc:
                failures += 1
                failed.append([derived])
                print("Error:", exc)

    print("\n===== SUMMARY =====")
    print("Successful:", success)
    print("Failed:", failures)
    if failed:
        print("\n===== FAILURES (DETAILS) =====")
        for group in failed:
            if group:
                collection, year, book, page = parse_album_filename(group[0])
                base = f"{collection}_{year}_B{book}_P{int(page):02d}"
            else:
                base = "Unknown"
            print(f"FAILED: {base}")
            print("Files:")
            for file_path in group:
                print(f"  - {file_path}")
    for group in failed:
        print(" -", ", ".join(os.path.basename(x) for x in group))


if __name__ == "__main__":
    main()
