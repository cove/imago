import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

from photoalbums.lib.image_limits import allow_large_pillow_images

warnings.filterwarnings("ignore", message=".*decompression bomb.*", category=UserWarning)

_MAX_STITCH_IMAGE_PIXELS = str(1 << 40)
# Album page scans can exceed OpenCV's default pixel guard; lift it for stitching.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", _MAX_STITCH_IMAGE_PIXELS)

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
except Exception:
    cv2 = None
    np = None
    Image = None
    ImageOps = None

try:
    from stitching import AffineStitcher
except Exception:
    AffineStitcher = None

from photoalbums.common import (
    PHOTO_ALBUMS_DIR,
    configure_imagemagick,
    dir_created_ts,
    is_ignored_artifact_name,
    list_archive_dirs,
    list_page_scan_groups,
)
from photoalbums.naming import (
    ALBUM_DIR_SUFFIX_PAGES,
    BASE_PAGE_NAME_RE,
    SCAN_NAME_RE,
    SCAN_TIFF_RE,
    pages_dir_for_album_dir,
    photos_dir_for_album_dir,
    parse_album_filename,
)


NEW_NAME_RE = SCAN_TIFF_RE
DERIVED_RE = re.compile(r"_D(?P<d1>\d{2})-(?P<d2>\d{2})", re.IGNORECASE)
FILENAME_RE = SCAN_NAME_RE
FILENAME_RE_NO_SCAN = BASE_PAGE_NAME_RE

IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
MEDIA_EXTS = (".mp4", ".pdf")
LEGACY_DERIVED_RE = re.compile(r"_D(?P<d1>\d{2})_(?P<d2>\d{2})", re.IGNORECASE)

AFFINE_STITCH_ATTEMPTS = (
    {"detector": "sift", "confidence_threshold": 0.3},
    {"detector": "sift", "confidence_threshold": 0.1},
    {"detector": "akaze", "confidence_threshold": 0.3},
    {"detector": "akaze", "confidence_threshold": 0.1},
    {"detector": "brisk", "confidence_threshold": 0.1},
)

LINEAR_FALLBACK_TARGET_WIDTH = 640
LINEAR_FALLBACK_MIN_OVERLAP_FRAC = 0.08
LINEAR_FALLBACK_MAX_OVERLAP_FRAC = 0.42
LINEAR_FALLBACK_MAX_VERTICAL_SHIFT_FRAC = 0.08
LINEAR_FALLBACK_MIN_SHARED_HEIGHT_FRAC = 0.6
LINEAR_FALLBACK_MIN_DETAIL_FRAC = 0.05
LINEAR_FALLBACK_OVERLAP_STEP = 12
LINEAR_FALLBACK_VERTICAL_STEP = 4
LINEAR_FALLBACK_REFINE_OVERLAP_RADIUS = 12
LINEAR_FALLBACK_REFINE_VERTICAL_RADIUS = 4
LINEAR_FALLBACK_EXPANSION_RATIO = 1.02


def _require_image_modules() -> None:
    if cv2 is None or np is None or Image is None:
        raise RuntimeError("cv2, numpy, and pillow are required for image processing.")


def _require_stitcher() -> None:
    if AffineStitcher is None:
        raise RuntimeError("stitching package is required for stitching.")


def _match_derived_tokens(value: str):
    stem = Path(value).stem
    return re.fullmatch(r".+_D(?P<d1>\d{2})-(?P<d2>\d{2})", stem, re.IGNORECASE)


def build_derived_output_name(base: str, output_suffix: str = ".jpg") -> str:
    collection, year, book, page = parse_album_filename(base)
    m_d = _match_derived_tokens(base)
    d1 = m_d.group("d1") if m_d else "00"
    d2 = m_d.group("d2") if m_d else "00"

    if collection == "Unknown":
        stem, _ = os.path.splitext(base)
        base_match = FILENAME_RE_NO_SCAN.search(stem)
        if base_match is not None:
            collection = str(base_match.group("collection"))
            year = str(base_match.group("year"))
            book = str(base_match.group("book"))
            page = str(base_match.group("page"))

    if collection != "Unknown":
        return f"{collection}_{year}_B{book}_P{int(page):02d}_D{d1}-{d2}_V{output_suffix}"

    stem, _ = os.path.splitext(base)
    m_view = re.match(
        r"^(?P<collection>[A-Za-z]+)_(?P<year>\d{4}(?:-\d{4})?)_(?P<rest>.+)$",
        stem,
    )
    if m_view:
        return f"{m_view.group('collection')}_{m_view.group('year')}_{m_view.group('rest')}_D{d1}-{d2}_V{output_suffix}"
    return f"{stem}_D{d1}-{d2}_V{output_suffix}"


def output_is_valid(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def validate_image_with_pillow(path: str | Path) -> bool:
    _require_image_modules()
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _validate_and_retry(path: str | Path, max_retries: int = 1) -> bool:
    for attempt in range(max_retries + 1):
        if validate_image_with_pillow(path):
            return True
        if attempt < max_retries:
            print(f"Validation failed for {path}, retrying ({attempt + 1}/{max_retries})...")
    return False


def _skip_existing_output(out: str | Path, label: str) -> bool:
    if output_is_valid(out):
        if validate_image_with_pillow(out):
            print(f"{label} SKIP (existing output)")
            return True
        print(f"{label} RE-RENDER (existing output failed validation)")
        return False
    return False


def _ensure_bgr_image(image):
    _require_image_modules()
    if image is None:
        raise RuntimeError("Could not read image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _read_with_pillow(path: str | Path):
    _require_image_modules()
    allow_large_pillow_images(Image)
    with Image.open(path) as img:
        img.load()
        if ImageOps is not None:
            img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            arr = np.array(img)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        arr = np.array(img)
        if img.mode == "L":
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _read_with_magick(path: str | Path):
    _require_image_modules()
    configure_imagemagick()
    fd, tmp_name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        subprocess.run(
            [
                "magick",
                str(path),
                "-auto-orient",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        image = cv2.imread(str(tmp_path), cv2.IMREAD_UNCHANGED)
        return _ensure_bgr_image(image)
    finally:
        tmp_path.unlink(missing_ok=True)


def _read_stitch_image(path: str | Path):
    _require_image_modules()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        return _ensure_bgr_image(image)

    last_exc = None
    try:
        return _read_with_pillow(path)
    except Exception as exc:
        last_exc = exc

    try:
        return _read_with_magick(path)
    except Exception as exc:
        if last_exc is not None:
            raise RuntimeError("Could not read image") from exc
        raise RuntimeError("Could not read image") from exc


def _stitcher_factory(stitcher_factory=None):
    if stitcher_factory is not None:
        return stitcher_factory
    _require_stitcher()
    return AffineStitcher


def _estimate_background_color(image, border: int = 80) -> tuple[int, int, int]:
    h, w = image.shape[:2]
    border = max(8, min(border, h // 4, w // 4))
    samples = np.concatenate(
        [
            image[:border].reshape(-1, 3),
            image[-border:].reshape(-1, 3),
            image[:, :border].reshape(-1, 3),
            image[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    color = np.median(samples, axis=0)
    b, g, r = (int(c) for c in color.tolist())
    return (b, g, r)


def _build_overlap_feature_map(image):
    background = np.asarray(_estimate_background_color(image), dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high_pass = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3))  # type: ignore[arg-type]
    color_dist = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    feature_map = (color_dist * 0.7) + (high_pass * 1.8)
    return cv2.GaussianBlur(feature_map, (0, 0), 1.0)


def _score_linear_overlap(
    left_map,
    right_map,
    overlap: int,
    dy: int,
) -> float | None:
    height = left_map.shape[0]
    y_left = max(0, dy)
    y_right = max(0, -dy)
    shared_height = min(height - y_left, right_map.shape[0] - y_right)
    if shared_height < int(height * LINEAR_FALLBACK_MIN_SHARED_HEIGHT_FRAC):
        return None

    left_strip = left_map[y_left : y_left + shared_height, left_map.shape[1] - overlap :]
    right_strip = right_map[y_right : y_right + shared_height, :overlap]
    if left_strip.size == 0 or right_strip.size == 0:
        return None

    left_thresh = float(np.percentile(left_strip, 70))
    right_thresh = float(np.percentile(right_strip, 70))
    detail_mask = (left_strip > left_thresh) | (right_strip > right_thresh)
    if float(detail_mask.mean()) < LINEAR_FALLBACK_MIN_DETAIL_FRAC:
        return None

    diff = np.abs(left_strip - right_strip)
    return float(diff[detail_mask].mean())


def _search_linear_overlap(left_img, right_img) -> tuple[float, int, int]:
    width = max(left_img.shape[1], right_img.shape[1])
    scale = min(1.0, LINEAR_FALLBACK_TARGET_WIDTH / max(width, 1))
    if scale < 1.0:
        left_small = cv2.resize(
            left_img,
            (
                max(1, int(left_img.shape[1] * scale)),
                max(1, int(left_img.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        right_small = cv2.resize(
            right_img,
            (
                max(1, int(right_img.shape[1] * scale)),
                max(1, int(right_img.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        left_small = left_img
        right_small = right_img

    left_map = _build_overlap_feature_map(left_small)
    right_map = _build_overlap_feature_map(right_small)

    min_overlap = max(
        8,
        int(min(left_map.shape[1], right_map.shape[1]) * LINEAR_FALLBACK_MIN_OVERLAP_FRAC),
    )
    max_overlap = max(
        min_overlap,
        int(min(left_map.shape[1], right_map.shape[1]) * LINEAR_FALLBACK_MAX_OVERLAP_FRAC),
    )
    max_dy = int(max(left_map.shape[0], right_map.shape[0]) * LINEAR_FALLBACK_MAX_VERTICAL_SHIFT_FRAC)

    best: tuple[float, int, int] | None = None
    for overlap in range(min_overlap, max_overlap + 1, LINEAR_FALLBACK_OVERLAP_STEP):
        for dy in range(-max_dy, max_dy + 1, LINEAR_FALLBACK_VERTICAL_STEP):
            score = _score_linear_overlap(left_map, right_map, overlap, dy)
            if score is None:
                continue
            candidate = (score, overlap, dy)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("Linear overlap search could not find a shared region")

    _, coarse_overlap, coarse_dy = best
    refine_overlap_min = max(min_overlap, coarse_overlap - LINEAR_FALLBACK_REFINE_OVERLAP_RADIUS)
    refine_overlap_max = min(max_overlap, coarse_overlap + LINEAR_FALLBACK_REFINE_OVERLAP_RADIUS)
    refine_dy_min = max(-max_dy, coarse_dy - LINEAR_FALLBACK_REFINE_VERTICAL_RADIUS)
    refine_dy_max = min(max_dy, coarse_dy + LINEAR_FALLBACK_REFINE_VERTICAL_RADIUS)
    for overlap in range(refine_overlap_min, refine_overlap_max + 1, 2):
        for dy in range(refine_dy_min, refine_dy_max + 1):
            score = _score_linear_overlap(left_map, right_map, overlap, dy)
            if score is None:
                continue
            candidate = (score, overlap, dy)
            if candidate < best:
                best = candidate

    score, overlap, dy = best
    return score, int(round(overlap / scale)), int(round(dy / scale))


def _compose_linear_pair(left_img, right_img, overlap: int, dy: int):
    left_h, left_w = left_img.shape[:2]
    right_h, right_w = right_img.shape[:2]
    overlap = max(1, min(overlap, left_w - 1, right_w - 1))
    x_right = left_w - overlap
    min_y = min(0, dy)
    max_y = max(left_h, dy + right_h)
    out_h = max_y - min_y
    out_w = max(left_w, x_right + right_w)

    fill_color = tuple(
        int(c)
        for c in np.median(
            np.vstack(
                [
                    np.asarray(_estimate_background_color(left_img), dtype=np.float32),
                    np.asarray(_estimate_background_color(right_img), dtype=np.float32),
                ]
            ),
            axis=0,
        ).tolist()
    )
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    out[:] = fill_color

    y_left = -min_y
    y_right = dy - min_y
    out[y_left : y_left + left_h, :left_w] = left_img

    right_roi = out[y_right : y_right + right_h, x_right : x_right + right_w]
    if overlap > 0:
        left_overlap = right_roi[:, :overlap]
        right_overlap = right_img[:, :overlap]
        alpha = np.linspace(1.0, 0.0, overlap, dtype=np.float32)[None, :, None]
        blended = (left_overlap.astype(np.float32) * alpha) + (right_overlap.astype(np.float32) * (1.0 - alpha))
        right_roi[:, :overlap] = np.clip(blended, 0, 255).astype(np.uint8)
    right_roi[:, overlap:] = right_img[:, overlap:]
    return out


def _result_expands_canvas(result, images) -> bool:
    shape = getattr(result, "shape", None)
    if not isinstance(shape, tuple) or len(shape) < 2:
        return True
    base_h = max(img.shape[0] for img in images)
    base_w = max(img.shape[1] for img in images)
    return shape[1] >= int(base_w * LINEAR_FALLBACK_EXPANSION_RATIO) or shape[0] >= int(
        base_h * LINEAR_FALLBACK_EXPANSION_RATIO
    )


def _stitch_linear_pair_images(images):
    _require_image_modules()
    if len(images) != 2:
        raise RuntimeError("Linear page fallback only supports two scans")

    cv2.ocl.setUseOpenCL(False)
    normalized = [_ensure_bgr_image(img) for img in images]
    candidates = []
    for left_idx, right_idx in ((0, 1), (1, 0)):
        score, overlap, dy = _search_linear_overlap(
            normalized[left_idx],
            normalized[right_idx],
        )
        composed = _compose_linear_pair(
            normalized[left_idx],
            normalized[right_idx],
            overlap,
            dy,
        )
        candidates.append((score, composed))

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def build_stitched_image(files, stitcher_factory=None):
    stitcher_factory = _stitcher_factory(stitcher_factory)
    attempts = [dict(cfg) for cfg in AFFINE_STITCH_ATTEMPTS]

    partial_warning = None
    last_exc: Exception | None = None
    loaded_images = None

    def ensure_loaded_images():
        nonlocal loaded_images
        if loaded_images is None:
            loaded_images = [_read_stitch_image(path) for path in files]
        return loaded_images

    for cfg in attempts:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = stitcher_factory(**cfg).stitch(files)
            partial_warning = next(
                (w for w in caught if "not all images are included in the final panorama" in str(w.message).lower()),
                None,
            )
            if partial_warning is not None:
                result = None
                continue
            if result is not None and getattr(result, "size", 0):
                shape = getattr(result, "shape", None)
                if (not isinstance(shape, tuple) or len(shape) < 2) or _result_expands_canvas(
                    result,
                    ensure_loaded_images(),
                ):
                    return result
        except Exception as exc:
            last_exc = exc

    if len(files) == 2:
        try:
            fallback = _stitch_linear_pair_images(ensure_loaded_images())
            if getattr(fallback, "size", 0) and _result_expands_canvas(
                fallback,
                ensure_loaded_images(),
            ):
                return fallback
        except Exception as exc:
            last_exc = exc

    if partial_warning is not None:
        raise RuntimeError(
            "Stitching produced a partial panorama (not all scans were included)",
        )
    if last_exc is not None:
        raise RuntimeError("All stitching attempts failed") from last_exc
    raise RuntimeError("All stitching attempts failed")


def get_view_dirname(path: str | Path) -> str:
    return str(pages_dir_for_album_dir(path))


def get_photos_dirname(path: str | Path) -> str:
    return str(photos_dir_for_album_dir(path))


def _scan_number(path: str | Path) -> int:
    match = SCAN_NAME_RE.search(Path(path).name)
    if match is None:
        return 0
    return int(match.group("scan"))


def _require_primary_scan(files: list[str]) -> str:
    ordered = sorted(files, key=_scan_number)
    primary = next((path for path in ordered if _scan_number(path) == 1), "")
    if not primary:
        raise RuntimeError(f"Page is missing required S01 scan: {', '.join(Path(path).name for path in ordered)}")
    return primary


def _view_page_output_path(scan_path: str | Path, output_dir: str | Path) -> Path:
    collection, year, book, page = parse_album_filename(Path(scan_path).name)
    return Path(output_dir) / f"{collection}_{year}_B{book}_P{int(page):02d}_V.jpg"


def _derived_view_output_path(src_path: str | Path, output_dir: str | Path) -> Path:
    return Path(output_dir) / build_derived_output_name(Path(src_path).name)


def _ensure_archive_page_sidecar(scan_path: str | Path) -> Path:
    scan_path = Path(scan_path)
    sidecar_path = scan_path.with_suffix(".xmp")
    if not (sidecar_path.is_file() and sidecar_path.stat().st_size > 0):
        from photoalbums.lib import ai_index  # pylint: disable=import-outside-toplevel

        result = int(ai_index.run(["--photo", str(scan_path)]) or 0)
        if result != 0:
            raise RuntimeError(f"AI index failed while creating archive sidecar: {scan_path}")
        if not sidecar_path.is_file() or sidecar_path.stat().st_size <= 0:
            raise RuntimeError(f"Archive sidecar was not created for render output: {sidecar_path}")
    from photoalbums.lib.xmpmm_provenance import assign_document_id  # pylint: disable=import-outside-toplevel

    assign_document_id(sidecar_path)
    return sidecar_path


def write_render_provenance(view_xmp_path: str | Path, scan_files: list[str | Path]) -> None:
    """Assign DocumentID to the view XMP and write DerivedFrom/Pantry from archive scans.

    DerivedFrom points to the primary scan (S01); Pantry lists every contributing scan.
    """
    from photoalbums.lib.xmpmm_provenance import (  # pylint: disable=import-outside-toplevel
        assign_document_id,
        read_document_id,
        write_creation_provenance,
    )

    view_xmp_path = Path(view_xmp_path)
    assign_document_id(view_xmp_path)

    ordered = sorted(scan_files, key=lambda p: _scan_number(p))
    primary = next((p for p in ordered if _scan_number(p) == 1), ordered[0] if ordered else None)
    if primary is None:
        return

    primary_doc_id = read_document_id(Path(primary).with_suffix(".xmp"))
    pantry_sources = []
    for scan in ordered:
        doc_id = read_document_id(Path(scan).with_suffix(".xmp"))
        if doc_id:
            pantry_sources.append({"source_document_id": doc_id, "source_path": Path(scan).name})

    write_creation_provenance(
        view_xmp_path,
        derived_from={"source_document_id": primary_doc_id, "source_path": Path(primary).name},
        pantry_sources=pantry_sources,
    )


def _copy_base_view_sidecar(scan_path: str | Path, output_dir: str | Path) -> Path:
    source_sidecar = _ensure_archive_page_sidecar(scan_path)
    target_sidecar = _view_page_output_path(scan_path, output_dir).with_suffix(".xmp")
    target_sidecar.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sidecar, target_sidecar)
    return target_sidecar


def _index_rendered_view_image(image_path: str | Path) -> None:
    image_path = Path(image_path)
    from photoalbums.lib import ai_index  # pylint: disable=import-outside-toplevel

    result = int(ai_index.run(["--photo", str(image_path)]) or 0)
    if result != 0:
        raise RuntimeError(f"AI index failed for rendered view image: {image_path}")


def _refresh_rendered_view_people(image_path: str | Path) -> None:
    image_path = Path(image_path)
    from photoalbums.lib.ai_index_runner import (  # pylint: disable=import-outside-toplevel
        refresh_rendered_view_people_metadata,
    )

    refresh_rendered_view_people_metadata(image_path)


def apply_ctm_to_image(image, matrix: list[float] | tuple[float, ...]):
    if len(matrix) != 9:
        raise RuntimeError("CTM matrix must contain 9 coefficients")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError("CTM application requires an RGB image")
    transform = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    rgb = array[:, :, :3].astype(np.float32) / 255.0
    corrected = np.einsum("...c,dc->...d", rgb, transform)
    corrected = np.clip(corrected, 0.0, 1.0)
    output = array.copy()
    output[:, :, :3] = np.rint(corrected * 255.0).astype(np.uint8)
    return output


def list_page_scans(directory: str | Path):
    return list_page_scan_groups(directory, NEW_NAME_RE)


def _sort_key(path: str):
    base = os.path.basename(path)
    m_page = FILENAME_RE.search(base) or FILENAME_RE_NO_SCAN.search(base)
    m_d = _match_derived_tokens(base)
    page = int(m_page.group("page")) if m_page else 0
    d1 = int(m_d.group("d1")) if m_d else 0
    d2 = int(m_d.group("d2")) if m_d else 0
    return page, d1, d2, base.lower()


def list_derived_images(directory: str | Path) -> list[str]:
    return _list_derived_media(directory, IMAGE_EXTS)


def list_derived_media(directory: str | Path) -> list[str]:
    return _list_derived_media(directory, MEDIA_EXTS)


def _list_derived_media(directory: str | Path, extensions: tuple[str, ...]) -> list[str]:
    files = []
    for name in os.listdir(directory):
        if is_ignored_artifact_name(name):
            continue
        if not name.lower().endswith(extensions):
            continue
        if not _match_derived_tokens(name):
            continue
        files.append(os.path.join(directory, name))

    files.sort(key=_sort_key)
    return files


def copy_derived_media(src_path: str, output_dir: str) -> bool:
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(src_path)
    suffix = Path(base).suffix.lower()
    out_name = build_derived_output_name(base, output_suffix=suffix)
    out = os.path.join(output_dir, out_name)
    label = out_name
    if Path(out).exists() and Path(out).stat().st_size > 0:
        print(f"{label} SKIP (existing output)")
        return False

    shutil.copy2(src_path, out)
    print(f"{label} OK")
    return True


def write_jpeg(image, path: str | Path, quality: int = 95) -> None:
    _require_image_modules()
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Failed to write image (permission denied or unsupported format): {path}")


def tif_to_jpg(tif_path: str, output_dir: str) -> bool:
    _require_image_modules()
    os.makedirs(output_dir, exist_ok=True)
    collection, year, book, page = parse_album_filename(os.path.basename(tif_path))
    out = os.path.join(
        output_dir,
        f"{collection}_{year}_B{book}_P{int(page):02d}_V.jpg",
    )
    label = f"{collection} B{book} P{int(page):02d}"
    if _skip_existing_output(out, label):
        xmp_out = Path(out).with_suffix(".xmp")
        if not xmp_out.exists():
            write_render_provenance(xmp_out, [tif_path])
        return False

    if not _validate_and_retry(tif_path):
        raise RuntimeError(f"Input validation failed: {tif_path}")

    img = _read_stitch_image(tif_path)
    write_jpeg(img, out)

    if not validate_image_with_pillow(out):
        raise RuntimeError(f"Output validation failed: {out}")

    write_render_provenance(Path(out).with_suffix(".xmp"), [tif_path])

    print(f"{label} OK")
    return True


def derived_to_jpg(src_path: str, output_dir: str, *, force: bool = False) -> bool:
    _require_image_modules()
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(src_path)
    collection, year, book, page = parse_album_filename(base)
    m_d = DERIVED_RE.search(base)
    d1 = m_d.group("d1") if m_d else "00"
    d2 = m_d.group("d2") if m_d else "00"

    out_name = build_derived_output_name(base)
    out = os.path.join(output_dir, out_name)
    if collection != "Unknown":
        label = f"{collection} B{book} P{int(page):02d} D{d1}_{d2}"
    else:
        label = out_name
    if not force and _skip_existing_output(out, label):
        xmp_out = Path(out).with_suffix(".xmp")
        if not xmp_out.exists():
            write_render_provenance(xmp_out, [src_path])
        return False

    if not _validate_and_retry(src_path):
        raise RuntimeError(f"Input validation failed: {src_path}")

    img = _read_stitch_image(src_path)

    original_size = os.path.getsize(src_path)
    quality = 80
    write_jpeg(img, out, quality=quality)

    while os.path.exists(out) and os.path.getsize(out) >= original_size and quality > 40:
        quality -= 10
        write_jpeg(img, out, quality=quality)

    if not validate_image_with_pillow(out):
        raise RuntimeError(f"Output validation failed: {out}")

    write_render_provenance(Path(out).with_suffix(".xmp"), [src_path])

    print(f"{label} OK")
    return True


def stitch(files, output_dir: str) -> bool:
    _require_image_modules()
    os.makedirs(output_dir, exist_ok=True)
    _require_primary_scan(files)

    collection, year, book, page = parse_album_filename(os.path.basename(files[0]))

    out = os.path.join(
        output_dir,
        f"{collection}_{year}_B{book}_P{int(page):02d}_V.jpg",
    )
    label = f"{collection} B{book} P{int(page):02d}"
    if _skip_existing_output(out, label):
        xmp_out = Path(out).with_suffix(".xmp")
        if not xmp_out.exists():
            write_render_provenance(xmp_out, files)
        return False

    for f in files:
        if not _validate_and_retry(f):
            raise RuntimeError(f"Input validation failed: {f}")

    result = build_stitched_image(files)
    write_jpeg(result, out)

    if not validate_image_with_pillow(out):
        raise RuntimeError(f"Output validation failed: {out}")

    write_render_provenance(Path(out).with_suffix(".xmp"), files)

    print(f"{label} OK")
    return True


def main() -> None:
    success = skipped = failures = 0
    failed = []

    archive_dirs = list_archive_dirs(PHOTO_ALBUMS_DIR)
    archive_dirs.sort(key=dir_created_ts, reverse=True)

    for archive in archive_dirs:
        view = get_view_dirname(archive)
        photos = get_photos_dirname(archive)

        for group in list_page_scans(archive):
            try:
                primary_scan = _require_primary_scan(group)
                if len(group) > 1:
                    wrote = stitch(group, view)
                else:
                    wrote = tif_to_jpg(primary_scan, view)
                page_view_path = _view_page_output_path(primary_scan, view)
                if wrote or not page_view_path.with_suffix(".xmp").is_file():
                    _copy_base_view_sidecar(primary_scan, view)
                    _refresh_rendered_view_people(page_view_path)
                if wrote:
                    success += 1
                else:
                    skipped += 1
            except Exception as exc:
                failures += 1
                failed.append(group)
                print("Error:", exc)

        for derived in list_derived_images(archive):
            try:
                wrote = derived_to_jpg(derived, photos)
                derived_view_path = _derived_view_output_path(derived, photos)
                if wrote or not derived_view_path.with_suffix(".xmp").is_file():
                    _index_rendered_view_image(derived_view_path)
                    _refresh_rendered_view_people(derived_view_path)
                if wrote:
                    success += 1
                else:
                    skipped += 1
            except Exception as exc:
                failures += 1
                failed.append([derived])
                print("Error:", exc)

        for media_path in list_derived_media(archive):
            try:
                wrote = copy_derived_media(media_path, photos)
                if wrote:
                    success += 1
                else:
                    skipped += 1
            except Exception as exc:
                failures += 1
                failed.append([media_path])
                print("Error:", exc)

    print("\n===== SUMMARY =====")
    print("Successful:", success)
    print("Skipped:", skipped)
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
