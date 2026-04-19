from __future__ import annotations

from pathlib import Path

from ..naming import DERIVED_VIEW_RE, is_pages_dir, parse_album_filename


def _match_archives(photos_root: str | Path, album_id: str = "") -> list[Path]:
    from photoalbums.common import list_archive_dirs

    root = Path(photos_root)
    album_filter = str(album_id or "").casefold()
    archives = [Path(path) for path in list_archive_dirs(root)]
    if not album_filter:
        return sorted(archives)
    return sorted(path for path in archives if album_filter in path.name.casefold())


def _iter_expected_targets(archive_dir: Path, *, page: str | None = None) -> list[tuple[Path, Path]]:
    from photoalbums.stitch_oversized_pages import _derived_view_output_path, get_view_dirname, list_derived_images

    page_filter = f"{int(page):02d}" if str(page or "").strip().isdigit() else ""
    view_dir = Path(get_view_dirname(archive_dir))
    expected: list[tuple[Path, Path]] = []
    for derived_source in list_derived_images(archive_dir):
        source_path = Path(derived_source)
        _, _, _, page_str = parse_album_filename(source_path.name)
        if page_filter and page_str != page_filter:
            continue
        expected.append((source_path, _derived_view_output_path(source_path, view_dir)))
    return expected


def _iter_actual_page_views(view_dir: Path, *, page: str | None = None) -> list[Path]:
    if not view_dir.is_dir() or not is_pages_dir(view_dir):
        return []
    page_filter = f"{int(page):02d}" if str(page or "").strip().isdigit() else ""
    actual: list[Path] = []
    for path in sorted(view_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".jpg":
            continue
        if not DERIVED_VIEW_RE.search(path.stem):
            continue
        _, _, _, page_str = parse_album_filename(path.name)
        if page_filter and page_str != page_filter:
            continue
        actual.append(path)
    return actual


def _group_by_page(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        _, _, _, page_str = parse_album_filename(path.name)
        grouped.setdefault(page_str, []).append(path)
    return grouped


def _group_expected_by_page(expected_targets: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for source_path, target_path in expected_targets:
        _, _, _, page_str = parse_album_filename(source_path.name)
        grouped.setdefault(page_str, []).append((source_path, target_path))
    return grouped


def _image_fingerprint(path: Path) -> list[float]:
    from PIL import Image, ImageOps

    from .image_limits import allow_large_pillow_images

    allow_large_pillow_images(Image)
    with Image.open(path) as image:
        image.load()
        image = ImageOps.exif_transpose(image).convert("L").resize((24, 24))
        return [float(value) for value in image.tobytes()]


def _fingerprint_distance(left: list[float], right: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _pair_expected_and_actual(
    expected_targets: list[tuple[Path, Path]],
    actual_views: list[Path],
) -> tuple[list[tuple[Path, Path, Path]], list[Path], int]:
    expected_entries = [
        {
            "source": source_path,
            "target": target_path,
            "fingerprint": _image_fingerprint(source_path),
        }
        for source_path, target_path in expected_targets
    ]
    actual_entries = [
        {
            "jpg": jpg_path,
            "fingerprint": _image_fingerprint(jpg_path),
        }
        for jpg_path in actual_views
    ]

    candidate_pairs: list[tuple[float, int, int]] = []
    for expected_index, expected in enumerate(expected_entries):
        for actual_index, actual in enumerate(actual_entries):
            penalty = 0.0 if Path(expected["target"]).name == Path(actual["jpg"]).name else 1e-3
            distance = _fingerprint_distance(expected["fingerprint"], actual["fingerprint"]) + penalty
            candidate_pairs.append((distance, expected_index, actual_index))
    candidate_pairs.sort(key=lambda item: item[0])

    matched_expected: set[int] = set()
    matched_actual: set[int] = set()
    matches: list[tuple[Path, Path, Path]] = []
    for _, expected_index, actual_index in candidate_pairs:
        if expected_index in matched_expected or actual_index in matched_actual:
            continue
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)
        matches.append(
            (
                Path(expected_entries[expected_index]["source"]),
                Path(expected_entries[expected_index]["target"]),
                Path(actual_entries[actual_index]["jpg"]),
            )
        )

    orphan_views = [
        Path(actual_entries[index]["jpg"]) for index in range(len(actual_entries)) if index not in matched_actual
    ]
    missing_expected = len(expected_entries) - len(matches)
    return matches, orphan_views, missing_expected


def _temporary_path(path: Path, ordinal: int) -> Path:
    return path.with_name(f"{path.stem}.tmp-page-derived-repair-{ordinal}{path.suffix}")


def _make_writable(path: Path) -> None:
    if path.exists():
        path.chmod(0o666)


def _repair_page_view_set(
    expected_targets: list[tuple[Path, Path]],
    actual_views: list[Path],
) -> tuple[int, int, int]:
    matches, orphan_views, missing_expected = _pair_expected_and_actual(expected_targets, actual_views)

    planned: list[dict[str, Path | None]] = []
    for _, target_jpg, current_jpg in matches:
        current_xmp = current_jpg.with_suffix(".xmp")
        target_xmp = target_jpg.with_suffix(".xmp")
        if current_jpg == target_jpg:
            continue
        planned.append(
            {
                "current_jpg": current_jpg,
                "current_xmp": current_xmp if current_xmp.is_file() else None,
                "target_jpg": target_jpg,
                "target_xmp": target_xmp,
            }
        )

    staged: list[dict[str, Path | None]] = []
    staged_target_jpgs = {Path(pair["target_jpg"]) for pair in planned}
    staged_target_xmps = {Path(pair["target_xmp"]) for pair in planned}
    for ordinal, pair in enumerate(planned, start=1):
        current_jpg = Path(pair["current_jpg"])
        temp_jpg = _temporary_path(current_jpg, ordinal)
        _make_writable(current_jpg)
        current_jpg.rename(temp_jpg)
        temp_xmp: Path | None = None
        current_xmp = pair["current_xmp"]
        if isinstance(current_xmp, Path) and current_xmp.is_file():
            temp_xmp = _temporary_path(current_xmp, ordinal)
            _make_writable(current_xmp)
            current_xmp.rename(temp_xmp)
        staged.append(
            {
                "temp_jpg": temp_jpg,
                "temp_xmp": temp_xmp,
                "target_jpg": Path(pair["target_jpg"]),
                "target_xmp": Path(pair["target_xmp"]),
            }
        )

    for pair in staged:
        temp_jpg = Path(pair["temp_jpg"])
        target_jpg = Path(pair["target_jpg"])
        if target_jpg.exists() and target_jpg not in staged_target_jpgs:
            raise FileExistsError(f"Unstaged page-derived repair target already exists: {target_jpg}")
        temp_jpg.rename(target_jpg)
        temp_xmp = pair["temp_xmp"]
        if isinstance(temp_xmp, Path):
            target_xmp = Path(pair["target_xmp"])
            if target_xmp.exists() and target_xmp not in staged_target_xmps:
                raise FileExistsError(f"Unstaged page-derived repair target already exists: {target_xmp}")
            temp_xmp.rename(target_xmp)

    orphan_files_removed = 0
    for orphan_jpg in orphan_views:
        _make_writable(orphan_jpg)
        orphan_jpg.unlink(missing_ok=True)
        orphan_files_removed += 1
        orphan_xmp = orphan_jpg.with_suffix(".xmp")
        if orphan_xmp.is_file():
            _make_writable(orphan_xmp)
            orphan_xmp.unlink(missing_ok=True)
            orphan_files_removed += 1

    return len(planned), orphan_files_removed, missing_expected


def repair_page_derived_views(
    photos_root: str | Path,
    *,
    album_id: str = "",
    page: str | None = None,
) -> dict[str, int]:
    from photoalbums.stitch_oversized_pages import get_view_dirname

    root = Path(photos_root)
    if not root.exists():
        raise FileNotFoundError(f"Photo albums root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Photo albums root is not a directory: {root}")

    archives_scanned = 0
    files_scanned = 0
    files_renamed = 0
    orphan_files_removed = 0
    missing_expected_views = 0

    for archive_dir in _match_archives(root, album_id=album_id):
        archives_scanned += 1
        view_dir = Path(get_view_dirname(archive_dir))
        expected_targets = _iter_expected_targets(archive_dir, page=page)
        actual_views = _iter_actual_page_views(view_dir, page=page)
        expected_by_page = _group_expected_by_page(expected_targets)
        actual_by_page = _group_by_page(actual_views)
        all_pages = sorted(set(expected_by_page) | set(actual_by_page))
        for page_str in all_pages:
            page_expected = expected_by_page.get(page_str, [])
            page_actual = actual_by_page.get(page_str, [])
            files_scanned += len(page_actual)
            renamed_count, removed_count, missing_count = _repair_page_view_set(page_expected, page_actual)
            files_renamed += renamed_count
            orphan_files_removed += removed_count
            missing_expected_views += missing_count

    return {
        "archives_scanned": archives_scanned,
        "files_scanned": files_scanned,
        "files_renamed": files_renamed,
        "orphan_files_removed": orphan_files_removed,
        "missing_expected_views": missing_expected_views,
    }
