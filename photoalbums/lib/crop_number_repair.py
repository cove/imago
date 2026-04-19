from __future__ import annotations

import re
from pathlib import Path

from .ai_photo_crops import crop_output_path, highest_archive_derived_number
from ..naming import is_photos_dir, pages_dir_for_album_dir, parse_album_filename

_CROP_STEM_RE = re.compile(r"^(?P<page_prefix>.+)_D(?P<derived>\d+)-00_V$", re.IGNORECASE)


def _iter_target_photo_dirs(photos_root: str | Path, album_id: str = "") -> list[Path]:
    root = Path(photos_root)
    album_filter = str(album_id or "").casefold()
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and is_photos_dir(path) and (not album_filter or album_filter in path.name.casefold())
    )


def _collect_page_crop_pairs(photos_dir: Path, *, page: str | None = None) -> dict[str, list[dict[str, Path | int]]]:
    page_filter = f"{int(page):02d}" if str(page or "").strip().isdigit() else ""
    grouped: dict[str, dict[str, dict[str, Path | int | None]]] = {}

    for path in sorted(photos_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".xmp"}:
            continue
        match = _CROP_STEM_RE.fullmatch(path.stem)
        if match is None:
            continue
        _, _, _, page_str = parse_album_filename(path.name)
        if page_filter and page_str != page_filter:
            continue
        page_prefix = str(match.group("page_prefix"))
        stem = path.stem
        pair = grouped.setdefault(page_prefix, {}).setdefault(
            stem,
            {
                "jpg": None,
                "xmp": None,
                "derived": int(match.group("derived")),
            },
        )
        if path.suffix.lower() == ".jpg":
            pair["jpg"] = path
        else:
            pair["xmp"] = path

    collected: dict[str, list[dict[str, Path | int]]] = {}
    for page_prefix, pairs_by_stem in grouped.items():
        page_pairs: list[dict[str, Path | int]] = []
        for stem, pair in sorted(pairs_by_stem.items(), key=lambda item: int(item[1]["derived"])):
            jpg_path = pair.get("jpg")
            xmp_path = pair.get("xmp")
            if not isinstance(jpg_path, Path) or not isinstance(xmp_path, Path):
                raise FileNotFoundError(f"Crop pair repair failed due to missing companion file for {photos_dir / stem}")
            page_pairs.append(
                {
                    "jpg": jpg_path,
                    "xmp": xmp_path,
                    "derived": int(pair["derived"]),
                }
            )
        collected[page_prefix] = page_pairs
    return collected


def _temporary_pair_path(path: Path, ordinal: int) -> Path:
    return path.with_name(f"{path.stem}.tmp-crop-number-repair-{ordinal}{path.suffix}")


def repair_album_crop_numbers(
    photos_root: str | Path,
    *,
    album_id: str = "",
    page: str | None = None,
) -> dict[str, object]:
    root = Path(photos_root)
    if not root.exists():
        raise FileNotFoundError(f"Photo albums root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Photo albums root is not a directory: {root}")

    pages_scanned = 0
    pages_changed = 0
    files_scanned = 0
    files_changed = 0
    renames: list[dict[str, str]] = []

    for photos_dir in _iter_target_photo_dirs(root, album_id=album_id):
        pages_dir = pages_dir_for_album_dir(photos_dir)
        page_pairs_by_prefix = _collect_page_crop_pairs(photos_dir, page=page)
        for page_prefix, page_pairs in sorted(page_pairs_by_prefix.items()):
            pages_scanned += 1
            files_scanned += len(page_pairs)
            view_path = pages_dir / f"{page_prefix}_V.jpg"
            archive_max_derived = highest_archive_derived_number(view_path)

            planned: list[dict[str, Path]] = []
            for index, pair in enumerate(page_pairs, start=1):
                target_jpg = crop_output_path(
                    view_path,
                    index,
                    photos_dir,
                    archive_max_derived=archive_max_derived,
                )
                target_xmp = target_jpg.with_suffix(".xmp")
                current_jpg = pair["jpg"]
                current_xmp = pair["xmp"]
                assert isinstance(current_jpg, Path)
                assert isinstance(current_xmp, Path)
                if current_jpg == target_jpg and current_xmp == target_xmp:
                    continue
                planned.append(
                    {
                        "current_jpg": current_jpg,
                        "current_xmp": current_xmp,
                        "target_jpg": target_jpg,
                        "target_xmp": target_xmp,
                    }
                )

            if not planned:
                continue

            staged: list[dict[str, Path]] = []
            for ordinal, pair in enumerate(planned, start=1):
                temp_jpg = _temporary_pair_path(pair["current_jpg"], ordinal)
                temp_xmp = _temporary_pair_path(pair["current_xmp"], ordinal)
                if temp_jpg.exists() or temp_xmp.exists():
                    raise FileExistsError(f"Temporary crop repair path already exists: {temp_jpg}")
                pair["current_jpg"].rename(temp_jpg)
                pair["current_xmp"].rename(temp_xmp)
                staged.append(
                    {
                        "temp_jpg": temp_jpg,
                        "temp_xmp": temp_xmp,
                        "target_jpg": pair["target_jpg"],
                        "target_xmp": pair["target_xmp"],
                        "old_jpg": pair["current_jpg"],
                        "old_xmp": pair["current_xmp"],
                    }
                )

            for pair in staged:
                target_jpg = pair["target_jpg"]
                target_xmp = pair["target_xmp"]
                temp_jpg = pair["temp_jpg"]
                temp_xmp = pair["temp_xmp"]
                if target_jpg.exists() or target_xmp.exists():
                    raise FileExistsError(f"Crop repair target already exists and was not staged away: {target_jpg}")
                temp_jpg.rename(target_jpg)
                temp_xmp.rename(target_xmp)
                renames.append(
                    {
                        "old_jpg": str(pair["old_jpg"]),
                        "new_jpg": str(target_jpg),
                        "old_xmp": str(pair["old_xmp"]),
                        "new_xmp": str(target_xmp),
                    }
                )

            pages_changed += 1
            files_changed += len(planned)

    return {
        "pages_scanned": pages_scanned,
        "pages_changed": pages_changed,
        "files_scanned": files_scanned,
        "files_changed": files_changed,
        "renames": renames,
    }
