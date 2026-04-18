from __future__ import annotations

from pathlib import Path

from photoalbums.lib.page_reference_migration import (
    find_sidecars_with_view_references,
    migrate_album_page_references,
    rewrite_page_reference_text,
)
from photoalbums.naming import (
    ALBUM_DIR_SUFFIX_ARCHIVE,
    ALBUM_DIR_SUFFIX_PAGES,
    ALBUM_DIR_SUFFIX_PHOTOS,
    archive_dir_for_album_dir,
    pages_dir_for_album_dir,
    photos_dir_for_album_dir,
)


def test_album_sibling_helpers_resolve_canonical_dirs() -> None:
    archive_dir = Path("C:/Photos/Egypt_1975_B00_Archive")
    pages_dir = Path("C:/Photos/Egypt_1975_B00_Pages")
    photos_dir = Path("C:/Photos/Egypt_1975_B00_Photos")

    assert pages_dir_for_album_dir(archive_dir) == pages_dir
    assert pages_dir_for_album_dir(photos_dir) == pages_dir
    assert archive_dir_for_album_dir(pages_dir) == archive_dir
    assert photos_dir_for_album_dir(archive_dir) == photos_dir
    assert archive_dir.name.endswith(ALBUM_DIR_SUFFIX_ARCHIVE)
    assert pages_dir.name.endswith(ALBUM_DIR_SUFFIX_PAGES)
    assert photos_dir.name.endswith(ALBUM_DIR_SUFFIX_PHOTOS)


def test_rewrite_page_reference_text_rewrites_pages_refs_only() -> None:
    original = (
        "../Egypt_1975_B00_View/Egypt_1975_B00_P26_V.jpg "
        "../Egypt_1975_B00_View/Egypt_1975_B00_P26_V.xmp "
        "Album_View should stay untouched"
    )

    updated, replacements = rewrite_page_reference_text(original)

    assert replacements == 2
    assert "../Egypt_1975_B00_Pages/Egypt_1975_B00_P26_V.jpg" in updated
    assert "../Egypt_1975_B00_Pages/Egypt_1975_B00_P26_V.xmp" in updated
    assert "Album_View should stay untouched" in updated


def test_migrate_album_page_references_updates_only_matching_sidecars(tmp_path: Path) -> None:
    unchanged = tmp_path / "unchanged.xmp"
    changed = tmp_path / "changed.xmp"
    changed.write_text(
        (
            '<stRef:filePath>../Europe_1985_B02_View/Europe_1985_B02_P18_V.jpg</stRef:filePath>'
            '<dc:description>Album_View label</dc:description>'
        ),
        encoding="utf-8",
    )
    unchanged.write_text("<x:xmpmeta>no rewrite</x:xmpmeta>", encoding="utf-8")

    result = migrate_album_page_references(tmp_path)

    assert result == {"files_scanned": 2, "files_changed": 1, "replacements": 1}
    assert "../Europe_1985_B02_Pages/Europe_1985_B02_P18_V.jpg" in changed.read_text(encoding="utf-8")
    assert "Album_View label" in changed.read_text(encoding="utf-8")
    assert find_sidecars_with_view_references(tmp_path) == [changed]
