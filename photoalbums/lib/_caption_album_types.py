from __future__ import annotations

from dataclasses import dataclass

ALBUM_KIND_FAMILY = "family_photo_album"
ALBUM_KIND_PHOTO_ESSAY = "photo_essay"


@dataclass(frozen=True)
class AlbumContext:
    kind: str = ""
    label: str = ""
    focus: str = ""
    title: str = ""
    canonical_title: str = ""
    printed_title: str = ""
