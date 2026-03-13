from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .matching import suggest_people
from .storage import TextFaceStore
from .xmp_writer import merge_persons_xmp, read_person_in_image

_DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
_DIVIDER = "─" * 60


@dataclass
class LabelResult:
    photo_path: Path
    status: str  # labeled | skipped_existing | skipped_no_faces | skipped_by_user
    faces_labeled: int = 0
    person_names: list[str] = field(default_factory=list)


def iter_photos(
    directory: Path,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> Iterator[Path]:
    """Recursively yield all photo files in directory, sorted."""
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in ext_set:
            yield path


def get_faces_for_photo(source_path: str, store: TextFaceStore) -> list[dict[str, Any]]:
    """Return existing face records for a photo path (avoids re-ingesting)."""
    normalized = str(source_path).replace("\\", "/")
    return [
        f for f in store.list_faces()
        if str(f.get("source_path", "")).replace("\\", "/") == normalized
    ]


def get_ai_suggestions(
    face: dict[str, Any],
    store: TextFaceStore,
    *,
    top_k: int = 3,
    min_similarity: float = 0.30,
) -> list[dict[str, Any]]:
    """
    Run suggest_people() on a face's embedding and resolve person_id -> display_name.
    Returns list of {person_id, display_name, score}.
    """
    emb = face.get("embedding")
    if emb is None:
        return []
    all_faces = [f for f in store.list_faces() if f.get("person_id")]
    if not all_faces:
        return []
    try:
        raw = suggest_people(
            query_embedding=emb,
            faces=all_faces,
            top_k=int(top_k),
            min_similarity=float(min_similarity),
        )
    except Exception:
        return []
    people_by_id = {
        str(p.get("person_id")): str(p.get("display_name", ""))
        for p in store.list_people()
    }
    out = []
    for r in raw:
        person_id = str(r.get("person_id", "")).strip()
        name = people_by_id.get(person_id, "")
        if not name:
            continue
        out.append({"person_id": person_id, "display_name": name, "score": float(r.get("score", 0.0))})
    return out


def resolve_or_create_person(name: str, store: TextFaceStore) -> dict[str, Any]:
    """
    Case-insensitive lookup for a person by display_name. Creates a new person
    record if not found. Returns the person dict.
    """
    name = name.strip()
    key = name.casefold()
    for person in store.list_people():
        if str(person.get("display_name", "")).casefold() == key:
            return person
        for alias in list(person.get("aliases") or []):
            if str(alias).casefold() == key:
                return person
    return store.add_person(display_name=name)


def open_crop_for_viewing(crop_path: Path) -> None:
    """Open the face crop in the default image viewer (Windows), or print the path."""
    print(f"  Crop: {crop_path}")
    if sys.platform == "win32":
        try:
            import os
            os.startfile(str(crop_path))
            print("  [Opening in default viewer...]")
        except Exception:
            pass


def _prompt_for_face(
    photo_path: Path,
    face_index: int,
    total_faces: int,
    suggestions: list[dict[str, Any]],
    crop_path: Path | None,
    *,
    existing_person: str | None,
) -> str | None:
    """
    Display the prompt for one face and return the raw user input string.
    Returns None to skip. Raises SystemExit on 'q'.
    """
    print(f"\n  Face {face_index} of {total_faces}", end="")
    if crop_path and crop_path.exists():
        open_crop_for_viewing(crop_path)
    else:
        print()

    if existing_person:
        print(f"  Currently assigned: {existing_person}")

    if suggestions:
        print("  AI suggestions:")
        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s['display_name']}  ({s['score']:.2f})")
    else:
        print("  AI suggestions: (none)")

    while True:
        try:
            raw = input("\n  Names (comma-sep), 's' skip, 'q' quit: ").strip()
        except EOFError:
            return None

        if raw.lower() in ("q", "quit"):
            print("\nQuitting.")
            raise SystemExit(0)
        if raw.lower() in ("s", "skip", ""):
            return None
        if raw.lower() in ("?", "help"):
            print("  Enter comma-separated names, 's' to skip this face, 'q' to quit.")
            continue
        return raw


def label_photo(
    photo_path: Path,
    store: TextFaceStore,
    ingestor: Any,
    *,
    overwrite: bool = False,
) -> LabelResult:
    """
    Process a single photo:
    1. Skip if XMP already has PersonInImage (unless overwrite).
    2. Reuse or ingest face records.
    3. Prompt user for each face.
    4. Assign face->person in Cast DB and write XMP sidecar.
    """
    xmp_path = photo_path.with_suffix(".xmp")

    if not overwrite:
        existing = read_person_in_image(xmp_path)
        if existing:
            return LabelResult(photo_path=photo_path, status="skipped_existing")

    # Reuse existing face records for this photo; ingest if none
    faces = get_faces_for_photo(str(photo_path), store)
    if not faces:
        try:
            faces = ingestor.ingest_photo(
                image_path=photo_path,
                source_path=str(photo_path),
            )
        except Exception as exc:
            print(f"  Warning: could not ingest {photo_path.name}: {exc}")
            faces = []

    if not faces:
        return LabelResult(photo_path=photo_path, status="skipped_no_faces")

    confirmed_names: list[str] = []
    faces_labeled = 0

    for idx, face in enumerate(faces, 1):
        face_id = str(face.get("face_id", "")).strip()
        crop_rel = face.get("crop_path", "")
        crop_path = (store.root_dir / crop_rel) if crop_rel else None

        existing_person_id = str(face.get("person_id") or "").strip()
        existing_person_name: str | None = None
        if existing_person_id:
            for p in store.list_people():
                if str(p.get("person_id")) == existing_person_id:
                    existing_person_name = str(p.get("display_name", ""))
                    break

        suggestions = get_ai_suggestions(face, store)

        raw = _prompt_for_face(
            photo_path,
            face_index=idx,
            total_faces=len(faces),
            suggestions=suggestions,
            crop_path=crop_path,
            existing_person=existing_person_name,
        )

        if raw is None:
            continue

        # Parse comma-separated names; assign first name to this face record
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            continue

        primary_name = parts[0]
        person = resolve_or_create_person(primary_name, store)
        if face_id:
            store.assign_face(
                face_id,
                str(person.get("person_id")),
                reviewed_by_human=True,
                review_status="confirmed",
            )

        for name in parts:
            if name not in confirmed_names:
                confirmed_names.append(name)

        faces_labeled += 1

    if not confirmed_names:
        return LabelResult(photo_path=photo_path, status="skipped_by_user", faces_labeled=0)

    merge_persons_xmp(xmp_path, confirmed_names)
    print(f"\n  Saved: {', '.join(confirmed_names)} → {xmp_path.name}")

    return LabelResult(
        photo_path=photo_path,
        status="labeled",
        faces_labeled=faces_labeled,
        person_names=confirmed_names,
    )


def run_label_photos(
    directory: Path,
    store: TextFaceStore,
    *,
    ingestor: Any,
    overwrite: bool = False,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> None:
    """
    Main loop: iterate all photos in directory, prompt user to label each face,
    and save results to Cast DB + XMP sidecars.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: directory does not exist: {directory}", file=sys.stderr)
        return

    photos = list(iter_photos(directory, extensions=extensions))
    total = len(photos)
    if total == 0:
        print(f"No photos found in {directory}")
        return

    print(f"Found {total} photo(s) in {directory}")
    print("Commands: enter names (comma-sep) | 's' skip face | 'q' quit\n")

    counts = {
        "labeled": 0,
        "skipped_existing": 0,
        "skipped_no_faces": 0,
        "skipped_by_user": 0,
    }
    total_faces_linked = 0

    for n, photo_path in enumerate(photos, 1):
        print(_DIVIDER)
        print(f"Photo {n}/{total}: {photo_path}")

        result = label_photo(photo_path, store, ingestor, overwrite=overwrite)
        counts[result.status] = counts.get(result.status, 0) + 1
        total_faces_linked += result.faces_labeled

    print(f"\n{_DIVIDER}")
    print("label-photos complete.")
    print(f"  Photos scanned:      {total}")
    print(f"  Already labeled:     {counts['skipped_existing']}  (skipped)")
    print(f"  No faces detected:   {counts['skipped_no_faces']}  (skipped)")
    print(f"  Labeled by user:     {counts['labeled']}")
    print(f"  Skipped by user:     {counts['skipped_by_user']}")
    print(f"  Total faces linked:  {total_faces_linked}")
