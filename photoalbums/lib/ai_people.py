from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


def _import_cast_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from cast.ingest import FaceIngestor, _expand_box, compute_simple_embedding
        from cast.matching import build_person_prototypes, suggest_people_from_prototypes
        from cast.storage import TextFaceStore

        return (
            FaceIngestor,
            _expand_box,
            compute_simple_embedding,
            build_person_prototypes,
            suggest_people_from_prototypes,
            TextFaceStore,
        )
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from cast.ingest import FaceIngestor, _expand_box, compute_simple_embedding
        from cast.matching import build_person_prototypes, suggest_people_from_prototypes
        from cast.storage import TextFaceStore

        return (
            FaceIngestor,
            _expand_box,
            compute_simple_embedding,
            build_person_prototypes,
            suggest_people_from_prototypes,
            TextFaceStore,
        )


@dataclass
class PersonMatch:
    name: str
    score: float


class CastPeopleMatcher:
    def __init__(
        self,
        *,
        cast_store_dir: str | Path,
        min_similarity: float = 0.72,
        min_face_size: int = 40,
        max_faces: int = 40,
        skip_artwork: bool = True,
    ):
        self.min_similarity = float(min_similarity)
        self.min_face_size = int(min_face_size)
        self.max_faces = int(max_faces)
        self.skip_artwork = bool(skip_artwork)

        (
            face_ingestor_cls,
            expand_box_fn,
            embed_fn,
            build_prototypes_fn,
            suggest_fn,
            store_cls,
        ) = _import_cast_modules()

        self._expand_box = expand_box_fn
        self._embed = embed_fn
        self._suggest = suggest_fn

        self._store = store_cls(Path(cast_store_dir))
        self._store.ensure_files()
        self._ingestor = face_ingestor_cls(self._store)

        people = self._store.list_people()
        self._person_name_by_id = {
            str(row.get("person_id")): str(row.get("display_name"))
            for row in people
            if str(row.get("person_id") or "").strip() and str(row.get("display_name") or "").strip()
        }
        self._prototypes = build_prototypes_fn(self._store.list_faces())

    def _detect_faces(self, image_bgr) -> list[tuple[int, int, int, int]]:
        detected = self._ingestor._detect(image_bgr, min_size=self.min_face_size)
        if self.max_faces > 0:
            return detected[: self.max_faces]
        return detected

    def match_image(self, image_path: str | Path) -> list[PersonMatch]:
        if not self._prototypes:
            return []

        import cv2

        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            return []

        h, w = image.shape[:2]
        by_name: dict[str, float] = {}

        for x, y, ww, hh in self._detect_faces(image):
            x0, y0, x1, y1 = self._expand_box(x, y, ww, hh, w, h)
            crop = image[y0:y1, x0:x1]
            if crop is None or crop.size == 0:
                continue
            if not self._ingestor.is_valid_face_crop(crop, skip_artwork=self.skip_artwork):
                continue
            embedding = self._embed(crop)
            suggestions = self._suggest(
                query_embedding=embedding,
                prototypes=self._prototypes,
                top_k=1,
                min_similarity=self.min_similarity,
            )
            if not suggestions:
                continue
            top = suggestions[0]
            person_id = str(top.get("person_id") or "").strip()
            name = self._person_name_by_id.get(person_id, "")
            if not name:
                continue
            score = float(top.get("score") or 0.0)
            current = by_name.get(name)
            if current is None or score > current:
                by_name[name] = score

        out = [PersonMatch(name=name, score=score) for name, score in by_name.items()]
        out.sort(key=lambda row: row.score, reverse=True)
        return out
