from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


def _import_cast_modules() -> tuple[Any, ...]:
    try:
        from cast.ingest import (
            CURRENT_FACE_DETECTOR_MODEL,
            CURRENT_FACE_EMBEDDING_MODEL,
            FALLBACK_FACE_DETECTOR_MODEL,
            FALLBACK_FACE_EMBEDDING_MODEL,
            FaceIngestor,
            _expand_box,
            compute_arcface_embedding,
            compute_simple_embedding,
            estimate_face_quality,
        )
        from cast.matching import (
            build_person_prototypes,
            choose_suggested_candidate,
            normalize_embedding,
            parse_embedding,
            suggest_people_from_prototypes,
        )
        from cast.storage import TextFaceStore, face_is_human_reviewed, face_review_status

        return (
            CURRENT_FACE_DETECTOR_MODEL,
            CURRENT_FACE_EMBEDDING_MODEL,
            FALLBACK_FACE_DETECTOR_MODEL,
            FALLBACK_FACE_EMBEDDING_MODEL,
            FaceIngestor,
            _expand_box,
            compute_arcface_embedding,
            compute_simple_embedding,
            estimate_face_quality,
            build_person_prototypes,
            choose_suggested_candidate,
            normalize_embedding,
            parse_embedding,
            suggest_people_from_prototypes,
            TextFaceStore,
            face_is_human_reviewed,
            face_review_status,
        )
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from cast.ingest import (
            CURRENT_FACE_DETECTOR_MODEL,
            CURRENT_FACE_EMBEDDING_MODEL,
            FALLBACK_FACE_DETECTOR_MODEL,
            FALLBACK_FACE_EMBEDDING_MODEL,
            FaceIngestor,
            _expand_box,
            compute_arcface_embedding,
            compute_simple_embedding,
            estimate_face_quality,
        )
        from cast.matching import (
            build_person_prototypes,
            choose_suggested_candidate,
            normalize_embedding,
            parse_embedding,
            suggest_people_from_prototypes,
        )
        from cast.storage import TextFaceStore, face_is_human_reviewed, face_review_status

        return (
            CURRENT_FACE_DETECTOR_MODEL,
            CURRENT_FACE_EMBEDDING_MODEL,
            FALLBACK_FACE_DETECTOR_MODEL,
            FALLBACK_FACE_EMBEDDING_MODEL,
            FaceIngestor,
            _expand_box,
            compute_arcface_embedding,
            compute_simple_embedding,
            estimate_face_quality,
            build_person_prototypes,
            choose_suggested_candidate,
            normalize_embedding,
            parse_embedding,
            suggest_people_from_prototypes,
            TextFaceStore,
            face_is_human_reviewed,
            face_review_status,
        )


_UNNAMED_PERSON_RE = re.compile(
    r'\b(?:a|an)\s+'
    r'(?:(?:young|elderly|old|middle[- ]aged|tall|short|heavyset|slender|thin|'
    r'unidentified|unknown|male|female|other|second|third|additional|accompanying|'
    r'nearby|standing|seated|smiling|laughing|older|another)\s+)*'
    r'(?:man|woman|boy|girl|person|individual|companion|figure|gentleman|lady|'
    r'teen(?:ager)?|friend|child|adult|bystander|tourist|visitor|guide)\b',
    re.IGNORECASE,
)


def _has_unnamed_person(caption: str) -> bool:
    return bool(_UNNAMED_PERSON_RE.search(str(caption or "")))


def _normalize_hint_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return f" {' '.join(text.split())} ".strip() + " "


def _dedupe_variants(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)


def _box_iou(left: list[int] | tuple[int, int, int, int], right: list[int] | tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = [int(v) for v in left[:4]]
    rx, ry, rw, rh = [int(v) for v in right[:4]]
    if lw <= 0 or lh <= 0 or rw <= 0 or rh <= 0:
        return 0.0
    lx2 = lx + lw
    ly2 = ly + lh
    rx2 = rx + rw
    ry2 = ry + rh
    inter_x1 = max(lx, rx)
    inter_y1 = max(ly, ry)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    union = float((lw * lh) + (rw * rh) - inter)
    if union <= 0.0:
        return 0.0
    return inter / union


def _offset_box(box: tuple[int, int, int, int], offset: tuple[int, int]) -> list[int]:
    x, y, w, h = [int(v) for v in box]
    dx, dy = [int(v) for v in offset]
    return [x + dx, y + dy, w, h]


@dataclass
class PersonMatch:
    name: str
    score: float
    face_id: str = ""
    certainty: float = 0.0
    reviewed_by_human: bool = False
    bbox: list = field(default_factory=list)


class CastPeopleMatcher:
    def __init__(
        self,
        *,
        cast_store_dir: str | Path,
        min_similarity: float = 0.40,
        min_margin: float = 0.06,
        min_face_size: int = 40,
        max_faces: int = 40,
        skip_artwork: bool = True,
        min_face_quality: float = 0.20,
        min_sample_count: int = 2,
        ignore_similarity: float = 0.88,
        review_top_k: int = 5,
    ):
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        self.min_face_size = int(min_face_size)
        self.max_faces = int(max_faces)
        self.skip_artwork = bool(skip_artwork)
        self.min_face_quality = float(min_face_quality)
        self.min_sample_count = int(min_sample_count)
        self.ignore_similarity = float(ignore_similarity)
        self.review_top_k = int(review_top_k)

        (
            current_detector_model,
            current_embedding_model,
            fallback_detector_model,
            fallback_embedding_model,
            face_ingestor_cls,
            expand_box_fn,
            arcface_embed_fn,
            embed_fn,
            estimate_quality_fn,
            build_prototypes_fn,
            choose_candidate_fn,
            normalize_embedding_fn,
            parse_embedding_fn,
            suggest_fn,
            store_cls,
            face_is_human_reviewed_fn,
            face_review_status_fn,
        ) = _import_cast_modules()

        self._current_detector_model = str(current_detector_model)
        self._current_embedding_model = str(current_embedding_model)
        self._fallback_detector_model = str(fallback_detector_model)
        self._fallback_embedding_model = str(fallback_embedding_model)
        self._active_embedding_models = {self._current_embedding_model}
        self._expand_box = expand_box_fn
        self._arcface_embed = arcface_embed_fn
        self._embed = embed_fn
        self._estimate_quality = estimate_quality_fn
        self._build_prototypes = build_prototypes_fn
        self._choose_candidate = choose_candidate_fn
        self._normalize_embedding = normalize_embedding_fn
        self._parse_embedding = parse_embedding_fn
        self._suggest = suggest_fn
        self._face_is_human_reviewed = face_is_human_reviewed_fn
        self._face_review_status = face_review_status_fn

        self._store = store_cls(Path(cast_store_dir))
        self._store.ensure_files()
        self._ingestor = face_ingestor_cls(self._store)

        self._store_signature = ""
        self._person_name_by_id: dict[str, str] = {}
        self._person_variants_by_id: dict[str, tuple[str, ...]] = {}
        self._faces_by_source: dict[str, list[dict[str, Any]]] = {}
        self._ignored_embeddings: list[np.ndarray] = []
        self._prototypes: dict[str, dict[str, Any]] = {}
        self._reload_state()

    def _face_embedding_model(self, face: dict[str, Any]) -> str:
        metadata = face.get("metadata")
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("embedding_model") or "").strip()

    def _face_uses_active_model(self, face: dict[str, Any]) -> bool:
        return self._face_embedding_model(face) in self._active_embedding_models

    def _model_metadata(self, *, source_path: str, analysis_image_path: str, arcface_embedding: Any) -> dict[str, Any]:
        metadata = {
            "ingest": "photoalbums_ai",
            "detector_model": (
                self._current_detector_model
                if getattr(self._ingestor, "_insightface", None) is not None
                else self._fallback_detector_model
            ),
            "embedding_model": (
                self._current_embedding_model
                if arcface_embedding is not None
                else self._fallback_embedding_model
            ),
        }
        if str(analysis_image_path).strip() and str(analysis_image_path).strip() != str(source_path).strip():
            metadata["analysis_image_path"] = str(analysis_image_path)
        return metadata

    def _reload_state(self) -> None:
        faces = self._store.list_faces()
        people = self._store.list_people()
        self._store_signature = self._store.store_signature()
        self._person_name_by_id = {
            str(row.get("person_id")): str(row.get("display_name"))
            for row in people
            if str(row.get("person_id") or "").strip() and str(row.get("display_name") or "").strip()
        }
        self._person_variants_by_id = {
            str(row.get("person_id")): _dedupe_variants(
                [str(row.get("display_name") or "")] + [str(item or "") for item in list(row.get("aliases") or [])]
            )
            for row in people
            if str(row.get("person_id") or "").strip()
        }
        faces_by_source: dict[str, list[dict[str, Any]]] = {}
        ignored_embeddings: list[np.ndarray] = []
        known_faces: list[dict[str, Any]] = []
        for face in faces:
            source_key = str(face.get("source_path") or "").strip()
            if source_key:
                faces_by_source.setdefault(source_key, []).append(face)
            status = str(self._face_review_status(face) or "").strip().lower()
            if status in {"ignored", "rejected"} and self._face_uses_active_model(face):
                emb = self._normalize_embedding_safe(face.get("embedding"))
                if emb is not None:
                    ignored_embeddings.append(emb)
            person_id = str(face.get("person_id") or "").strip()
            if not person_id:
                continue
            if status in {"ignored", "rejected"}:
                continue
            if not self._face_uses_active_model(face):
                continue
            known_faces.append(face)
        self._faces_by_source = faces_by_source
        self._ignored_embeddings = ignored_embeddings
        self._prototypes = self._build_prototypes(
            known_faces,
            allowed_embedding_model_ids=self._active_embedding_models,
        )

    def _maybe_refresh(self) -> None:
        current = self._store.store_signature()
        if current != self._store_signature:
            self._reload_state()

    def store_signature(self) -> str:
        return self._store.store_signature()

    def _normalize_embedding_safe(self, raw_embedding: Any) -> np.ndarray | None:
        try:
            return self._normalize_embedding(self._parse_embedding(raw_embedding))
        except Exception:
            return None

    def _detect_faces(self, image_bgr) -> list[tuple[int, int, int, int]]:
        detected = self._ingestor._detect(image_bgr, min_size=self.min_face_size)
        if self.max_faces > 0:
            return detected[: self.max_faces]
        return detected

    def _remember_face(self, face: dict[str, Any]) -> dict[str, Any]:
        source_key = str(face.get("source_path") or "").strip()
        face_id = str(face.get("face_id") or "").strip()
        if source_key and face_id:
            bucket = list(self._faces_by_source.get(source_key, []))
            replaced = False
            for index, row in enumerate(bucket):
                if str(row.get("face_id") or "").strip() != face_id:
                    continue
                bucket[index] = dict(face)
                replaced = True
                break
            if not replaced:
                bucket.append(dict(face))
            self._faces_by_source[source_key] = bucket
        self._store_signature = self._store.store_signature()
        return dict(face)

    def _find_existing_face(self, *, source_path: str, bbox: list[int]) -> dict[str, Any] | None:
        best_face: dict[str, Any] | None = None
        best_iou = 0.0
        for face in self._faces_by_source.get(str(source_path or "").strip(), []):
            face_bbox = list(face.get("bbox") or [])
            if len(face_bbox) < 4:
                continue
            overlap = _box_iou(face_bbox, bbox)
            if overlap <= best_iou:
                continue
            best_face = face
            best_iou = overlap
        if best_face is None or best_iou < 0.55:
            return None
        return dict(best_face)

    def _ensure_crop(self, face: dict[str, Any], crop_bgr) -> dict[str, Any]:
        crop_rel = str(face.get("crop_path") or "").strip()
        if crop_rel:
            crop_abs = (self._store.root_dir / crop_rel).resolve()
            if crop_abs.exists():
                return dict(face)
        face_id = str(face.get("face_id") or "").strip()
        if not face_id:
            return dict(face)
        crop_rel = self._ingestor._save_crop(face_id, crop_bgr)
        updated = self._store.update_face(face_id, crop_path=crop_rel)
        return self._remember_face(updated)

    def _create_face_record(
        self,
        *,
        crop_bgr,
        source_path: str,
        bbox: list[int],
        analysis_image_path: str,
    ) -> dict[str, Any]:
        arcface_embedding = self._arcface_embed(crop_bgr)
        embedding = arcface_embedding or self._embed(crop_bgr)
        quality = self._estimate_quality(crop_bgr)
        metadata = self._model_metadata(
            source_path=source_path,
            analysis_image_path=analysis_image_path,
            arcface_embedding=arcface_embedding,
        )
        face = self._store.add_face(
            embedding=embedding,
            source_type="photo",
            source_path=str(source_path),
            timestamp="",
            bbox=[int(v) for v in bbox[:4]],
            quality=quality,
            metadata=metadata,
        )
        crop_rel = self._ingestor._save_crop(str(face.get("face_id")), crop_bgr)
        face = self._store.update_face(str(face.get("face_id")), crop_path=crop_rel)
        return self._remember_face(face)

    def _refresh_face_record(
        self,
        face: dict[str, Any],
        *,
        crop_bgr,
        bbox: list[int],
        analysis_image_path: str,
    ) -> dict[str, Any]:
        face_id = str(face.get("face_id") or "").strip()
        if not face_id:
            return dict(face)
        source_path = str(face.get("source_path") or "").strip()
        arcface_embedding = self._arcface_embed(crop_bgr)
        embedding = arcface_embedding or self._embed(crop_bgr)
        quality = self._estimate_quality(crop_bgr)
        metadata = dict(face.get("metadata") or {})
        metadata.update(
            self._model_metadata(
                source_path=source_path,
                analysis_image_path=analysis_image_path,
                arcface_embedding=arcface_embedding,
            )
        )
        updated = self._store.update_face(
            face_id,
            bbox=[int(v) for v in bbox[:4]],
            embedding=embedding,
            quality=quality,
            metadata=metadata,
        )
        updated = self._ensure_crop(updated, crop_bgr)
        # Legacy reviewed/ignored rows need to re-enter the active prototype cache immediately.
        self._reload_state()
        return updated

    def _hint_bonus_by_person_id(self, hint_text: str) -> dict[str, float]:
        normalized = _normalize_hint_text(hint_text)
        if not normalized.strip():
            return {}
        bonuses: dict[str, float] = {}
        for person_id, variants in self._person_variants_by_id.items():
            best = 0.0
            for variant in variants:
                phrase = _normalize_hint_text(variant).strip()
                if not phrase:
                    continue
                tokens = [token for token in phrase.split() if len(token) >= 4]
                phrase_match = f" {phrase} " in normalized
                if phrase_match:
                    best = max(best, 0.08 if len(tokens) >= 2 else 0.05)
                    continue
                token_hits = [token for token in tokens if f" {token} " in normalized]
                if len(token_hits) >= 2:
                    best = max(best, 0.05)
                elif len(token_hits) == 1 and len(token_hits[0]) >= 5:
                    best = max(best, 0.015)
            if best > 0.0:
                bonuses[person_id] = best
        return bonuses

    def _apply_hint_scores(self, candidates: list[dict[str, Any]], hint_text: str) -> list[dict[str, Any]]:
        if not candidates:
            return []
        bonuses = self._hint_bonus_by_person_id(hint_text)
        strong_hint_ids = {person_id for person_id, bonus in bonuses.items() if bonus >= 0.05}
        adjusted: list[dict[str, Any]] = []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            person_id = str(row.get("person_id") or "").strip()
            if not person_id:
                continue
            bonus = float(bonuses.get(person_id, 0.0))
            if len(strong_hint_ids) == 1 and person_id in strong_hint_ids:
                bonus += 0.02
            adjusted.append(
                {
                    **row,
                    "score": float(row.get("score") or 0.0) + bonus,
                    "hint_bonus": bonus,
                }
            )
        adjusted.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return adjusted

    def _matches_ignored_face(self, embedding: Any) -> bool:
        if not self._ignored_embeddings:
            return False
        query = self._normalize_embedding_safe(embedding)
        if query is None:
            return False
        best = 0.0
        for ignored in self._ignored_embeddings:
            if not isinstance(ignored, np.ndarray) or ignored.shape != query.shape:
                continue
            best = max(best, float(np.dot(query, ignored)))
            if best >= self.ignore_similarity:
                return True
        return False

    def _queue_for_review(self, face: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
        face_id = str(face.get("face_id") or "").strip()
        if not face_id:
            return
        quality = face.get("quality")
        if quality is not None and float(quality) < self.min_face_quality:
            return
        for review in self._store.list_review_items():
            if str(review.get("face_id") or "").strip() != face_id:
                continue
            if str(review.get("status") or "").strip().lower() == "pending":
                return
        self._store.add_review_item(
            face_id=face_id,
            candidates=[dict(row) for row in candidates if isinstance(row, dict)],
            suggested_person_id=None,
            suggested_score=None,
            status="pending",
        )
        self._store_signature = self._store.store_signature()

    def match_image(
        self,
        image_path: str | Path,
        *,
        source_path: str | Path | None = None,
        bbox_offset: tuple[int, int] = (0, 0),
        hint_text: str = "",
    ) -> list[PersonMatch]:
        self._maybe_refresh()

        import cv2

        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            return []
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        source_key = str(source_path or path)
        height, width = image.shape[:2]
        by_name: dict[str, PersonMatch] = {}

        for x, y, ww, hh in self._detect_faces(image):
            x0, y0, x1, y1 = self._expand_box(x, y, ww, hh, width, height)
            crop = image[y0:y1, x0:x1]
            if crop is None or crop.size == 0:
                continue
            if not self._ingestor.is_valid_face_crop(crop, skip_artwork=self.skip_artwork):
                continue

            absolute_bbox = _offset_box((x, y, ww, hh), bbox_offset)
            face = self._find_existing_face(source_path=source_key, bbox=absolute_bbox)
            if face is None:
                face = self._create_face_record(
                    crop_bgr=crop,
                    source_path=source_key,
                    bbox=absolute_bbox,
                    analysis_image_path=str(path),
                )
            else:
                if self._face_uses_active_model(face):
                    face = self._ensure_crop(face, crop)
                else:
                    face = self._refresh_face_record(
                        face,
                        crop_bgr=crop,
                        bbox=absolute_bbox,
                        analysis_image_path=str(path),
                    )

            review_status = str(self._face_review_status(face) or "").strip().lower()
            if review_status in {"ignored", "rejected"}:
                continue

            person_id = str(face.get("person_id") or "").strip()
            if person_id and self._face_is_human_reviewed(face) and review_status == "confirmed":
                name = self._person_name_by_id.get(person_id, "")
                if name:
                    current = by_name.get(name)
                    candidate = PersonMatch(
                        name=name,
                        score=1.0,
                        face_id=str(face.get("face_id") or ""),
                        certainty=1.0,
                        reviewed_by_human=True,
                        bbox=list(face.get("bbox") or []),
                    )
                    if current is None or candidate.certainty > current.certainty or candidate.score > current.score:
                        by_name[name] = candidate
                continue

            embedding = face.get("embedding") or self._arcface_embed(crop) or self._embed(crop)
            if self._matches_ignored_face(embedding):
                continue

            raw_candidates: list[dict[str, Any]] = []
            if self._prototypes:
                try:
                    raw_candidates = self._suggest(
                        query_embedding=embedding,
                        prototypes=self._prototypes,
                        top_k=max(3, self.review_top_k),
                        min_similarity=-1.0,
                    )
                except Exception:
                    raw_candidates = []
            candidates = self._apply_hint_scores(raw_candidates, hint_text)
            suggested, _margin = self._choose_candidate(
                candidates=candidates,
                face_quality=face.get("quality"),
                min_similarity=self.min_similarity,
                min_margin=self.min_margin,
                min_face_quality=self.min_face_quality,
                min_sample_count=self.min_sample_count,
            )
            if suggested:
                suggested_person_id = str(suggested.get("person_id") or "").strip()
                name = self._person_name_by_id.get(suggested_person_id, "")
                if name:
                    score = float(suggested.get("score") or 0.0)
                    current = by_name.get(name)
                    candidate = PersonMatch(
                        name=name,
                        score=score,
                        face_id=str(face.get("face_id") or ""),
                        certainty=min(0.99, max(0.0, score)),
                        reviewed_by_human=False,
                        bbox=list(face.get("bbox") or []),
                    )
                    if (
                        current is None
                        or candidate.certainty > current.certainty
                        or (candidate.certainty == current.certainty and candidate.score > current.score)
                    ):
                        by_name[name] = candidate
                    continue

            self._queue_for_review(face, candidates)

        out = list(by_name.values())
        out.sort(key=lambda row: (-float(row.certainty), -float(row.score), row.name.casefold()))
        return out

    def queue_caption_unknown(
        self,
        *,
        source_path: str | Path,
        caption: str,
    ) -> bool:
        """
        If the caption mentions an unnamed person, create a placeholder face record and
        queue it for Cast review so the user can provide the name.

        Returns True if a new review was queued, False if skipped (already pending or no match).
        """
        if not _has_unnamed_person(caption):
            return False
        source_str = str(source_path or "").strip()
        if not source_str:
            return False
        self._maybe_refresh()
        # Skip if there is already a pending caption_unknown review for this source.
        for face in self._store.list_faces():
            meta = face.get("metadata") or {}
            if str(meta.get("ingest") or "") != "caption_unknown":
                continue
            if str(face.get("source_path") or "").strip() != source_str:
                continue
            face_id = str(face.get("face_id") or "").strip()
            for review in self._store.list_review_items():
                if str(review.get("face_id") or "").strip() == face_id:
                    if str(review.get("status") or "").strip().lower() == "pending":
                        return False
        face = self._store.add_face(
            embedding=[],
            source_type="photo",
            source_path=source_str,
            timestamp="",
            bbox=[],
            quality=None,
            metadata={
                "ingest": "caption_unknown",
                "caption": str(caption or "")[:500],
            },
        )
        self._queue_for_review(face, [])
        self._store_signature = self._store.store_signature()
        return True
