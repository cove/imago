from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from .xmp_sidecar import _dedupe
from .ai_people_preprocess import build_rembg_bgr


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
        from cast.storage import (
            TextFaceStore,
            face_is_human_reviewed,
            face_review_status,
        )

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
        from cast.storage import (
            TextFaceStore,
            face_is_human_reviewed,
            face_review_status,
        )

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


def _normalize_hint_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return f" {' '.join(text.split())} ".strip() + " "


def _dedupe_variants(values: list[str]) -> tuple[str, ...]:
    return tuple(_dedupe(values))


def _box_iou(
    left: list[int] | tuple[int, int, int, int],
    right: list[int] | tuple[int, int, int, int],
) -> float:
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
        min_face_quality: float = 0.20,
        min_sample_count: int = 2,
        ignore_similarity: float = 0.88,
        review_top_k: int = 5,
    ):
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        self.min_face_size = int(min_face_size)
        self.max_faces = int(max_faces)
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
        self._face_ingestor_cls = face_ingestor_cls
        self._ingestor = None
        self._last_detector_model = self._fallback_detector_model

        self._store_signature = ""
        self.last_faces_detected: int = 0
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

    def _get_ingestor(self):
        if self._ingestor is None:
            self._ingestor = self._face_ingestor_cls(self._store)
        return self._ingestor

    def _model_metadata(
        self,
        *,
        source_path: str,
        analysis_image_path: str,
        arcface_embedding: Any,
        analysis_variant: str,
    ) -> dict[str, Any]:
        metadata = {
            "ingest": "photoalbums_ai",
            "detector_model": self._last_detector_model,
            "embedding_model": (
                self._current_embedding_model if arcface_embedding is not None else self._fallback_embedding_model
            ),
            "analysis_variant": str(analysis_variant or "original").strip() or "original",
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
        ingestor = self._get_ingestor()
        self._last_detector_model = (
            self._current_detector_model
            if getattr(ingestor, "_insightface", None) is not None
            else self._fallback_detector_model
        )
        detected = ingestor._detect(image_bgr, min_size=self.min_face_size)
        if self.max_faces > 0:
            return detected[: self.max_faces]
        return detected

    def _is_valid_face_crop(self, crop_bgr) -> bool:
        return bool(self._get_ingestor().is_valid_face_crop(crop_bgr))

    def _save_crop(self, face_id: str, crop_bgr) -> str:
        import cv2

        crops_dir = self._store.root_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"{str(face_id).strip()}.jpg"
        crop_path = crops_dir / crop_name
        ok = cv2.imwrite(str(crop_path), crop_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write crop image: {crop_path}")
        return crop_path.relative_to(self._store.root_dir).as_posix()

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

    def _find_existing_face(self, *, source_path: str, bbox: list[int], min_iou: float = 0.55) -> dict[str, Any] | None:
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
        if best_face is None or best_iou < float(min_iou):
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
        crop_rel = self._save_crop(face_id, crop_bgr)
        updated = self._store.update_face(face_id, crop_path=crop_rel)
        return self._remember_face(updated)

    def _create_face_record(
        self,
        *,
        crop_bgr,
        source_path: str,
        bbox: list[int],
        analysis_image_path: str,
        analysis_variant: str,
    ) -> dict[str, Any]:
        arcface_embedding = self._arcface_embed(crop_bgr)
        embedding = arcface_embedding or self._embed(crop_bgr)
        quality = self._estimate_quality(crop_bgr)
        metadata = self._model_metadata(
            source_path=source_path,
            analysis_image_path=analysis_image_path,
            arcface_embedding=arcface_embedding,
            analysis_variant=analysis_variant,
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
        crop_rel = self._save_crop(str(face.get("face_id")), crop_bgr)
        face = self._store.update_face(str(face.get("face_id")), crop_path=crop_rel)
        return self._remember_face(face)

    def _refresh_face_record(
        self,
        face: dict[str, Any],
        *,
        crop_bgr,
        bbox: list[int],
        analysis_image_path: str,
        analysis_variant: str,
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
                analysis_variant=analysis_variant,
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

    def _find_person_id_by_name(self, name: str) -> str | None:
        """Return the person_id if `name` matches any Cast person's display name or alias."""
        target = _normalize_hint_text(name).strip()
        if not target or len(target) < 2:
            return None
        for person_id, variants in self._person_variants_by_id.items():
            for variant in variants:
                phrase = _normalize_hint_text(variant).strip()
                if phrase and phrase == target:
                    return person_id
        return None

    def _build_face_name_hints(
        self,
        hint_text: str,
        source_path: str,
        face_index: int,
        total_faces: int,
    ) -> list[dict[str, Any]]:
        """Build name hints for a specific face from OCR/hint text and filename.

        Hyphen-separated sequences (e.g. "Karl-Billy-Leslie") are assigned
        positionally when the count matches total_faces; otherwise all names
        are shown as non-positional hints. Individual Cast person names found
        anywhere in hint_text are also included.
        """
        hints: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        def _add(name: str, person_id: str | None, confidence: float, source: str) -> None:
            key = name.casefold()
            if key in seen_names:
                return
            seen_names.add(key)
            hints.append(
                {
                    "name": name,
                    "person_id": person_id,
                    "confidence": confidence,
                    "source": source,
                }
            )

        # 1. Hyphen-separated sequences in hint_text
        for seq_match in re.finditer(r"\b([a-zA-Z]{2,}(?:-[a-zA-Z]{2,})+)\b", hint_text):
            parts = [p.strip() for p in seq_match.group(0).split("-") if len(p.strip()) >= 2]
            if len(parts) < 2:
                continue
            if len(parts) == total_faces:
                # Positional: this face gets the name at its left-to-right index
                name = parts[face_index].title()
                person_id = self._find_person_id_by_name(name)
                _add(name, person_id, 0.85 if person_id else 0.65, "positional_caption")
            else:
                for name_raw in parts:
                    name = name_raw.title()
                    person_id = self._find_person_id_by_name(name)
                    _add(name, person_id, 0.65 if person_id else 0.45, "caption")

        # 2. Individual Cast person names found in hint_text
        normalized = _normalize_hint_text(hint_text)
        for person_id, variants in self._person_variants_by_id.items():
            for variant in variants:
                phrase = _normalize_hint_text(variant).strip()
                if not phrase:
                    continue
                if f" {phrase} " in normalized:
                    name = self._person_name_by_id.get(person_id, variant)
                    _add(name, person_id, 0.5, "caption")
                    break

        # 3. Hyphen-separated sequences in the source filename stem
        stem = Path(source_path).stem
        stem_parts = [p.strip() for p in stem.split("-") if re.match(r"^[a-zA-Z]{2,}$", p.strip())]
        if len(stem_parts) >= 2:
            if len(stem_parts) == total_faces:
                name = stem_parts[face_index].title()
                person_id = self._find_person_id_by_name(name)
                _add(name, person_id, 0.85 if person_id else 0.65, "positional_filename")
            else:
                for name_raw in stem_parts:
                    name = name_raw.title()
                    person_id = self._find_person_id_by_name(name)
                    _add(name, person_id, 0.65 if person_id else 0.45, "filename")

        return hints

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

    def _queue_for_review(
        self,
        face: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        name_hints: list[dict[str, Any]] | None = None,
    ) -> None:
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
            name_hints=name_hints or [],
        )
        self._store_signature = self._store.store_signature()

    def _analysis_image(self, image_bgr, analysis_variant: str):
        variant = str(analysis_variant or "original").strip().lower() or "original"
        if variant == "original":
            return image_bgr
        if variant == "rembg":
            return build_rembg_bgr(image_bgr)
        raise ValueError(f"Unsupported analysis variant: {analysis_variant}")

    def match_image(
        self,
        image_path: str | Path,
        *,
        source_path: str | Path | None = None,
        bbox_offset: tuple[int, int] = (0, 0),
        hint_text: str = "",
        analysis_variant: str = "original",
        match_iou: float = 0.55,
        refresh_active_face: bool = False,
    ) -> list[PersonMatch]:
        self._maybe_refresh()

        import cv2

        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            return []
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = self._analysis_image(image, analysis_variant)

        source_key = str(source_path or path)
        height, width = image.shape[:2]
        by_name: dict[str, PersonMatch] = {}
        normalized_variant = str(analysis_variant or "original").strip().lower() or "original"

        # Collect valid faces sorted left-to-right for positional name hint assignment.
        all_detected = self._detect_faces(image)
        all_detected_sorted = sorted(all_detected, key=lambda f: f[0])
        valid_faces: list[tuple[int, int, int, int]] = []
        for x, y, ww, hh in all_detected_sorted:
            x0, y0, x1, y1 = self._expand_box(x, y, ww, hh, width, height)
            crop = image[y0:y1, x0:x1]
            if crop is None or crop.size == 0:
                continue
            if not self._is_valid_face_crop(crop):
                continue
            valid_faces.append((x, y, ww, hh))
        total_valid = len(valid_faces)
        self.last_faces_detected = total_valid

        for face_rank, (x, y, ww, hh) in enumerate(valid_faces):
            x0, y0, x1, y1 = self._expand_box(x, y, ww, hh, width, height)
            crop = image[y0:y1, x0:x1]

            absolute_bbox = _offset_box((x, y, ww, hh), bbox_offset)
            face = self._find_existing_face(
                source_path=source_key,
                bbox=absolute_bbox,
                min_iou=float(match_iou),
            )
            if face is None:
                face = self._create_face_record(
                    crop_bgr=crop,
                    source_path=source_key,
                    bbox=absolute_bbox,
                    analysis_image_path=str(path),
                    analysis_variant=normalized_variant,
                )
            else:
                review_status = str(self._face_review_status(face) or "").strip().lower()
                person_id = str(face.get("person_id") or "").strip()
                face_is_locked = review_status == "confirmed" and bool(person_id) and self._face_is_human_reviewed(face)
                should_refresh_active = (
                    bool(refresh_active_face) and review_status not in {"ignored", "rejected"} and not face_is_locked
                )
                if self._face_uses_active_model(face) and not should_refresh_active:
                    face = self._ensure_crop(face, crop)
                else:
                    face = self._refresh_face_record(
                        face,
                        crop_bgr=crop,
                        bbox=absolute_bbox,
                        analysis_image_path=str(path),
                        analysis_variant=normalized_variant,
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

            name_hints = self._build_face_name_hints(
                hint_text=hint_text,
                source_path=source_key,
                face_index=face_rank,
                total_faces=total_valid,
            )
            self._queue_for_review(face, candidates, name_hints=name_hints)

        out = list(by_name.values())
        out.sort(
            key=lambda row: (
                -float(row.certainty),
                -float(row.score),
                row.name.casefold(),
            )
        )
        return out

    def match_image_recovery(
        self,
        image_path: str | Path,
        *,
        source_path: str | Path | None = None,
        bbox_offset: tuple[int, int] = (0, 0),
        hint_text: str = "",
    ) -> list[PersonMatch]:
        return self.match_image(
            image_path,
            source_path=source_path,
            bbox_offset=bbox_offset,
            hint_text=hint_text,
            analysis_variant="rembg",
            match_iou=0.30,
            refresh_active_face=True,
        )
