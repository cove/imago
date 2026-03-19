from __future__ import annotations

import json
import sys
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2

from .ingest import CURRENT_FACE_EMBEDDING_MODEL, FaceIngestor
from .matching import (
    _coerce_float,
    _coerce_int,
    build_person_prototypes,
    choose_suggested_candidate,
    face_embedding_model,
    parse_embedding,
    suggest_people,
    suggest_people_from_prototypes,
)
from .storage import TextFaceStore, face_is_human_reviewed, face_review_status
from .xmp_writer import merge_persons_xmp, read_person_in_image, read_xmp_description

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_INDEX = _STATIC / "index.html"
DEFAULT_PHOTO_ALBUMS_ROOT = (
    "C:/Users/covec/OneDrive/Cordell, Leslie & Audrey/Photo Albums"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_LMSTUDIO_URL = "http://192.168.4.72:1234/v1"
DEFAULT_MIN_SIMILARITY = 0.72
DEFAULT_MIN_MARGIN = 0.015
DEFAULT_MIN_FACE_QUALITY = 0.20
DEFAULT_MIN_SAMPLE_COUNT = 2
ACTIVE_EMBEDDING_MODELS = {CURRENT_FACE_EMBEDDING_MODEL}


def _rewrite_description_via_lmstudio(
    description: str,
    person_names: list[str],
    *,
    base_url: str,
) -> str | None:
    """Best-effort: ask LM Studio to substitute generic references with actual names.

    Returns the updated description string, or None if the call fails or is skipped.
    """
    if not description or not person_names:
        return None
    names_str = ", ".join(person_names)
    prompt = (
        "You are editing a photo description. "
        "Some people in the photo are now identified by name. "
        "Where the description uses a generic or collective reference to a person "
        "(any phrase that does not use a name, regardless of how many people it refers to), "
        "replace it with the actual name if you can confidently determine which person is meant. "
        "If there are more people in the description than known names, substitute only what "
        "is unambiguous and leave the rest as a generic reference. "
        "Do not add, remove, or invent any detail beyond substituting names. "
        "Return only the updated description with no explanation.\n\n"
        f"Known people in this photo: {names_str}\n\n"
        f"Description: {description}"
    )
    payload = json.dumps(
        {
            "model": "loaded-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = str(
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
                or ""
            ).strip()
            if text:
                return text
    except Exception as exc:
        print(f"[cast] description rewrite skipped: {exc}", file=sys.stderr)
    return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


class CastHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        store: TextFaceStore,
        lmstudio_url: str = DEFAULT_LMSTUDIO_URL,
    ):
        self.store = store
        self.ingestor = FaceIngestor(store, require_primary_model=True)
        self.lmstudio_url = str(lmstudio_url or DEFAULT_LMSTUDIO_URL).strip()
        super().__init__((host, int(port)), CastHandler)


class CastHandler(BaseHTTPRequestHandler):
    server: CastHTTPServer  # type: ignore[assignment]

    def log_message(self, _format: str, *args: Any) -> None:
        return

    @property
    def store(self) -> TextFaceStore:
        return self.server.store

    @property
    def lmstudio_url(self) -> str:
        return self.server.lmstudio_url

    def _read_json(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            content_length = 0
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, text: str, status: int = 200) -> None:
        data = text.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", str(content_type))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self) -> None:
        self._send_json(
            {"ok": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND
        )

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": str(message)}, status=int(status))

    def _face_detector_model(self, face: dict[str, Any]) -> str:
        if not isinstance(face, dict):
            return ""
        metadata = face.get("metadata")
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("detector_model") or "").strip()

    def _face_summary(
        self, face: dict[str, Any], people_by_id: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        person_id = str(face.get("person_id") or "").strip() or None
        person = people_by_id.get(person_id or "")
        embedding = face.get("embedding")
        dim = len(embedding) if isinstance(embedding, list) else 0
        face_id = str(face.get("face_id", ""))
        source_path = str(face.get("source_path", ""))
        source_kind = self._face_source_kind(face)
        crop_rel = str(face.get("crop_path") or "").strip()
        review_status = face_review_status(face)
        return {
            "face_id": face_id,
            "person_id": person_id,
            "person_name": (
                str(person.get("display_name")) if isinstance(person, dict) else ""
            ),
            "source_type": str(face.get("source_type", "")),
            "source_path": source_path,
            "timestamp": str(face.get("timestamp", "")),
            "quality": face.get("quality"),
            "embedding_dim": int(dim),
            "crop_path": crop_rel,
            "crop_url": f"/api/faces/{face_id}/crop" if crop_rel else "",
            "source_url": f"/api/faces/{face_id}/source" if source_path else "",
            "source_is_image": source_kind == "image",
            "reviewed_by_human": bool(face_is_human_reviewed(face)),
            "review_status": review_status,
            "created_at": str(face.get("created_at", "")),
            "updated_at": str(face.get("updated_at", "")),
        }

    def _review_summary(
        self,
        review: dict[str, Any],
        *,
        people_by_id: dict[str, dict[str, Any]],
        faces_by_id: dict[str, dict[str, Any]],
        prototypes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        face_id = str(review.get("face_id", ""))
        face = faces_by_id.get(face_id, {})
        status = str(review.get("status", ""))
        source_candidates = list(review.get("candidates") or [])
        suggested_person_id = (
            str(review.get("suggested_person_id") or "").strip() or None
        )
        suggested_score = review.get("suggested_score")
        suggested_margin = None
        suggested_confident = bool(suggested_person_id)

        if (
            status == "pending"
            and face
            and face_embedding_model(face) not in ACTIVE_EMBEDDING_MODELS
        ):
            source_candidates = []
        elif status == "pending" and face and prototypes and not source_candidates:
            top_k = max(3, len(source_candidates))
            try:
                live_candidates = suggest_people_from_prototypes(
                    query_embedding=face.get("embedding") or [],
                    prototypes=prototypes,
                    top_k=top_k,
                    min_similarity=-1.0,
                )
            except Exception:
                live_candidates = []
            if live_candidates:
                source_candidates = live_candidates
        if status == "pending":
            suggested, margin = choose_suggested_candidate(
                candidates=source_candidates,
                face_quality=face.get("quality") if isinstance(face, dict) else None,
                min_similarity=DEFAULT_MIN_SIMILARITY,
                min_margin=DEFAULT_MIN_MARGIN,
                min_face_quality=DEFAULT_MIN_FACE_QUALITY,
                min_sample_count=DEFAULT_MIN_SAMPLE_COUNT,
            )
            suggested_confident = suggested is not None
            suggested_margin = float(margin)
            if suggested is None:
                suggested_person_id = None
                suggested_score = None
            else:
                suggested_person_id = (
                    str(suggested.get("person_id") or "").strip() or None
                )
                _score = suggested.get("score")
                suggested_score = float(_score) if _score is not None else None

        candidates = []
        for row in source_candidates:
            if not isinstance(row, dict):
                continue
            person_id = str(row.get("person_id") or "").strip()
            person = people_by_id.get(person_id, {})
            candidates.append(
                {
                    "person_id": person_id,
                    "person_name": str(person.get("display_name", "")),
                    "score": float(row.get("score", 0.0)),
                    "sample_count": int(row.get("sample_count", 0)),
                }
            )
        decided_person_id = str(review.get("decided_person_id") or "").strip() or None
        decided_person = people_by_id.get(decided_person_id or "")
        suggested_person = people_by_id.get(suggested_person_id or "")
        name_hints = [
            dict(h) for h in (review.get("name_hints") or []) if isinstance(h, dict)
        ]
        return {
            "review_id": str(review.get("review_id", "")),
            "face_id": face_id,
            "status": status,
            "suggested_person_id": suggested_person_id,
            "suggested_person_name": (
                str(suggested_person.get("display_name", ""))
                if suggested_person
                else ""
            ),
            "suggested_score": suggested_score,
            "suggested_margin": suggested_margin,
            "suggested_confident": bool(suggested_confident),
            "decided_person_id": decided_person_id,
            "decided_person_name": (
                str(decided_person.get("display_name", "")) if decided_person else ""
            ),
            "candidates": candidates,
            "name_hints": name_hints,
            "created_at": str(review.get("created_at", "")),
            "updated_at": str(review.get("updated_at", "")),
            "face": self._face_summary(face, people_by_id) if face else None,
        }

    def _state_payload(self) -> dict[str, Any]:
        people = self.store.list_people()
        faces = self.store.list_faces()
        reviews = self.store.list_review_items()
        people_by_id = {str(row.get("person_id")): row for row in people}
        faces_by_id = {str(row.get("face_id")): row for row in faces}
        prototypes = build_person_prototypes(
            faces,
            allowed_embedding_model_ids=ACTIVE_EMBEDDING_MODELS,
        )
        legacy_faces = 0
        for face in faces:
            if (
                not self._face_detector_model(face)
                or face_embedding_model(face) not in ACTIVE_EMBEDDING_MODELS
            ):
                legacy_faces += 1

        face_summaries = [self._face_summary(face, people_by_id) for face in faces]
        review_summaries = [
            self._review_summary(
                item,
                people_by_id=people_by_id,
                faces_by_id=faces_by_id,
                prototypes=prototypes,
            )
            for item in reviews
        ]
        unknown_faces = [
            row
            for row in face_summaries
            if (not row.get("person_id"))
            and str(row.get("review_status") or "").strip().lower()
            not in {"ignored", "rejected"}
        ]
        pending_reviews = [
            row for row in review_summaries if row.get("status") == "pending"
        ]
        return {
            "ok": True,
            "counts": {
                "people": len(people),
                "faces": len(face_summaries),
                "unknown_faces": len(unknown_faces),
                "pending_reviews": len(pending_reviews),
                "legacy_faces": int(legacy_faces),
            },
            "people": people,
            "faces": face_summaries,
            "unknown_faces": unknown_faces,
            "reviews": review_summaries,
            "pending_reviews": pending_reviews,
            "runtime": self.server.ingestor.runtime_status(),
        }

    def _handle_get_state(self) -> None:
        self._send_json(self._state_payload())

    def _handle_get_people(self) -> None:
        people = self.store.list_people()
        self._send_json({"ok": True, "people": people})

    def _handle_get_faces(self, query: dict[str, list[str]]) -> None:
        include_embedding = (
            str((query.get("include_embedding") or ["0"])[0]).strip() == "1"
        )
        people = self.store.list_people()
        people_by_id = {str(row.get("person_id")): row for row in people}
        faces = self.store.list_faces()
        rows = []
        for face in faces:
            entry = self._face_summary(face, people_by_id)
            if include_embedding:
                entry["embedding"] = list(face.get("embedding") or [])
            rows.append(entry)
        self._send_json({"ok": True, "faces": rows})

    def _handle_get_review(self, query: dict[str, list[str]]) -> None:
        status_filter = str((query.get("status") or [""])[0]).strip().lower()
        people = self.store.list_people()
        faces = self.store.list_faces()
        reviews = self.store.list_review_items()
        people_by_id = {str(row.get("person_id")): row for row in people}
        faces_by_id = {str(row.get("face_id")): row for row in faces}
        rows = []
        prototypes = build_person_prototypes(
            faces,
            allowed_embedding_model_ids=ACTIVE_EMBEDDING_MODELS,
        )
        for item in reviews:
            status = str(item.get("status") or "").strip().lower()
            if status_filter and status != status_filter:
                continue
            rows.append(
                self._review_summary(
                    item,
                    people_by_id=people_by_id,
                    faces_by_id=faces_by_id,
                    prototypes=prototypes,
                )
            )
        self._send_json({"ok": True, "reviews": rows})

    def _handle_create_person(self, payload: dict[str, Any]) -> None:
        aliases_raw = payload.get("aliases")
        aliases: list[str]
        if isinstance(aliases_raw, str):
            aliases = [item.strip() for item in aliases_raw.split(",") if item.strip()]
        elif isinstance(aliases_raw, list):
            aliases = [str(item).strip() for item in aliases_raw if str(item).strip()]
        else:
            aliases = []
        person = self.store.add_person(
            name=str(payload.get("display_name") or payload.get("name") or ""),
            aliases=aliases,
            notes=str(payload.get("notes") or ""),
        )
        self._send_json({"ok": True, "person": person}, status=HTTPStatus.CREATED)

    def _handle_update_person(self, payload: dict[str, Any]) -> None:
        person_id = str(payload.get("person_id") or "").strip()
        if not person_id:
            self._error("person_id is required.")
            return
        updates: dict[str, Any] = {}
        if "display_name" in payload or "name" in payload:
            updates["display_name"] = str(
                payload.get("display_name") or payload.get("name") or ""
            ).strip()
        if "aliases" in payload:
            updates["aliases"] = payload.get("aliases")
        if "notes" in payload:
            updates["notes"] = payload.get("notes")
        if not updates:
            self._error("No updates provided.")
            return
        try:
            person = self.store.update_person(person_id, **updates)
        except Exception as exc:
            self._error(str(exc))
            return
        self._send_json({"ok": True, "person": person})

    def _handle_create_face(self, payload: dict[str, Any]) -> None:
        try:
            embedding = parse_embedding(payload.get("embedding")).astype(float).tolist()
        except Exception as exc:
            self._error(str(exc))
            return
        try:
            face = self.store.add_face(
                embedding=embedding,
                person_id=str(payload.get("person_id") or "").strip() or None,
                source_type=str(payload.get("source_type") or "photo"),
                source_path=str(payload.get("source_path") or ""),
                timestamp=str(payload.get("timestamp") or ""),
                bbox=list(payload.get("bbox") or []),
                quality=payload.get("quality"),
                metadata=(
                    payload.get("metadata")
                    if isinstance(payload.get("metadata"), dict)
                    else {}
                ),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        self._send_json({"ok": True, "face": face}, status=HTTPStatus.CREATED)

    def _handle_assign_face(self, payload: dict[str, Any]) -> None:
        face_id = str(payload.get("face_id") or "").strip()
        person_id = str(payload.get("person_id") or "").strip() or None
        has_reviewed_flag = "reviewed_by_human" in payload
        reviewed_by_human = (
            _coerce_bool(payload.get("reviewed_by_human"), False)
            if has_reviewed_flag
            else None
        )
        review_status = str(payload.get("review_status") or "").strip().lower() or None
        if not face_id:
            self._error("face_id is required.")
            return
        try:
            face = self.store.assign_face(
                face_id,
                person_id,
                reviewed_by_human=reviewed_by_human,
                review_status=review_status,
            )
        except Exception as exc:
            self._error(str(exc))
            return
        if reviewed_by_human:
            pending_status = (
                "accepted"
                if str(face.get("person_id") or "").strip()
                else str(review_status or "skipped")
            )
            self._resolve_pending_reviews_for_face(
                face_id,
                status=pending_status,
                decided_person_id=str(face.get("person_id") or "").strip() or None,
            )
        self._send_json({"ok": True, "face": face})

    def _suggestion_policy_from_payload(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "min_similarity": _coerce_float(
                payload.get("min_similarity"), DEFAULT_MIN_SIMILARITY
            ),
            "min_margin": _coerce_float(payload.get("min_margin"), DEFAULT_MIN_MARGIN),
            "min_face_quality": _coerce_float(
                payload.get("min_face_quality"), DEFAULT_MIN_FACE_QUALITY
            ),
            "min_sample_count": max(
                1,
                _coerce_int(payload.get("min_sample_count"), DEFAULT_MIN_SAMPLE_COUNT),
            ),
        }

    def _find_pending_review_for_face(self, face_id: str) -> dict[str, Any] | None:
        for row in self.store.list_review_items():
            if (
                str(row.get("face_id")) == str(face_id)
                and str(row.get("status")) == "pending"
            ):
                return row
        return None

    def _resolve_pending_reviews_for_face(
        self,
        face_id: str,
        *,
        status: str,
        decided_person_id: str | None,
    ) -> None:
        for row in self.store.list_review_items():
            if str(row.get("face_id")) != str(face_id):
                continue
            if str(row.get("status") or "").strip().lower() != "pending":
                continue
            try:
                self.store.resolve_review_item(
                    review_id=str(row.get("review_id") or ""),
                    status=status,
                    decided_person_id=decided_person_id,
                )
            except Exception:
                continue

    def _enqueue_review_for_face(
        self,
        *,
        face_id: str,
        top_k: int = 3,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_margin: float = DEFAULT_MIN_MARGIN,
        min_face_quality: float = DEFAULT_MIN_FACE_QUALITY,
        min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
    ) -> tuple[dict[str, Any], bool]:
        existing = self._find_pending_review_for_face(face_id)
        if existing:
            return existing, True
        face = self.store.get_face(face_id)
        if not face:
            raise ValueError("Unknown face_id.")
        if face_review_status(face) in {"ignored", "rejected"}:
            raise ValueError("Face is already marked ignored/rejected.")
        candidates: list[dict[str, Any]] = []
        if face_embedding_model(face) in ACTIVE_EMBEDDING_MODELS:
            try:
                candidates = suggest_people(
                    query_embedding=face.get("embedding") or [],
                    faces=self.store.list_faces(),
                    top_k=max(1, int(top_k)),
                    min_similarity=float(min_similarity),
                    allowed_embedding_model_ids=ACTIVE_EMBEDDING_MODELS,
                )
            except Exception:
                candidates = []
        suggested, _margin = choose_suggested_candidate(
            candidates=candidates,
            face_quality=face.get("quality") if isinstance(face, dict) else None,
            min_similarity=float(min_similarity),
            min_margin=float(min_margin),
            min_face_quality=float(min_face_quality),
            min_sample_count=int(min_sample_count),
        )
        suggested = suggested or {}
        _score = suggested.get("score")
        review = self.store.add_review_item(
            face_id=face_id,
            candidates=candidates,
            suggested_person_id=str(suggested.get("person_id") or "").strip() or None,
            suggested_score=float(_score) if _score is not None else None,
            status="pending",
        )
        return review, False

    def _resolve_media_path(self, raw_path: Any) -> Path:
        text = str(raw_path or "").strip()
        if not text:
            raise ValueError("Path is required.")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _validate_store_relative_file(self, rel_path: str) -> Path:
        clean_rel = str(rel_path or "").strip()
        if not clean_rel:
            raise ValueError("Face crop is missing.")
        root = self.store.root_dir.resolve()
        candidate = (root / clean_rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Invalid crop path.") from exc
        return candidate

    def _resolve_source_image_path(self, raw_path: str) -> Path | None:
        text = str(raw_path or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return None
        return path

    def _resolve_source_video_path(self, raw_path: str) -> Path | None:
        text = str(raw_path or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return None
        return path

    def _resolve_face_source_path(
        self, face: dict[str, Any]
    ) -> tuple[str, Path] | None:
        source_path = str(face.get("source_path") or "").strip()
        if not source_path:
            return None
        image = self._resolve_source_image_path(source_path)
        if image is not None:
            return ("image", image)
        video = self._resolve_source_video_path(source_path)
        if video is not None:
            return ("video", video)
        return None

    def _face_source_kind(self, face: dict[str, Any]) -> str:
        source_type = str(face.get("source_type") or "").strip().lower()
        if source_type == "vhs":
            return "video"
        source_path = str(face.get("source_path") or "").strip().lower()
        if source_path and Path(source_path).suffix.lower() in IMAGE_SUFFIXES:
            return "image"
        if source_path:
            return "video"
        return ""

    def _parse_timestamp_seconds(self, raw_timestamp: Any) -> float | None:
        text = str(raw_timestamp or "").strip()
        if not text:
            return None
        if ":" not in text:
            try:
                seconds = float(text)
            except Exception:
                return None
            return seconds if seconds >= 0.0 else None

        parts = text.split(":")
        if len(parts) == 2:
            h_raw = "0"
            m_raw, s_raw = parts
        elif len(parts) == 3:
            h_raw, m_raw, s_raw = parts
        else:
            return None
        try:
            hours = float(h_raw)
            minutes = float(m_raw)
            seconds = float(s_raw)
        except Exception:
            return None
        if hours < 0.0 or minutes < 0.0 or seconds < 0.0:
            return None
        return (hours * 3600.0) + (minutes * 60.0) + seconds

    def _extract_video_frame(
        self, video_path: Path, face: dict[str, Any]
    ) -> Any | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            attempts: list[tuple[str, float]] = []
            metadata = face.get("metadata")
            if isinstance(metadata, dict):
                raw_frame = metadata.get("frame_index")
                if raw_frame is not None:
                    try:
                        frame_index = max(0, int(float(raw_frame)))
                        attempts.append(("frame", float(frame_index)))
                    except Exception:
                        pass

            seconds = self._parse_timestamp_seconds(face.get("timestamp"))
            if seconds is not None:
                attempts.append(("msec", float(seconds) * 1000.0))
            if not attempts:
                attempts.append(("frame", 0.0))

            for mode, value in attempts:
                if mode == "frame":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(value))
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(value))
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
            return None
        finally:
            cap.release()

    def _handle_suggest(self, payload: dict[str, Any]) -> None:
        top_k = max(1, _coerce_int(payload.get("top_k"), 3))
        min_similarity = _coerce_float(
            payload.get("min_similarity"), DEFAULT_MIN_SIMILARITY
        )
        try:
            query = parse_embedding(payload.get("embedding"))
        except Exception as exc:
            self._error(str(exc))
            return
        faces = self.store.list_faces()
        try:
            candidates = suggest_people(
                query_embedding=query,
                faces=faces,
                top_k=int(top_k),
                min_similarity=float(min_similarity),
                allowed_embedding_model_ids=ACTIVE_EMBEDDING_MODELS,
            )
        except Exception as exc:
            self._error(str(exc))
            return
        people = self.store.list_people()
        people_by_id = {str(row.get("person_id")): row for row in people}
        for row in candidates:
            person = people_by_id.get(str(row.get("person_id")), {})
            row["person_name"] = str(person.get("display_name", ""))
        self._send_json({"ok": True, "candidates": candidates})

    def _handle_review_enqueue(self, payload: dict[str, Any]) -> None:
        face_id = str(payload.get("face_id") or "").strip()
        if not face_id:
            self._error("face_id is required.")
            return
        top_k = max(1, _coerce_int(payload.get("top_k"), 3))
        policy = self._suggestion_policy_from_payload(payload)
        try:
            review, existing = self._enqueue_review_for_face(
                face_id=face_id,
                top_k=top_k,
                min_similarity=float(policy["min_similarity"]),
                min_margin=float(policy["min_margin"]),
                min_face_quality=float(policy["min_face_quality"]),
                min_sample_count=int(policy["min_sample_count"]),
            )
        except Exception as exc:
            self._error(str(exc))
            return
        status = HTTPStatus.OK if existing else HTTPStatus.CREATED
        self._send_json(
            {"ok": True, "review": review, "existing": bool(existing)}, status=status
        )

    def _handle_review_resolve(self, payload: dict[str, Any]) -> None:
        review_id = str(payload.get("review_id") or "").strip()
        status = str(payload.get("status") or "").strip().lower()
        person_id = str(payload.get("person_id") or "").strip() or None
        if not review_id:
            self._error("review_id is required.")
            return
        review = None
        for row in self.store.list_review_items():
            if str(row.get("review_id")) == review_id:
                review = row
                break
        if not review:
            self._error("Unknown review_id.")
            return
        if status == "accepted" and not person_id:
            person_id = str(review.get("suggested_person_id") or "").strip() or None
            if not person_id:
                self._error(
                    "person_id is required for accepted decisions with no suggestion."
                )
                return
        if status == "skipped":
            try:
                updated_review = self.store.defer_review_item(review_id)
            except Exception as exc:
                self._error(str(exc))
                return
            self._send_json({"ok": True, "review": updated_review, "face": None})
            return
        try:
            updated_review = self.store.resolve_review_item(
                review_id=review_id,
                status=status,
                decided_person_id=person_id,
            )
        except Exception as exc:
            self._error(str(exc))
            return
        assigned_face = None
        if status == "accepted":
            try:
                assigned_face = self.store.assign_face(
                    str(updated_review.get("face_id")),
                    person_id,
                    reviewed_by_human=True,
                    review_status="confirmed",
                )
            except Exception as exc:
                self._error(f"Review updated but face assignment failed: {exc}")
                return
            # Write the confirmed name back to the source image's XMP sidecar,
            # and optionally rewrite the description to use the actual name.
            try:
                assert person_id is not None  # guaranteed by early-return at line ~720
                source_path = str(assigned_face.get("source_path") or "").strip()
                person = self.store.get_person(person_id)
                display_name = (
                    str(person.get("display_name") or "").strip() if person else ""
                )
                if source_path and display_name:
                    img_path = Path(source_path)
                    xmp_path = img_path.with_suffix(".xmp")
                    existing = read_person_in_image(xmp_path)
                    all_names = (
                        existing
                        if display_name.casefold() in {n.casefold() for n in existing}
                        else existing + [display_name]
                    )
                    updated_description: str | None = None
                    if self.lmstudio_url:
                        old_desc = read_xmp_description(xmp_path)
                        if old_desc:
                            updated_description = _rewrite_description_via_lmstudio(
                                old_desc,
                                all_names,
                                base_url=self.lmstudio_url,
                            )
                    merge_persons_xmp(
                        xmp_path,
                        all_names,
                        creator_tool="cast-review",
                        description=updated_description,
                    )

            except Exception as exc:
                # XMP write is best-effort; don't fail the review response.
                xmp_warning = f"XMP write-back failed: {exc}"
                print(f"[cast] {xmp_warning}", file=sys.stderr)
                self._send_json(
                    {
                        "ok": True,
                        "review": updated_review,
                        "face": assigned_face,
                        "xmp_warning": xmp_warning,
                    }
                )
                return
        elif status in {"ignored", "rejected"}:
            try:
                assigned_face = self.store.assign_face(
                    str(updated_review.get("face_id")),
                    None,
                    reviewed_by_human=True,
                    review_status=status,
                )
            except Exception as exc:
                self._error(f"Review updated but face assignment failed: {exc}")
                return
        self._send_json({"ok": True, "review": updated_review, "face": assigned_face})

    def _handle_prune_false_positives(self, payload: dict[str, Any]) -> None:
        max_items = int(payload.get("max_items") or 0)
        pending = [
            row
            for row in self.store.list_review_items()
            if str(row.get("status", "")).strip().lower() == "pending"
        ]
        if max_items > 0:
            pending = pending[: int(max_items)]

        pruned = 0
        checked = 0
        for review in pending:
            review_id = str(review.get("review_id") or "").strip()
            face_id = str(review.get("face_id") or "").strip()
            if not review_id or not face_id:
                continue
            face = self.store.get_face(face_id)
            if not face:
                continue
            crop_rel = str(face.get("crop_path") or "").strip()
            if not crop_rel:
                continue
            try:
                crop_abs = self._validate_store_relative_file(crop_rel)
            except Exception:
                continue
            if not crop_abs.exists():
                continue
            image = cv2.imread(str(crop_abs))
            if image is None:
                continue
            checked += 1
            looks_valid = bool(
                self.server.ingestor.is_valid_face_crop(
                    image,
                )
            )
            if looks_valid:
                continue
            self.store.resolve_review_item(
                review_id=review_id,
                status="rejected",
                decided_person_id=None,
            )
            self.store.assign_face(
                face_id,
                None,
                reviewed_by_human=True,
                review_status="rejected",
            )
            pruned += 1

        self._send_json(
            {
                "ok": True,
                "checked": int(checked),
                "pruned": int(pruned),
                "remaining_pending": int(
                    len(
                        [
                            row
                            for row in self.store.list_review_items()
                            if str(row.get("status", "")).strip().lower() == "pending"
                        ]
                    )
                ),
            }
        )

    def _handle_reset_pending_unknown(self, payload: dict[str, Any]) -> None:
        remove_crops = str(payload.get("remove_crops", "1")).strip() not in {
            "0",
            "false",
            "False",
        }
        result = self.store.reset_pending_unknown(remove_crops=remove_crops)
        state = self._state_payload()
        self._send_json(
            {
                "ok": True,
                "remove_crops": bool(remove_crops),
                **result,
                "counts": state.get("counts", {}),
            }
        )

    def _handle_ingest_photo(self, payload: dict[str, Any]) -> None:
        image_path = payload.get("image_path")
        source_path = payload.get("source_path")
        min_size = int(payload.get("min_size") or 40)
        max_faces = int(payload.get("max_faces") or 50)
        auto_queue = str(payload.get("auto_queue", "1")).strip() not in {
            "0",
            "false",
            "False",
        }
        top_k = max(1, _coerce_int(payload.get("top_k"), 3))
        policy = self._suggestion_policy_from_payload(payload)

        try:
            path = self._resolve_media_path(image_path)
            faces = self.server.ingestor.ingest_photo(
                image_path=path,
                source_path=str(source_path or path),
                min_size=min_size,
                max_faces=max_faces,
            )
        except Exception as exc:
            self._error(str(exc))
            return

        reviews_created = 0
        reviews_reused = 0
        if auto_queue:
            for face in faces:
                face_id = str(face.get("face_id") or "").strip()
                if not face_id:
                    continue
                try:
                    _review, existing = self._enqueue_review_for_face(
                        face_id=face_id,
                        top_k=top_k,
                        min_similarity=float(policy["min_similarity"]),
                        min_margin=float(policy["min_margin"]),
                        min_face_quality=float(policy["min_face_quality"]),
                        min_sample_count=int(policy["min_sample_count"]),
                    )
                    if existing:
                        reviews_reused += 1
                    else:
                        reviews_created += 1
                except Exception:
                    continue

        people_by_id = {
            str(row.get("person_id")): row for row in self.store.list_people()
        }
        face_rows = [self._face_summary(face, people_by_id) for face in faces]
        self._send_json(
            {
                "ok": True,
                "faces": face_rows,
                "faces_created": len(face_rows),
                "reviews_created": int(reviews_created),
                "reviews_reused": int(reviews_reused),
                "source_path": str(path),
            },
            status=HTTPStatus.CREATED,
        )

    def _handle_ingest_vhs(self, payload: dict[str, Any]) -> None:
        video_path = payload.get("video_path")
        source_path = payload.get("source_path")
        min_size = int(payload.get("min_size") or 40)
        max_faces = int(payload.get("max_faces") or 120)
        sample_every_seconds = float(payload.get("sample_every_seconds") or 2.0)
        max_duration_seconds = float(payload.get("max_duration_seconds") or 0.0)
        auto_queue = str(payload.get("auto_queue", "1")).strip() not in {
            "0",
            "false",
            "False",
        }
        top_k = max(1, _coerce_int(payload.get("top_k"), 3))
        policy = self._suggestion_policy_from_payload(payload)

        try:
            path = self._resolve_media_path(video_path)
            result = self.server.ingestor.ingest_vhs(
                video_path=path,
                source_path=str(source_path or path),
                sample_every_seconds=sample_every_seconds,
                min_size=min_size,
                max_faces=max_faces,
                max_duration_seconds=max_duration_seconds,
            )
        except Exception as exc:
            self._error(str(exc))
            return

        faces = list(result.get("faces") or [])
        reviews_created = 0
        reviews_reused = 0
        if auto_queue:
            for face in faces:
                face_id = str(face.get("face_id") or "").strip()
                if not face_id:
                    continue
                try:
                    _review, existing = self._enqueue_review_for_face(
                        face_id=face_id,
                        top_k=top_k,
                        min_similarity=float(policy["min_similarity"]),
                        min_margin=float(policy["min_margin"]),
                        min_face_quality=float(policy["min_face_quality"]),
                        min_sample_count=int(policy["min_sample_count"]),
                    )
                    if existing:
                        reviews_reused += 1
                    else:
                        reviews_created += 1
                except Exception:
                    continue

        people_by_id = {
            str(row.get("person_id")): row for row in self.store.list_people()
        }
        face_rows = [self._face_summary(face, people_by_id) for face in faces]
        self._send_json(
            {
                "ok": True,
                "faces": face_rows,
                "faces_created": int(result.get("faces_created", len(face_rows))),
                "sampled_frames": int(result.get("sampled_frames", 0)),
                "reviews_created": int(reviews_created),
                "reviews_reused": int(reviews_reused),
                "source_path": str(path),
            },
            status=HTTPStatus.CREATED,
        )

    def _handle_ingest_photo_scan(self, payload: dict[str, Any]) -> None:
        root_dir = payload.get("photo_albums_root") or DEFAULT_PHOTO_ALBUMS_ROOT
        view_glob = str(payload.get("view_glob") or "*_View")
        recursive = str(payload.get("recursive", "1")).strip() not in {
            "0",
            "false",
            "False",
        }
        min_size = int(payload.get("min_size") or 40)
        max_faces_per_photo = int(payload.get("max_faces_per_photo") or 50)
        max_files = int(payload.get("max_files") or 0)
        auto_queue = str(payload.get("auto_queue", "1")).strip() not in {
            "0",
            "false",
            "False",
        }
        top_k = max(1, _coerce_int(payload.get("top_k"), 3))
        policy = self._suggestion_policy_from_payload(payload)

        try:
            result = self.server.ingestor.ingest_photo_album_views(
                photo_albums_root=self._resolve_media_path(root_dir),
                view_glob=view_glob,
                recursive=recursive,
                min_size=min_size,
                max_faces_per_photo=max_faces_per_photo,
                max_files=max_files,
            )
        except Exception as exc:
            self._error(str(exc))
            return

        faces = list(result.get("faces") or [])
        reviews_created = 0
        reviews_reused = 0
        if auto_queue:
            for face in faces:
                face_id = str(face.get("face_id") or "").strip()
                if not face_id:
                    continue
                try:
                    _review, existing = self._enqueue_review_for_face(
                        face_id=face_id,
                        top_k=top_k,
                        min_similarity=float(policy["min_similarity"]),
                        min_margin=float(policy["min_margin"]),
                        min_face_quality=float(policy["min_face_quality"]),
                        min_sample_count=int(policy["min_sample_count"]),
                    )
                    if existing:
                        reviews_reused += 1
                    else:
                        reviews_created += 1
                except Exception:
                    continue

        top_photos = sorted(
            list(result.get("per_photo") or []),
            key=lambda row: int(row.get("faces_created", 0)),
            reverse=True,
        )[:10]
        self._send_json(
            {
                "ok": True,
                "photo_files_scanned": int(result.get("photo_files_scanned", 0)),
                "faces_created": int(result.get("faces_created", len(faces))),
                "reviews_created": int(reviews_created),
                "reviews_reused": int(reviews_reused),
                "photo_albums_root": str(result.get("photo_albums_root", root_dir)),
                "view_glob": str(result.get("view_glob", view_glob)),
                "top_photos": top_photos,
            },
            status=HTTPStatus.CREATED,
        )

    def _handle_get_face_crop(self, face_id: str) -> None:
        face = self.store.get_face(str(face_id))
        if not face:
            self._not_found()
            return
        crop_path = str(face.get("crop_path") or "").strip()
        if not crop_path:
            self._not_found()
            return
        try:
            path = self._validate_store_relative_file(crop_path)
        except Exception as exc:
            self._error(str(exc), status=HTTPStatus.BAD_REQUEST)
            return
        if not path.exists():
            self._not_found()
            return
        try:
            data = path.read_bytes()
        except Exception as exc:
            self._error(
                f"Unable to read crop image: {exc}",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_bytes(data, "image/jpeg", status=HTTPStatus.OK)

    def _handle_get_face_source(
        self, face_id: str, query: dict[str, list[str]]
    ) -> None:
        face = self.store.get_face(str(face_id))
        if not face:
            self._not_found()
            return
        source_ref = self._resolve_face_source_path(face)
        if source_ref is None:
            self._error(
                "Source image is unavailable for this face.",
                status=HTTPStatus.NOT_FOUND,
            )
            return
        source_kind, source_path = source_ref
        if source_kind == "video":
            image = self._extract_video_frame(source_path, face)
            if image is None:
                self._error(
                    "Unable to read source frame from video.",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
        else:
            image = cv2.imread(str(source_path))
            if image is None:
                self._error(
                    "Unable to read source image.",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

        highlight = str((query.get("highlight") or ["1"])[0]).strip() not in {
            "0",
            "false",
            "False",
        }
        max_dim_raw = str((query.get("max_dim") or ["1800"])[0]).strip()
        try:
            max_dim = int(max_dim_raw)
        except Exception:
            max_dim = 1800
        max_dim = max(256, min(5000, int(max_dim)))

        if highlight:
            bbox = list(face.get("bbox") or [])
            if len(bbox) >= 4:
                try:
                    x, y, w, h = [int(float(v)) for v in bbox[:4]]
                    if w > 0 and h > 0:
                        x0 = max(0, min(x, image.shape[1] - 1))
                        y0 = max(0, min(y, image.shape[0] - 1))
                        x1 = max(x0 + 1, min(x + w, image.shape[1]))
                        y1 = max(y0 + 1, min(y + h, image.shape[0]))
                        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 3)
                except Exception:
                    pass

        h, w = image.shape[:2]
        top = max(int(w), int(h))
        if top > max_dim:
            scale = float(max_dim) / float(top)
            out_w = max(1, int(round(float(w) * scale)))
            out_h = max(1, int(round(float(h) * scale)))
            image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # type: ignore[arg-type]
        if not ok:
            self._error(
                "Failed to encode source image.",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_bytes(bytes(encoded), "image/jpeg", status=HTTPStatus.OK)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        parts = [item for item in path.split("/") if item]
        if (
            len(parts) == 4
            and parts[0] == "api"
            and parts[1] == "faces"
            and parts[3] == "crop"
        ):
            self._handle_get_face_crop(parts[2])
            return
        if (
            len(parts) == 4
            and parts[0] == "api"
            and parts[1] == "faces"
            and parts[3] == "source"
        ):
            self._handle_get_face_source(parts[2], query)
            return

        if path == "/":
            if not _INDEX.exists():
                self._send_html("<h1>cast web UI is missing static/index.html</h1>")
                return
            self._send_html(_INDEX.read_text(encoding="utf-8"))
            return
        if path == "/api/state":
            self._handle_get_state()
            return
        if path == "/api/people":
            self._handle_get_people()
            return
        if path == "/api/faces":
            self._handle_get_faces(query)
            return
        if path == "/api/review":
            self._handle_get_review(query)
            return
        self._not_found()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self._read_json()
        except Exception as exc:
            self._error(str(exc))
            return

        if path == "/api/people":
            self._handle_create_person(payload)
            return
        if path == "/api/people/update":
            self._handle_update_person(payload)
            return
        if path == "/api/faces":
            self._handle_create_face(payload)
            return
        if path == "/api/faces/assign":
            self._handle_assign_face(payload)
            return
        if path == "/api/suggest":
            self._handle_suggest(payload)
            return
        if path == "/api/review/enqueue":
            self._handle_review_enqueue(payload)
            return
        if path == "/api/review/resolve":
            self._handle_review_resolve(payload)
            return
        if path == "/api/review/prune_false_positives":
            self._handle_prune_false_positives(payload)
            return
        if path == "/api/reset/pending_unknown":
            self._handle_reset_pending_unknown(payload)
            return
        if path == "/api/ingest/photo":
            self._handle_ingest_photo(payload)
            return
        if path == "/api/ingest/vhs":
            self._handle_ingest_vhs(payload)
            return
        if path == "/api/ingest/photos/scan":
            self._handle_ingest_photo_scan(payload)
            return
        self._not_found()


def run(
    host: str, port: int, store: TextFaceStore, lmstudio_url: str = DEFAULT_LMSTUDIO_URL
) -> None:
    server = CastHTTPServer(
        host=host, port=port, store=store, lmstudio_url=lmstudio_url
    )
    print(f"Cast web UI running at http://{host}:{int(port)}")
    print(f"Store directory: {store.root_dir}")
    print(f"LM Studio URL: {server.lmstudio_url} (description rewrite on face confirm)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
