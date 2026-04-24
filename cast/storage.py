from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(path: Path) -> Any:
    if path.name == "people.json":
        return {"people": []}
    return []


FACE_REVIEW_STATUSES = {"confirmed", "ignored", "rejected"}
_WRITE_RETRY_ATTEMPTS = 8
_WRITE_RETRY_DELAY_SECONDS = 0.1


def normalize_face_review_status(value: Any, *, person_id: Any = None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "confirmed" if str(person_id or "").strip() else ""
    if text not in FACE_REVIEW_STATUSES:
        raise ValueError("review_status must be one of: confirmed, ignored, rejected")
    if text == "confirmed" and not str(person_id or "").strip():
        raise ValueError("confirmed review_status requires a person_id")
    return text


def face_review_status(face: dict[str, Any]) -> str:
    if not isinstance(face, dict):
        return ""
    person_id = str(face.get("person_id") or "").strip()
    raw = str(face.get("review_status") or "").strip().lower()
    if raw in FACE_REVIEW_STATUSES:
        return "confirmed" if raw == "confirmed" and person_id else raw
    reviewed = face.get("reviewed_by_human")
    if isinstance(reviewed, bool):
        return "confirmed" if reviewed and person_id else ""
    if person_id:
        # Legacy rows with an assigned identity were created via manual review.
        return "confirmed"
    return ""


def face_is_human_reviewed(face: dict[str, Any]) -> bool:
    if not isinstance(face, dict):
        return False
    reviewed = face.get("reviewed_by_human")
    if isinstance(reviewed, bool):
        return reviewed
    return face_review_status(face) in FACE_REVIEW_STATUSES


def normalize_face_record(face: dict[str, Any]) -> dict[str, Any]:
    row = dict(face or {})
    row["person_id"] = str(row.get("person_id") or "").strip() or None
    row["reviewed_by_human"] = bool(face_is_human_reviewed(row))
    row["review_status"] = face_review_status(row)
    row["reviewed_at"] = str(row.get("reviewed_at") or "").strip()
    return row


_FACE_CHUNK_SIZE = 1000


def _chunk_filename(index: int) -> str:
    return f"faces_{index:04d}.jsonl"


class TextFaceStore:
    """Small file-backed store for people, faces, and review queue."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.people_path = self.root_dir / "people.json"
        self.faces_queue_path = self.root_dir / "faces.jsonl"
        self.review_path = self.root_dir / "review_queue.jsonl"
        self._lock = threading.RLock()
        self._face_manifest: dict[str, int] | None = None
        self._chunk_cache: dict[int, list[dict[str, Any]]] = {}
        self._chunk_dir = self.root_dir

    @property
    def _faces_manifest_path(self) -> Path:
        return self.root_dir / "faces_manifest.json"

    @property
    def faces_path(self) -> Path:
        return self.faces_queue_path

    def _load_manifest(self) -> dict[str, int]:
        if self._face_manifest is not None:
            return self._face_manifest
        manifest_path = self._faces_manifest_path
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._face_manifest = {str(k): int(v) for k, v in data.items()}
                    return self._face_manifest
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        self._face_manifest = {}
        return self._face_manifest

    def _save_manifest(self, manifest: dict[str, int]) -> None:
        self._face_manifest = manifest
        self._write_json(self._faces_manifest_path, manifest)

    def _chunk_path(self, index: int) -> Path:
        return self._chunk_dir / _chunk_filename(index)

    def _read_chunk(self, index: int) -> list[dict[str, Any]]:
        if index in self._chunk_cache:
            return self._chunk_cache[index]
        path = self._chunk_path(index)
        if not path.exists():
            return []
        rows = self._read_json(path)
        self._chunk_cache[index] = rows
        return rows

    def _write_chunk(self, index: int, rows: list[dict[str, Any]]) -> None:
        self._write_json(self._chunk_path(index), rows)
        self._chunk_cache[index] = rows

    def _migrate_queue_to_chunks(self) -> dict[str, int]:
        queue_path = self.faces_queue_path
        if not queue_path.exists():
            return {"migrated": 0, "chunks": 0}
        queue_rows = self._read_json(queue_path)
        if not isinstance(queue_rows, list):
            return {"migrated": 0, "chunks": 0}
        if not queue_rows:
            return {"migrated": 0, "chunks": 0}
        manifest = self._load_manifest()
        migrated = 0
        chunk_index = 0
        while queue_rows:
            chunk_rows = queue_rows[:_FACE_CHUNK_SIZE]
            queue_rows = queue_rows[_FACE_CHUNK_SIZE:]
            for row in chunk_rows:
                face_id = str(row.get("face_id") or "").strip()
                if face_id:
                    manifest[face_id] = chunk_index
            self._write_chunk(chunk_index, chunk_rows)
            chunk_index += 1
            migrated += len(chunk_rows)
        self._save_manifest(manifest)
        queue_path.unlink(missing_ok=True)
        return {"migrated": migrated, "chunks": chunk_index}

    def ensure_files(self) -> None:
        with self._lock:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_json_file(self.people_path)
            self._ensure_json_file(self.faces_queue_path)
            self._ensure_json_file(self.review_path)

    def _read_all_faces(self) -> list[dict[str, Any]]:
        manifest = self._load_manifest()
        rows: list[dict[str, Any]] = []
        chunk_indices = set(manifest.values()) if manifest else set()
        for idx in chunk_indices:
            chunk_rows = self._read_chunk(idx)
            if isinstance(chunk_rows, list):
                rows.extend(chunk_rows)
        queue_rows = self._read_json(self.faces_queue_path)
        if isinstance(queue_rows, list):
            rows.extend(queue_rows)
        return rows

    def _write_all_faces(self, face_rows: list[dict[str, Any]]) -> None:
        manifest = self._load_manifest()
        while face_rows:
            chunk_rows = face_rows[:_FACE_CHUNK_SIZE]
            face_rows = face_rows[_FACE_CHUNK_SIZE:]
            chunk_idx = len([p for p in manifest.values() if isinstance(p, int)])
            for row in chunk_rows:
                face_id = str(row.get("face_id") or "").strip()
                if face_id:
                    manifest[face_id] = chunk_idx
            self._write_chunk(chunk_idx, chunk_rows)
            chunk_idx += 1
        self._save_manifest(manifest)
        self.faces_queue_path.unlink(missing_ok=True)

    def _clear_chunks(self) -> None:
        manifest = self._load_manifest()
        for idx in set(manifest.values()):
            self._chunk_path(idx).unlink(missing_ok=True)
        self._save_manifest({})
        self._face_manifest = {}

    def _ensure_json_file(self, path: Path) -> None:
        if path.exists():
            return
        self._write_json(path, _json_default(path))

    def _is_jsonl(self, path: Path) -> bool:
        return path.suffix.lower() == ".jsonl"

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            return _json_default(path)
        if self._is_jsonl(path):
            text = path.read_text(encoding="utf-8")
            rows = []
            for raw in text.splitlines():
                line = str(raw or "").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
            if rows:
                return rows
            stripped = text.strip()
            if stripped.startswith("["):
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return []
                if isinstance(payload, list):
                    return [dict(item) for item in payload if isinstance(item, dict)]
            return rows

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return _json_default(path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _json_default(path)

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            suffix=".tmp",
        ) as handle:
            if self._is_jsonl(path):
                for row in list(data or []):
                    if not isinstance(row, dict):
                        continue
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            else:
                json.dump(data, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
            temp_path = Path(handle.name)
        for attempt in range(_WRITE_RETRY_ATTEMPTS):
            try:
                temp_path.replace(path)
                return
            except PermissionError:
                if attempt >= _WRITE_RETRY_ATTEMPTS - 1:
                    temp_path.unlink(missing_ok=True)
                    raise
                time.sleep(_WRITE_RETRY_DELAY_SECONDS)

    def list_people(self) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._read_json(self.people_path)
            rows = payload.get("people", []) if isinstance(payload, dict) else []
            return [dict(row) for row in rows if isinstance(row, dict)]

    def add_person(
        self,
        name: str,
        aliases: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("Person name is required.")
        clean_aliases = [str(item).strip() for item in (aliases or []) if str(item).strip()]
        now = utc_now_iso()
        row = {
            "person_id": str(uuid.uuid4()),
            "display_name": clean_name,
            "aliases": clean_aliases,
            "notes": str(notes or "").strip(),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            payload = self._read_json(self.people_path)
            people = payload.get("people", []) if isinstance(payload, dict) else []
            people.append(row)
            self._write_json(self.people_path, {"people": people})
        return dict(row)

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        for person in self.list_people():
            if str(person.get("person_id")) == str(person_id):
                return person
        return None

    def update_person(self, person_id: str, **fields: Any) -> dict[str, Any]:
        person_key = str(person_id or "").strip()
        if not person_key:
            raise ValueError("person_id is required.")
        now = utc_now_iso()
        with self._lock:
            payload = self._read_json(self.people_path)
            people = payload.get("people", []) if isinstance(payload, dict) else []
            if not isinstance(people, list):
                people = []
            for row in people:
                if not isinstance(row, dict):
                    continue
                if str(row.get("person_id")) != person_key:
                    continue
                for key, value in dict(fields or {}).items():
                    if key in {"person_id", "created_at"}:
                        continue
                    if key == "display_name":
                        clean_name = str(value or "").strip()
                        if not clean_name:
                            raise ValueError("Person name is required.")
                        row[key] = clean_name
                        continue
                    if key == "aliases":
                        if isinstance(value, list):
                            row[key] = [str(item).strip() for item in value if str(item).strip()]
                        elif isinstance(value, str):
                            row[key] = [part.strip() for part in value.split(",") if part.strip()]
                        continue
                    if key == "notes":
                        row[key] = str(value or "").strip()
                row["updated_at"] = now
                self._write_json(self.people_path, {"people": people})
                return dict(row)
        raise ValueError(f"Unknown person_id: {person_key}")

    def list_faces(self) -> list[dict[str, Any]]:
        with self._lock:
            manifest = self._load_manifest()
            rows: list[dict[str, Any]] = []
            chunk_indices = set(manifest.values()) if manifest else set()
            for idx in chunk_indices:
                chunk_rows = self._read_chunk(idx)
                if isinstance(chunk_rows, list):
                    rows.extend(chunk_rows)
            queue_rows = self._read_json(self.faces_queue_path)
            if isinstance(queue_rows, list):
                rows.extend(queue_rows)
            return [normalize_face_record(row) for row in rows if isinstance(row, dict)]

    def get_face(self, face_id: str) -> dict[str, Any] | None:
        face_key = str(face_id or "").strip()
        if not face_key:
            return None
        with self._lock:
            manifest = self._load_manifest()
            if face_key in manifest:
                chunk_idx = manifest[face_key]
                chunk_rows = self._read_chunk(chunk_idx)
                for row in chunk_rows:
                    if str(row.get("face_id")) == face_key:
                        return normalize_face_record(row)
            queue_rows = self._read_json(self.faces_queue_path)
            if isinstance(queue_rows, list):
                for row in queue_rows:
                    if str(row.get("face_id")) == face_key:
                        return normalize_face_record(row)
            return None

    def list_faces_for_source(self, source_path: str) -> list[dict[str, Any]]:
        source_key = str(source_path or "").strip()
        if not source_key:
            return []
        with self._lock:
            manifest = self._load_manifest()
            results: list[dict[str, Any]] = []
            chunk_indices = set(manifest.values()) if manifest else set()
            for idx in chunk_indices:
                chunk_rows = self._read_chunk(idx)
                if isinstance(chunk_rows, list):
                    for row in chunk_rows:
                        if str(row.get("source_path") or "").strip() == source_key:
                            results.append(normalize_face_record(row))
            queue_rows = self._read_json(self.faces_queue_path)
            if isinstance(queue_rows, list):
                for row in queue_rows:
                    if str(row.get("source_path") or "").strip() == source_key:
                        results.append(normalize_face_record(row))
            return results

    def add_face(
        self,
        *,
        embedding: list[float] | None = None,
        person_id: str | None = None,
        source_type: str = "photo",
        source_path: str = "",
        timestamp: str | None = None,
        bbox: list[float] | None = None,
        quality: float | None = None,
        metadata: dict[str, Any] | None = None,
        crop_path: str = "",
    ) -> dict[str, Any]:
        parsed_embedding = [float(item) for item in list(embedding or [])]
        person_key = str(person_id).strip() if person_id is not None else ""
        now = utc_now_iso()
        row = {
            "face_id": str(uuid.uuid4()),
            "person_id": person_key or None,
            "source_type": str(source_type or "photo").strip().lower(),
            "source_path": str(source_path or "").strip(),
            "timestamp": str(timestamp or "").strip(),
            "bbox": list(bbox or []),
            "quality": float(quality) if quality is not None else None,
            "embedding": parsed_embedding,
            "crop_path": str(crop_path or "").strip(),
            "metadata": dict(metadata or {}),
            "reviewed_by_human": False,
            "review_status": "",
            "reviewed_at": "",
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            rows = self._read_json(self.faces_queue_path)
            if not isinstance(rows, list):
                rows = []
            rows.append(row)
            self._write_json(self.faces_queue_path, rows)
        return normalize_face_record(row)

    def update_face(self, face_id: str, **fields: Any) -> dict[str, Any]:
        face_key = str(face_id or "").strip()
        if not face_key:
            raise ValueError("face_id is required.")
        now = utc_now_iso()
        with self._lock:
            manifest = self._load_manifest()
            if face_key in manifest:
                chunk_idx = manifest[face_key]
                rows = self._read_chunk(chunk_idx)
                if not isinstance(rows, list):
                    rows = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("face_id")) != face_key:
                        continue
                    for key, value in dict(fields or {}).items():
                        if key in {"face_id", "created_at"}:
                            continue
                        if key == "reviewed_by_human":
                            row[key] = bool(value)
                            continue
                        if key == "review_status":
                            row[key] = normalize_face_review_status(
                                value,
                                person_id=row.get("person_id"),
                            )
                            continue
                        if key == "reviewed_at":
                            row[key] = str(value or "").strip()
                            continue
                        row[key] = value
                    row["updated_at"] = now
                    self._write_chunk(chunk_idx, rows)
                    return normalize_face_record(row)
            queue_rows = self._read_json(self.faces_queue_path)
            if not isinstance(queue_rows, list):
                queue_rows = []
            for row in queue_rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("face_id")) != face_key:
                    continue
                for key, value in dict(fields or {}).items():
                    if key in {"face_id", "created_at"}:
                        continue
                    if key == "reviewed_by_human":
                        row[key] = bool(value)
                        continue
                    if key == "review_status":
                        row[key] = normalize_face_review_status(
                            value,
                            person_id=row.get("person_id"),
                        )
                        continue
                    if key == "reviewed_at":
                        row[key] = str(value or "").strip()
                        continue
                    row[key] = value
                row["updated_at"] = now
                self._write_json(self.faces_queue_path, queue_rows)
                return normalize_face_record(row)
        raise ValueError(f"Unknown face_id: {face_key}")

    def assign_face(
        self,
        face_id: str,
        person_id: str | None,
        *,
        reviewed_by_human: bool | None = None,
        review_status: str | None = None,
    ) -> dict[str, Any]:
        face_key = str(face_id or "").strip()
        if not face_key:
            raise ValueError("face_id is required.")
        person_key = str(person_id or "").strip() or None
        now = utc_now_iso()
        normalized_status: str | None = None
        if review_status is not None:
            normalized_status = normalize_face_review_status(review_status, person_id=person_key)
        with self._lock:
            manifest = self._load_manifest()
            if face_key in manifest:
                chunk_idx = manifest[face_key]
                rows = self._read_chunk(chunk_idx)
                if not isinstance(rows, list):
                    rows = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("face_id")) != face_key:
                        continue
                    row["person_id"] = person_key
                    if normalized_status is not None:
                        row["review_status"] = normalized_status
                        if normalized_status in {"ignored", "rejected"}:
                            row["person_id"] = None
                    elif reviewed_by_human is True and person_key:
                        row["review_status"] = "confirmed"
                    if reviewed_by_human is not None:
                        row["reviewed_by_human"] = bool(reviewed_by_human)
                        row["reviewed_at"] = now if bool(reviewed_by_human) else ""
                    elif normalized_status in FACE_REVIEW_STATUSES:
                        row["reviewed_by_human"] = True
                        row["reviewed_at"] = now
                    row["updated_at"] = now
                    self._write_chunk(chunk_idx, rows)
                    return normalize_face_record(row)
            queue_rows = self._read_json(self.faces_queue_path)
            if not isinstance(queue_rows, list):
                queue_rows = []
            for row in queue_rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("face_id")) != face_key:
                    continue
                row["person_id"] = person_key
                if normalized_status is not None:
                    row["review_status"] = normalized_status
                    if normalized_status in {"ignored", "rejected"}:
                        row["person_id"] = None
                elif reviewed_by_human is True and person_key:
                    row["review_status"] = "confirmed"
                if reviewed_by_human is not None:
                    row["reviewed_by_human"] = bool(reviewed_by_human)
                    row["reviewed_at"] = now if bool(reviewed_by_human) else ""
                elif normalized_status in FACE_REVIEW_STATUSES:
                    row["reviewed_by_human"] = True
                    row["reviewed_at"] = now
                row["updated_at"] = now
                self._write_json(self.faces_queue_path, queue_rows)
                return normalize_face_record(row)
        raise ValueError(f"Unknown face_id: {face_key}")

    def list_review_items(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._read_json(self.review_path)
            if not isinstance(rows, list):
                return []
            return [dict(row) for row in rows if isinstance(row, dict)]

    def add_review_item(
        self,
        *,
        face_id: str,
        candidates: list[dict[str, Any]],
        suggested_person_id: str | None,
        suggested_score: float | None,
        status: str = "pending",
        name_hints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        face_key = str(face_id or "").strip()
        if not face_key:
            raise ValueError("face_id is required.")
        suggested_key = str(suggested_person_id).strip() if suggested_person_id is not None else ""
        now = utc_now_iso()
        row = {
            "review_id": str(uuid.uuid4()),
            "face_id": face_key,
            "candidates": [dict(item) for item in (candidates or []) if isinstance(item, dict)],
            "suggested_person_id": suggested_key or None,
            "suggested_score": (float(suggested_score) if suggested_score is not None else None),
            "status": str(status or "pending").strip().lower(),
            "decided_person_id": None,
            "decided_at": "",
            "created_at": now,
            "updated_at": now,
            "name_hints": [dict(h) for h in (name_hints or []) if isinstance(h, dict)],
        }
        with self._lock:
            rows = self._read_json(self.review_path)
            if not isinstance(rows, list):
                rows = []
            rows.append(row)
            self._write_json(self.review_path, rows)
        return dict(row)

    def resolve_review_item(
        self,
        *,
        review_id: str,
        status: str,
        decided_person_id: str | None = None,
    ) -> dict[str, Any]:
        review_key = str(review_id or "").strip()
        clean_status = str(status or "").strip().lower()
        if clean_status not in {"accepted", "rejected", "ignored", "skipped"}:
            raise ValueError("status must be one of: accepted, rejected, ignored, skipped")
        now = utc_now_iso()
        person_key = str(decided_person_id or "").strip() or None
        with self._lock:
            rows = self._read_json(self.review_path)
            if not isinstance(rows, list):
                rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("review_id")) != review_key:
                    continue
                row["status"] = clean_status
                row["decided_person_id"] = person_key
                row["decided_at"] = now
                row["updated_at"] = now
                self._write_json(self.review_path, rows)
                return dict(row)
        raise ValueError(f"Unknown review_id: {review_key}")

    def defer_review_item(self, review_id: str) -> dict[str, Any]:
        review_key = str(review_id or "").strip()
        if not review_key:
            raise ValueError("review_id is required.")
        now = utc_now_iso()
        with self._lock:
            rows = self._read_json(self.review_path)
            if not isinstance(rows, list):
                rows = []
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                if str(row.get("review_id")) != review_key:
                    continue
                row["status"] = "pending"
                row["skip_count"] = int(row.get("skip_count") or 0) + 1
                row["last_skipped_at"] = now
                row["updated_at"] = now
                moved = dict(row)
                rows.pop(index)
                rows.append(moved)
                self._write_json(self.review_path, rows)
                return dict(moved)
        raise ValueError(f"Unknown review_id: {review_key}")

    def bulk_resolve_reviews(
        self,
        *,
        review_ids: list[str],
        status: str,
    ) -> dict[str, int]:
        clean_status = str(status or "").strip().lower()
        if clean_status not in {"ignored", "rejected", "skipped"}:
            raise ValueError("status must be one of: ignored, rejected, skipped")

        ordered_review_ids: list[str] = []
        review_id_set: set[str] = set()
        for raw_review_id in review_ids:
            review_id = str(raw_review_id or "").strip()
            if not review_id or review_id in review_id_set:
                continue
            review_id_set.add(review_id)
            ordered_review_ids.append(review_id)
        if not ordered_review_ids:
            raise ValueError("review_ids is required.")

        now = utc_now_iso()
        with self._lock:
            review_rows = self._read_json(self.review_path)
            if not isinstance(review_rows, list):
                review_rows = []

            matched_reviews: list[dict[str, Any]] = []
            matched_review_ids: set[str] = set()
            matched_face_ids: set[str] = set()
            kept_reviews: list[dict[str, Any]] = []

            for row in review_rows:
                if not isinstance(row, dict):
                    continue
                review_id = str(row.get("review_id") or "").strip()
                if review_id not in review_id_set:
                    kept_reviews.append(row)
                    continue
                matched_review_ids.add(review_id)
                updated = dict(row)
                if clean_status == "skipped":
                    updated["status"] = "pending"
                    updated["skip_count"] = int(updated.get("skip_count") or 0) + 1
                    updated["last_skipped_at"] = now
                    updated["decided_person_id"] = None
                    updated["decided_at"] = ""
                    matched_reviews.append(updated)
                    continue
                updated["status"] = clean_status
                updated["decided_person_id"] = None
                updated["decided_at"] = now
                face_id = str(updated.get("face_id") or "").strip()
                if face_id:
                    matched_face_ids.add(face_id)
                matched_reviews.append(updated)

            missing_review_ids = [review_id for review_id in ordered_review_ids if review_id not in matched_review_ids]
            if missing_review_ids:
                raise ValueError(f"Unknown review_id: {missing_review_ids[0]}")

            for row in matched_reviews:
                row["updated_at"] = now
            if clean_status == "skipped":
                self._write_json(self.review_path, kept_reviews + matched_reviews)
                return {
                    "updated_reviews": int(len(matched_reviews)),
                    "updated_faces": 0,
                }

            face_rows = self._read_all_faces()
            if not isinstance(face_rows, list):
                face_rows = []

            updated_faces = 0
            for row in face_rows:
                if not isinstance(row, dict):
                    continue
                face_id = str(row.get("face_id") or "").strip()
                if face_id not in matched_face_ids:
                    continue
                row["person_id"] = None
                row["review_status"] = clean_status
                row["reviewed_by_human"] = True
                row["reviewed_at"] = now
                row["updated_at"] = now
                updated_faces += 1

            if updated_faces != len(matched_face_ids):
                raise ValueError("One or more review items reference an unknown face_id.")

            self._write_json(self.review_path, kept_reviews + matched_reviews)
            self._write_all_faces(face_rows)
            return {
                "updated_reviews": int(len(matched_reviews)),
                "updated_faces": int(updated_faces),
            }

    def _remove_crop_paths(self, crop_paths: list[Path]) -> int:
        removed_crops = 0
        root = self.root_dir.resolve()
        for rel in crop_paths:
            path = (root / rel).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                continue
            try:
                if path.exists():
                    path.unlink()
                    removed_crops += 1
            except Exception:
                continue
        return int(removed_crops)

    def remove_faces_for_sources(
        self,
        *,
        source_paths: list[str],
        remove_crops: bool = True,
    ) -> dict[str, int]:
        source_keys = {str(path or "").strip() for path in source_paths if str(path or "").strip()}
        if not source_keys:
            return {
                "removed_faces": 0,
                "removed_reviews": 0,
                "removed_crops": 0,
            }

        removed_faces = 0
        removed_reviews = 0
        removed_crops = 0

        with self._lock:
            face_rows = self._read_all_faces()
            if not isinstance(face_rows, list):
                face_rows = []
            review_rows = self._read_json(self.review_path)
            if not isinstance(review_rows, list):
                review_rows = []

            removed_face_ids: set[str] = set()
            crop_paths: list[Path] = []
            kept_faces: list[dict[str, Any]] = []
            for row in face_rows:
                if not isinstance(row, dict):
                    continue
                source_path = str(row.get("source_path") or "").strip()
                if source_path not in source_keys:
                    kept_faces.append(row)
                    continue
                face_id = str(row.get("face_id") or "").strip()
                if face_id:
                    removed_face_ids.add(face_id)
                crop_rel = str(row.get("crop_path") or "").strip()
                if crop_rel:
                    crop_paths.append(Path(crop_rel))
                removed_faces += 1

            kept_reviews: list[dict[str, Any]] = []
            for row in review_rows:
                if not isinstance(row, dict):
                    continue
                face_id = str(row.get("face_id") or "").strip()
                if face_id in removed_face_ids:
                    removed_reviews += 1
                    continue
                kept_reviews.append(row)

            self._write_all_faces(kept_faces)
            self._write_json(self.review_path, kept_reviews)

            if remove_crops:
                removed_crops = self._remove_crop_paths(crop_paths)

        return {
            "removed_faces": int(removed_faces),
            "removed_reviews": int(removed_reviews),
            "removed_crops": int(removed_crops),
        }

    def store_signature(self) -> str:
        with self._lock:
            parts: list[str] = []
            for path in (self.people_path, self.faces_path, self.review_path):
                try:
                    stat = path.stat()
                    parts.append(f"{path.name}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}")
                except FileNotFoundError:
                    parts.append(f"{path.name}:missing")
            return "|".join(parts)

    def reviewed_identity_signature(self) -> str:
        with self._lock:
            people_rows = self.list_people()
            face_rows = self.list_faces()

        people_payload = [
            {
                "person_id": str(row.get("person_id") or "").strip(),
                "display_name": str(row.get("display_name") or "").strip(),
                "aliases": [str(item or "").strip() for item in list(row.get("aliases") or []) if str(item or "").strip()],
                "notes": str(row.get("notes") or "").strip(),
                "updated_at": str(row.get("updated_at") or "").strip(),
            }
            for row in people_rows
            if str(row.get("person_id") or "").strip()
        ]
        people_payload.sort(key=lambda row: row["person_id"])

        reviewed_face_payload: list[dict[str, Any]] = []
        for row in face_rows:
            face_id = str(row.get("face_id") or "").strip()
            review_status = face_review_status(row)
            if not face_id or review_status not in FACE_REVIEW_STATUSES:
                continue
            reviewed_face_payload.append(
                {
                    "face_id": face_id,
                    "person_id": str(row.get("person_id") or "").strip(),
                    "review_status": review_status,
                    "reviewed_by_human": bool(face_is_human_reviewed(row)),
                    "reviewed_at": str(row.get("reviewed_at") or "").strip(),
                    "updated_at": str(row.get("updated_at") or "").strip(),
                }
            )
        reviewed_face_payload.sort(key=lambda row: row["face_id"])

        payload = {
            "people": people_payload,
            "reviewed_faces": reviewed_face_payload,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def reset_pending_unknown(self, *, remove_crops: bool = True) -> dict[str, int]:
        removed_faces = 0
        removed_reviews = 0
        removed_crops = 0

        with self._lock:
            face_rows = self._read_all_faces()
            if not isinstance(face_rows, list):
                face_rows = []
            review_rows = self._read_json(self.review_path)
            if not isinstance(review_rows, list):
                review_rows = []

            unknown_face_ids: set[str] = set()
            crop_paths: list[Path] = []
            kept_faces: list[dict[str, Any]] = []
            for row in face_rows:
                if not isinstance(row, dict):
                    continue
                face_id = str(row.get("face_id") or "").strip()
                person_id = str(row.get("person_id") or "").strip()
                review_status = face_review_status(row)
                if person_id or review_status in {"ignored", "rejected"}:
                    kept_faces.append(row)
                    continue
                if face_id:
                    unknown_face_ids.add(face_id)
                crop_rel = str(row.get("crop_path") or "").strip()
                if crop_rel:
                    crop_paths.append(Path(crop_rel))
                removed_faces += 1

            kept_reviews: list[dict[str, Any]] = []
            for row in review_rows:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status") or "").strip().lower()
                face_id = str(row.get("face_id") or "").strip()
                if status == "pending" or face_id in unknown_face_ids:
                    removed_reviews += 1
                    continue
                kept_reviews.append(row)

            self._write_all_faces(kept_faces)
            self._write_json(self.review_path, kept_reviews)

            if remove_crops:
                removed_crops = self._remove_crop_paths(crop_paths)

        return {
            "removed_faces": int(removed_faces),
            "removed_reviews": int(removed_reviews),
            "removed_crops": int(removed_crops),
            "kept_faces": int(len(self.list_faces())),
            "kept_reviews": int(len(self.list_review_items())),
        }
