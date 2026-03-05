from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _json_default(path: Path) -> Any:
    if path.name == "people.json":
        return {"people": []}
    return []


class TextFaceStore:
    """Small file-backed store for people, faces, and review queue."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.people_path = self.root_dir / "people.json"
        self.faces_path = self.root_dir / "faces.jsonl"
        self.review_path = self.root_dir / "review_queue.jsonl"
        self._lock = threading.RLock()

    def ensure_files(self) -> None:
        with self._lock:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_json_file(self.people_path)
            self._ensure_json_file(self.faces_path)
            self._ensure_json_file(self.review_path)

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
        temp_path.replace(path)

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
            rows = self._read_json(self.faces_path)
            if not isinstance(rows, list):
                return []
            return [dict(row) for row in rows if isinstance(row, dict)]

    def get_face(self, face_id: str) -> dict[str, Any] | None:
        for face in self.list_faces():
            if str(face.get("face_id")) == str(face_id):
                return face
        return None

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
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            rows = self._read_json(self.faces_path)
            if not isinstance(rows, list):
                rows = []
            rows.append(row)
            self._write_json(self.faces_path, rows)
        return dict(row)

    def update_face(self, face_id: str, **fields: Any) -> dict[str, Any]:
        face_key = str(face_id or "").strip()
        if not face_key:
            raise ValueError("face_id is required.")
        now = utc_now_iso()
        with self._lock:
            rows = self._read_json(self.faces_path)
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
                    row[key] = value
                row["updated_at"] = now
                self._write_json(self.faces_path, rows)
                return dict(row)
        raise ValueError(f"Unknown face_id: {face_key}")

    def assign_face(self, face_id: str, person_id: str | None) -> dict[str, Any]:
        face_key = str(face_id or "").strip()
        if not face_key:
            raise ValueError("face_id is required.")
        person_key = str(person_id or "").strip() or None
        now = utc_now_iso()
        with self._lock:
            rows = self._read_json(self.faces_path)
            if not isinstance(rows, list):
                rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("face_id")) != face_key:
                    continue
                row["person_id"] = person_key
                row["updated_at"] = now
                self._write_json(self.faces_path, rows)
                return dict(row)
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
            "suggested_score": float(suggested_score) if suggested_score is not None else None,
            "status": str(status or "pending").strip().lower(),
            "decided_person_id": None,
            "decided_at": "",
            "created_at": now,
            "updated_at": now,
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
        if clean_status not in {"accepted", "rejected", "skipped"}:
            raise ValueError("status must be one of: accepted, rejected, skipped")
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

    def reset_pending_unknown(self, *, remove_crops: bool = True) -> dict[str, int]:
        removed_faces = 0
        removed_reviews = 0
        removed_crops = 0

        with self._lock:
            face_rows = self._read_json(self.faces_path)
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
                if person_id:
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

            self._write_json(self.faces_path, kept_faces)
            self._write_json(self.review_path, kept_reviews)

            if remove_crops:
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

        return {
            "removed_faces": int(removed_faces),
            "removed_reviews": int(removed_reviews),
            "removed_crops": int(removed_crops),
            "kept_faces": int(len(self.list_faces())),
            "kept_reviews": int(len(self.list_review_items())),
        }
