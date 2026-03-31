from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

PROCESSING_LOCK_SUFFIX = ".photoalbums-ai.lock"
BATCH_LOCK_SUFFIX = ".photoalbums-ai.batch.lock"
JOB_ID_ENV = "IMAGO_JOB_ID"


def _processing_lock_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.name}{PROCESSING_LOCK_SUFFIX}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_processing_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _release_image_processing_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()


def _release_batch_processing_lock(lock_path: Path | None) -> None:
    _release_image_processing_lock(lock_path)


def _clear_stale_processing_lock(lock_path: Path) -> bool:
    payload = _read_processing_lock(lock_path)
    pid = payload.get("pid")
    if isinstance(pid, int) and pid > 0 and not _pid_alive(pid):
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()
        return True
    return False


def _acquire_image_processing_lock(image_path: Path) -> Path:
    lock_path = _processing_lock_path(image_path)
    payload = {
        "image_path": str(image_path.resolve()),
        "pid": os.getpid(),
        "job_id": str(os.environ.get(JOB_ID_ENV) or "").strip(),
    }
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _clear_stale_processing_lock(lock_path):
                continue
            current = _read_processing_lock(lock_path)
            owner_parts = []
            job_id = str(current.get("job_id") or "").strip()
            if job_id:
                owner_parts.append(f"job {job_id}")
            pid = current.get("pid")
            if isinstance(pid, int):
                owner_parts.append(f"pid {pid}")
            owner = ", ".join(owner_parts) if owner_parts else str(lock_path)
            raise RuntimeError(f"already processing {image_path.name} ({owner})")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return lock_path
    raise RuntimeError(f"could not acquire processing lock for {image_path.name}")


def _batch_processing_lock_path(photos_root: Path) -> Path:
    return photos_root / BATCH_LOCK_SUFFIX


def _acquire_batch_processing_lock(photos_root: Path) -> Path:
    lock_path = _batch_processing_lock_path(photos_root)
    payload = {
        "photos_root": str(photos_root.resolve()),
        "pid": os.getpid(),
        "job_id": str(os.environ.get(JOB_ID_ENV) or "").strip(),
    }
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _clear_stale_processing_lock(lock_path):
                continue
            current = _read_processing_lock(lock_path)
            owner_parts = []
            current_root = str(current.get("photos_root") or "").strip()
            if current_root:
                owner_parts.append(current_root)
            job_id = str(current.get("job_id") or "").strip()
            if job_id:
                owner_parts.append(f"job {job_id}")
            pid = current.get("pid")
            if isinstance(pid, int):
                owner_parts.append(f"pid {pid}")
            owner = ", ".join(owner_parts) if owner_parts else str(lock_path)
            raise RuntimeError(f"another photoalbums ai batch run is already active ({owner})")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return lock_path
    raise RuntimeError("could not acquire photoalbums ai batch lock")
