from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from stitching import AffineStitcher
except Exception:
    AffineStitcher = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _require_stitcher() -> None:
    if AffineStitcher is None:
        raise RuntimeError("stitching package is required to validate stitches.")


def alert_beep() -> None:
    if sys.platform.startswith("win"):
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONHAND)
            winsound.Beep(1000, 300)
        except Exception:
            print("\a", end="", flush=True)

        try:
            import ctypes

            threading.Thread(
                target=lambda: ctypes.windll.user32.MessageBoxW(
                    0,
                    "Scan stitch failed. Scan another copy of the same page.",
                    "Photo Albums",
                    0x00000010,
                ),
                daemon=True,
            ).start()
        except Exception:
            pass
        return

    print("\a", end="", flush=True)


def save_stitch_preview(panorama) -> Path | None:
    try:
        import cv2
    except Exception:
        return None

    fd, temp_name = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        if not cv2.imwrite(str(temp_path), panorama):
            temp_path.unlink(missing_ok=True)
            return None
    except Exception:
        temp_path.unlink(missing_ok=True)
        return None

    return temp_path


def _cleanup_temp_file(path: Path, attempts: int = 60, delay: float = 2.0, initial_delay: float = 15.0) -> None:
    if initial_delay:
        time.sleep(initial_delay)
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)
        except Exception:
            return


def cleanup_preview_file(path: Path, viewer_process=None) -> None:
    def _wait_and_cleanup() -> None:
        if viewer_process is not None:
            try:
                viewer_process.wait()
            except Exception:
                pass
            _cleanup_temp_file(path, initial_delay=0.0)
        else:
            _cleanup_temp_file(path)

    threading.Thread(target=_wait_and_cleanup, daemon=True).start()


def validate_stitch(files: list[str], *, save_preview: bool = True) -> tuple[bool, Path | None]:
    _require_stitcher()
    if len(files) < 2:
        return True, None

    attempts = [
        {"detector": "sift", "confidence_threshold": 0.3},
        {"detector": "brisk", "confidence_threshold": 0.1},
    ]

    for cfg in attempts:
        try:
            result = AffineStitcher(**cfg).stitch(files)
            if result is not None and getattr(result, "size", 0):
                if save_preview:
                    return True, save_stitch_preview(result)
                return True, None
        except Exception:
            continue

    return False, None


@dataclass(slots=True)
class ScanEvent:
    id: str
    archive_dir: str
    incoming_path: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    status: str = "pending"
    target_name: str = ""
    page_num: int | None = None
    stitch_validated: bool | None = None
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "archive_dir": self.archive_dir,
            "incoming_path": self.incoming_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "target_name": self.target_name,
            "page_num": self.page_num,
            "stitch_validated": self.stitch_validated,
            "note": self.note,
        }


@dataclass(slots=True)
class ArchiveState:
    archive_dir: str
    incoming_path: str = ""
    pending_event_ids: list[str] = field(default_factory=list)
    page_scan_counts: dict[int, int] = field(default_factory=dict)
    needs_rescan_pages: set[int] = field(default_factory=set)

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_dir": self.archive_dir,
            "incoming_path": self.incoming_path,
            "pending_event_ids": list(self.pending_event_ids),
            "page_scan_counts": {str(page): count for page, count in sorted(self.page_scan_counts.items())},
            "needs_rescan_pages": sorted(self.needs_rescan_pages),
        }
