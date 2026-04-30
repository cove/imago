from __future__ import annotations

import time
import uuid
import threading
from pathlib import Path
from typing import Callable

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:
    Observer = None

    class FileSystemEventHandler:
        pass


try:
    from .common import (
        INCOMING_NAME,
        PAGE_SCAN_RE,
        PHOTO_ALBUMS_DIR,
        PHOTO_SCANNING_DIR,
        configure_imagemagick,
        get_next_filename,
        list_page_scans_for_page,
        open_image_fullscreen,
        process_tiff_in_place,
        rename_with_retry,
    )
except ImportError:
    from common import (
        INCOMING_NAME,
        PAGE_SCAN_RE,
        PHOTO_ALBUMS_DIR,
        PHOTO_SCANNING_DIR,
        configure_imagemagick,
        get_next_filename,
        list_page_scans_for_page,
        open_image_fullscreen,
        process_tiff_in_place,
        rename_with_retry,
    )

try:
    from .scanwatch_core import (
        ArchiveState,
        ScanEvent,
        alert_beep,
        cleanup_preview_file,
        save_stitch_preview,
        _normalize_path,
        _now,
        validate_stitch,
    )
except ImportError:
    from scanwatch_core import (
        ArchiveState,
        ScanEvent,
        alert_beep,
        cleanup_preview_file,
        save_stitch_preview,
        _normalize_path,
        _now,
        validate_stitch,
    )


class ScanWatchService:
    def __init__(
        self,
        root: str | Path = PHOTO_SCANNING_DIR,
        *,
        incoming_name: str = INCOMING_NAME,
        sleep_fn: Callable[[float], None] = time.sleep,
        log_info_fn: Callable[[str], None] = print,
        log_error_fn: Callable[[str], None] = print,
        alert_fn: Callable[[], None] = alert_beep,
        rename_fn=rename_with_retry,
        process_tiff_fn=process_tiff_in_place,
        validate_stitch_fn=validate_stitch,
        open_image_fn=open_image_fullscreen,
    ) -> None:
        self.root = _normalize_path(root)
        self.incoming_name = incoming_name
        self.sleep_fn = sleep_fn
        self.log_info_fn = log_info_fn
        self.log_error_fn = log_error_fn
        self.alert_fn = alert_fn
        self.rename_fn = rename_fn
        self.process_tiff_fn = process_tiff_fn
        self.validate_stitch_fn = validate_stitch_fn
        self.open_image_fn = open_image_fn
        self._lock = threading.RLock()
        self._events: dict[str, ScanEvent] = {}
        self._events_by_path: dict[str, str] = {}
        self._archives: dict[str, ArchiveState] = {}
        self._observer: Observer | None = None
        self._handler: IncomingScanHandler | None = None

    def set_root(self, root: str | Path) -> None:
        self.root = _normalize_path(root)

    def status(self) -> dict[str, object]:
        with self._lock:
            pending = sum(1 for event in self._events.values() if event.status in {"pending", "processing"})
            rescans = sum(len(state.needs_rescan_pages) for state in self._archives.values())
            return {
                "root": str(self.root),
                "running": bool(self._observer and self._observer.is_alive()),
                "event_count": len(self._events),
                "pending_event_count": pending,
                "needs_rescan_count": rescans,
                "archive_count": len(self._archives),
            }

    def list_events(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self._lock:
            events = list(self._events.values())
        if status:
            events = [event for event in events if event.status == status]
        events.sort(key=lambda event: (event.created_at, event.id), reverse=True)
        return [event.to_dict() for event in events[:limit]]

    def list_rescans(self, *, limit: int = 100) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        with self._lock:
            archives = list(self._archives.values())
        for archive in sorted(archives, key=lambda item: item.archive_dir):
            for page in sorted(archive.needs_rescan_pages):
                items.append(
                    {
                        "archive_dir": archive.archive_dir,
                        "page_num": page,
                        "scan_count": archive.page_scan_counts.get(page, 0),
                        "files": [str(path) for path in list_page_scans_for_page(archive.archive_dir, page)],
                    }
                )
                if len(items) >= limit:
                    return items
        return items

    def get_event(self, event_id: str) -> dict[str, object]:
        with self._lock:
            event = self._events.get(event_id)
        if event is None:
            raise ValueError(f"Event {event_id} not found")
        return event.to_dict()

    def get_event_context(self, event_id: str) -> dict[str, object]:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                raise ValueError(f"Event {event_id} not found")
            archive = self._archives.get(event.archive_dir)
        context = event.to_dict()
        context["archive"] = archive.to_dict() if archive is not None else {"archive_dir": event.archive_dir}
        context["page_files"] = {}
        if archive is not None:
            for page in sorted(archive.page_scan_counts):
                context["page_files"][str(page)] = [
                    str(path) for path in list_page_scans_for_page(archive.archive_dir, page)
                ]
        return context

    def get_archive_context(self, archive_dir: str | Path) -> dict[str, object]:
        archive_key = str(_normalize_path(archive_dir))
        with self._lock:
            archive = self._archives.get(archive_key)
        if archive is None:
            raise ValueError(f"Archive {archive_key} not found")
        return archive.to_dict()

    def rebuild(self) -> dict[str, object]:
        if not self.root.exists():
            raise ValueError(f"Root not found: {self.root}")

        events: dict[str, ScanEvent] = {}
        events_by_path: dict[str, str] = {}
        archives: dict[str, ArchiveState] = {}

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower() == self.incoming_name.lower():
                archive_dir = str(path.parent.resolve(strict=False))
                event = ScanEvent(
                    id=str(uuid.uuid4())[:8],
                    archive_dir=archive_dir,
                    incoming_path=str(path.resolve(strict=False)),
                )
                events[event.id] = event
                events_by_path[event.incoming_path] = event.id
            if path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            archive_dir = str(path.parent.resolve(strict=False))
            archives.setdefault(archive_dir, ArchiveState(archive_dir=archive_dir))

        for archive_dir in list(archives):
            self._sync_archive_state(archive_dir, archives=archives, validate=False)
            self._sync_pending_events_for_archive(archive_dir, events=events, archives=archives)

        with self._lock:
            self._events = events
            self._events_by_path = events_by_path
            self._archives = archives
            return self.status()

    def refresh_archive(self, archive_dir: str | Path) -> ArchiveState:
        archive_key = str(_normalize_path(archive_dir))
        with self._lock:
            archive = self._archives.get(archive_key)
            if archive is None:
                archive = ArchiveState(archive_dir=archive_key)
                self._archives[archive_key] = archive
            self._sync_archive_state(archive_key)
            self._sync_pending_events_for_archive(archive_key)
            return self._archives[archive_key]

    def start(self) -> dict[str, object]:
        configure_imagemagick()
        self.rebuild()
        if self._observer is not None and self._observer.is_alive():
            return self.status()
        if Observer is None:
            raise RuntimeError("watchdog is required to run this service.")

        handler = IncomingScanHandler(self)
        observer = Observer()
        observer.schedule(handler, str(self.root), recursive=True)
        observer.start()
        self._handler = handler
        self._observer = observer
        return self.status()

    def stop(self) -> dict[str, object]:
        observer = self._observer
        self._observer = None
        self._handler = None
        if observer is None:
            return self.status()
        observer.stop()
        observer.join()
        return self.status()

    def register_incoming(self, incoming_path: str | Path) -> ScanEvent:
        path = _normalize_path(incoming_path)
        if path.name.lower() != self.incoming_name.lower():
            raise ValueError(f"Expected {self.incoming_name}, got {path.name}")
        archive_dir = str(path.parent)

        with self._lock:
            event_id = self._events_by_path.get(str(path))
            if event_id is not None:
                return self._events[event_id]

            event = ScanEvent(
                id=str(uuid.uuid4())[:8],
                archive_dir=archive_dir,
                incoming_path=str(path),
            )
            self._events[event.id] = event
            self._events_by_path[event.incoming_path] = event.id
            archive = self._archives.get(archive_dir)
            if archive is None:
                archive = ArchiveState(archive_dir=archive_dir)
                self._archives[archive_dir] = archive
            archive.incoming_path = event.incoming_path
            archive.pending_event_ids.append(event.id)
            return event

    def apply_decision(
        self,
        event_id: str,
        target_name: str,
        *,
        validate_stitch: bool = True,
        open_preview: bool = True,
    ) -> dict[str, object]:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                raise ValueError(f"Event {event_id} not found")
            if event.status not in {"pending", "needs_rescan"}:
                raise ValueError(f"Event {event_id} is not actionable (status: {event.status})")
            if not target_name or not target_name.lower().endswith((".tif", ".tiff")):
                raise ValueError("target_name must end with .tif or .tiff")
            event.status = "processing"
            event.updated_at = _now()

        archive_dir = Path(event.archive_dir)
        old_path = Path(event.incoming_path)
        new_path = archive_dir / target_name
        if new_path.exists():
            raise ValueError(f"Target already exists: {new_path}")

        if not self.rename_fn(old_path, new_path, log_error=self.log_error_fn):
            with self._lock:
                event.status = "failed"
                event.note = "rename failed"
                event.updated_at = _now()
            return event.to_dict()

        if not self.process_tiff_fn(new_path, log_error=self.log_error_fn):
            with self._lock:
                event.status = "failed"
                event.target_name = target_name
                event.note = "processing failed"
                event.updated_at = _now()
            return event.to_dict()

        with self._lock:
            self._events_by_path.pop(event.incoming_path, None)

        page_num = self._parse_page_num(target_name)
        preview_path: Path | None = None
        stitch_validated: bool | None = None

        if validate_stitch and page_num is not None:
            files = list_page_scans_for_page(archive_dir, page_num)
            if len(files) >= 2:
                stitch_validated, preview_path = self.validate_stitch_fn(files, save_preview=True)
                if stitch_validated and preview_path is not None and open_preview:
                    viewer = self.open_image_fn(str(preview_path))
                    cleanup_preview_file(preview_path, viewer)
                elif preview_path is not None:
                    cleanup_preview_file(preview_path)

        self._sync_archive_state(str(archive_dir))
        self._sync_pending_events_for_archive(str(archive_dir))
        archive = self._archives.get(str(archive_dir))

        with self._lock:
            event.target_name = target_name
            event.page_num = page_num
            event.stitch_validated = stitch_validated
            event.updated_at = _now()

            if page_num is None or stitch_validated is None:
                event.status = "completed"
                event.note = "processed"
                if archive is not None and page_num is not None:
                    archive.needs_rescan_pages.discard(page_num)
            elif stitch_validated:
                event.status = "completed"
                event.note = "stitch validated"
                if archive is not None:
                    archive.needs_rescan_pages.discard(page_num)
            else:
                event.status = "needs_rescan"
                event.note = f"page {page_num:02d} stitch failed; scan another copy of the same page"
                if archive is not None:
                    archive.needs_rescan_pages.add(page_num)
                self.log_error_fn(f"{target_name} STITCH FAILED")
                self.alert_fn()

            if archive is not None and event.id not in archive.pending_event_ids:
                archive.pending_event_ids.append(event.id)

            return {
                "event": event.to_dict(),
                "archive": archive.to_dict() if archive is not None else {"archive_dir": str(archive_dir)},
                "new_path": str(new_path),
            }

    def _sync_archive_state(
        self,
        archive_dir: str,
        *,
        archives: dict[str, ArchiveState] | None = None,
        validate: bool = True,
    ) -> ArchiveState:
        archive_map = archives if archives is not None else self._archives
        archive = archive_map.get(archive_dir)
        if archive is None:
            archive = ArchiveState(archive_dir=archive_dir)
            archive_map[archive_dir] = archive

        archive.page_scan_counts = {}
        archive.needs_rescan_pages = set()

        dir_path = Path(archive_dir)
        for entry in dir_path.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in {".tif", ".tiff"}:
                continue
            match = PAGE_SCAN_RE.search(entry.name)
            if not match:
                continue
            page = int(match.group("page"))
            archive.page_scan_counts[page] = max(archive.page_scan_counts.get(page, 0), int(match.group("scan")))

        if validate:
            for page, count in archive.page_scan_counts.items():
                if count < 2:
                    continue
                files = list_page_scans_for_page(dir_path, page)
                try:
                    stitch_validated, _ = self.validate_stitch_fn(files, save_preview=False)
                except Exception as exc:
                    self.log_error_fn(f"{dir_path.name} page {page:02d} validation failed: {exc}")
                    stitch_validated = False
                if not stitch_validated:
                    archive.needs_rescan_pages.add(page)

        return archive

    def _sync_pending_events_for_archive(
        self,
        archive_dir: str,
        *,
        events: dict[str, ScanEvent] | None = None,
        archives: dict[str, ArchiveState] | None = None,
    ) -> None:
        event_map = events if events is not None else self._events
        archive_map = archives if archives is not None else self._archives
        archive = archive_map.get(archive_dir)
        if archive is None:
            return

        archive.pending_event_ids = [
            event.id
            for event in event_map.values()
            if event.archive_dir == archive_dir and event.status in {"pending", "processing", "needs_rescan"}
        ]
        for event in event_map.values():
            if event.archive_dir == archive_dir and event.status == "pending":
                archive.incoming_path = event.incoming_path
                break

    @staticmethod
    def _parse_page_num(target_name: str) -> int | None:
        match = PAGE_SCAN_RE.search(target_name)
        if match is None:
            return None
        return int(match.group("page"))


class IncomingScanHandler(FileSystemEventHandler):
    def __init__(
        self,
        service: ScanWatchService,
        *,
        sleep_fn: Callable[[float], None] | None = None,
        log_info_fn: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.service = service
        self.sleep_fn = sleep_fn or time.sleep
        self.log_info_fn = log_info_fn or print

    def _handle_incoming(self, path: str) -> None:
        if Path(path).name.lower() != self.service.incoming_name.lower():
            return
        self.sleep_fn(2.0)
        scan_event = self.service.register_incoming(path)
        archive_dir = scan_event.archive_dir
        target_name = get_next_filename(archive_dir)
        self.log_info_fn(f"Auto-applying scan event {scan_event.id} -> {target_name}")
        threading.Thread(
            target=self.service.apply_decision,
            args=(scan_event.id, target_name),
            daemon=True,
        ).start()

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self._handle_incoming(event.src_path)

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        self._handle_incoming(event.dest_path)


def main() -> None:
    service = ScanWatchService(root=PHOTO_ALBUMS_DIR)
    print(f"Starting watcher — scanning {service.root} ...")
    status = service.start()
    print(f"Watching for {service.incoming_name} in:")
    print(status["root"])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()


__all__ = [
    "ArchiveState",
    "IncomingScanHandler",
    "ScanEvent",
    "ScanWatchService",
    "alert_beep",
    "cleanup_preview_file",
    "main",
    "save_stitch_preview",
    "validate_stitch",
]
