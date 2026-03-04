import os
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    from stitching import AffineStitcher
except Exception:
    AffineStitcher = None

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:
    Observer = None

    class FileSystemEventHandler:
        pass

from common import (
    FILENAME_PATTERN,
    INCOMING_NAME,
    PAGE_SCAN_RE,
    PHOTO_ALBUMS_DIR,
    configure_imagemagick,
    derive_prefix,
    get_next_filename,
    list_page_scans_for_page,
    open_image_fullscreen,
    process_tiff_in_place,
    rename_with_retry,
)

WATCH_ROOT = str(PHOTO_ALBUMS_DIR)

COLOR_GREEN = "\x1b[32m"
COLOR_RED = "\x1b[31m"
COLOR_YELLOW = "\x1b[33m"
COLOR_RESET = "\x1b[0m"

if sys.platform.startswith("win"):
    try:
        import winsound
    except Exception:
        winsound = None
else:
    winsound = None


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _colorize(text: str, color: str) -> str:
    if not _supports_color():
        return text
    return f"{color}{text}{COLOR_RESET}"


def log_ok(message: str) -> None:
    print(_colorize(message, COLOR_GREEN))


def log_warn(message: str) -> None:
    print(_colorize(message, COLOR_YELLOW))


def log_error(message: str) -> None:
    print(_colorize(message, COLOR_RED))


def log_info(message: str) -> None:
    print(message)


def alert_beep() -> None:
    if winsound is not None:
        try:
            winsound.MessageBeep(winsound.MB_ICONHAND)
            winsound.Beep(1000, 300)
        except Exception:
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)

    if sys.platform.startswith("win"):
        def _popup():
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(
                    0,
                    "STITCH FAILED",
                    "Photo Albums",
                    0x00000010,
                )
            except Exception:
                pass

        threading.Thread(target=_popup, daemon=True).start()


def _require_stitcher() -> None:
    if AffineStitcher is None:
        raise RuntimeError("stitching package is required to validate stitches.")


def validate_stitch(files) -> tuple[bool, Path | None]:
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
                preview_path = save_stitch_preview(result)
                return True, preview_path
        except Exception:
            continue

    return False, None


def _cleanup_temp_file(path: Path, attempts=60, delay=2.0, initial_delay=15.0) -> None:
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


def cleanup_preview_file(path: Path, viewer_process=None) -> None:
    def _wait_and_cleanup():
        if viewer_process is not None:
            try:
                viewer_process.wait()
            except Exception:
                pass
            _cleanup_temp_file(path, initial_delay=0.0)
        else:
            _cleanup_temp_file(path)

    threading.Thread(target=_wait_and_cleanup, daemon=True).start()


class IncomingScanHandler(FileSystemEventHandler):
    def __init__(
        self,
        *,
        get_next_filename_fn=get_next_filename,
        list_page_scans_fn=list_page_scans_for_page,
        derive_prefix_fn=derive_prefix,
        rename_fn=rename_with_retry,
        process_tiff_fn=process_tiff_in_place,
        validate_stitch_fn=validate_stitch,
        open_image_fn=open_image_fullscreen,
        sleep_fn=time.sleep,
        log_ok_fn=log_ok,
        log_error_fn=log_error,
        alert_fn=alert_beep,
        incoming_name=INCOMING_NAME,
    ):
        super().__init__()
        self.retry_pages: dict[str, int] = {}
        self.get_next_filename_fn = get_next_filename_fn
        self.list_page_scans_fn = list_page_scans_fn
        self.derive_prefix_fn = derive_prefix_fn
        self.rename_fn = rename_fn
        self.process_tiff_fn = process_tiff_fn
        self.validate_stitch_fn = validate_stitch_fn
        self.open_image_fn = open_image_fn
        self.sleep_fn = sleep_fn
        self.log_ok_fn = log_ok_fn
        self.log_error_fn = log_error_fn
        self.alert_fn = alert_fn
        self.incoming_name = incoming_name

    def _get_retry_filename(self, watch_dir: str, page_num: int) -> str:
        files = self.list_page_scans_fn(watch_dir, page_num)
        if files:
            last = files[-1]
            m = PAGE_SCAN_RE.search(last)
            scan = (int(m.group("scan")) if m else 1) + 1
        else:
            scan = 1
        prefix = self.derive_prefix_fn(watch_dir)
        return f"{prefix}_P{page_num:02d}_S{scan:02d}.tif"

    def on_created(self, event):
        if event.is_directory:
            return
        if os.path.basename(event.src_path).lower() != self.incoming_name.lower():
            return

        self.sleep_fn(2.0)

        watch_dir = os.path.dirname(event.src_path)
        retry_page = self.retry_pages.get(watch_dir)
        if retry_page is not None:
            new_name = self._get_retry_filename(watch_dir, retry_page)
        else:
            new_name = self.get_next_filename_fn(watch_dir, FILENAME_PATTERN)

        old_path = os.path.join(watch_dir, self.incoming_name)
        new_path = os.path.join(watch_dir, new_name)

        if not self.rename_fn(old_path, new_path, log_error=self.log_error_fn):
            return

        if not self.process_tiff_fn(Path(new_path), log_error=self.log_error_fn):
            self.log_error_fn(f"{Path(new_name).name} ERROR")
            return

        page_match = PAGE_SCAN_RE.search(new_name)
        if not page_match:
            self.log_ok_fn(f"{Path(new_name).name} OK")
            return

        page_num = int(page_match.group("page"))
        files = self.list_page_scans_fn(watch_dir, page_num)
        if len(files) >= 2:
            success, preview_path = self.validate_stitch_fn(files)
            if success:
                if retry_page == page_num:
                    self.retry_pages.pop(watch_dir, None)
                if preview_path is not None:
                    viewer = self.open_image_fn(str(preview_path))
                    cleanup_preview_file(preview_path, viewer)
                self.log_ok_fn(f"{Path(new_name).name} OK")
            else:
                self.retry_pages[watch_dir] = page_num
                self.alert_fn()
                self.log_error_fn(f"{Path(new_name).name} STITCH FAILED")
        else:
            self.log_ok_fn(f"{Path(new_name).name} OK")


def main() -> None:
    if Observer is None:
        raise RuntimeError("watchdog is required to run this script.")

    configure_imagemagick()
    log_info(f"Watching for {INCOMING_NAME} in:")
    log_info(WATCH_ROOT)

    event_handler = IncomingScanHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_ROOT, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
