import sys
import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanwatch  # noqa: E402


class TtyBuffer:
    def __init__(self):
        self.text = ""

    def write(self, value):
        self.text += value

    def flush(self):
        pass

    def isatty(self):
        return True


class TestScanWatch(unittest.TestCase):
    def test_transient_status_writes_and_clears_tty_line(self):
        stream = TtyBuffer()

        with scanwatch._TransientStatus("Working ...", stream=stream):
            self.assertIn("| Working ...", stream.text)

        self.assertTrue(stream.text.endswith("\r"))
        self.assertIn("\r" + (" " * len("| Working ...")) + "\r", stream.text)

    def test_apply_decision_marks_rescan(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            (archive_dir / "Album_P02_S01.tif").touch()
            incoming = archive_dir / "incoming_scan.tif"
            incoming.touch()

            process_mock = mock.Mock(return_value=True)
            validate_mock = mock.Mock(return_value=(False, None))
            alert_mock = mock.Mock()
            error_mock = mock.Mock()

            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                process_tiff_fn=process_mock,
                validate_stitch_fn=validate_mock,
                open_image_fn=mock.Mock(),
                display_image_fn=mock.Mock(return_value=False),
                alert_fn=alert_mock,
                log_error_fn=error_mock,
                sleep_fn=lambda *_: None,
            )

            event = service.register_incoming(incoming)
            result = service.apply_decision(event.id, "Album_P02_S02.tif")

            self.assertEqual(result["event"]["status"], "needs_rescan")
            self.assertIn(2, result["archive"]["needs_rescan_pages"])
            process_mock.assert_called_once()
            self.assertEqual(validate_mock.call_count, 2)
            alert_mock.assert_called_once()
            error_mock.assert_called()

    def test_rebuild_recovers_pending_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            incoming = archive_dir / "incoming_scan.tif"
            incoming.touch()
            (archive_dir / "Album_P02_S01.tif").touch()
            (archive_dir / "Album_P02_S02.tif").touch()

            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                validate_stitch_fn=lambda *_args, **_kwargs: (True, None),
                display_image_fn=mock.Mock(return_value=False),
                sleep_fn=lambda *_: None,
            )

            status = service.rebuild()

            self.assertEqual(status["pending_event_count"], 1)
            self.assertEqual(len(service.list_events()), 1)
            self.assertEqual(service.list_rescans(), [])

    def test_stop_uses_bounded_observer_join(self):
        class HangingObserver:
            def __init__(self):
                self.join_timeout = None
                self.stopped = False

            def stop(self):
                self.stopped = True

            def join(self, timeout=None):
                self.join_timeout = timeout

            def is_alive(self):
                return True

        error_mock = mock.Mock()
        service = scanwatch.ScanWatchService(log_error_fn=error_mock)
        observer = HangingObserver()
        service._observer = observer

        status = service.stop(timeout=0.25)

        self.assertFalse(status["running"])
        self.assertTrue(observer.stopped)
        self.assertEqual(observer.join_timeout, 0.25)
        error_mock.assert_called_once_with("Watcher observer did not stop within 0.25 seconds.")

    def test_apply_pending_incoming_scans_uses_numeric_backlog_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            (archive_dir / "incoming_scan0002.tif").write_text("second")
            (archive_dir / "incoming_scan0001.tif").write_text("first")

            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                process_tiff_fn=mock.Mock(return_value=True),
                validate_stitch_fn=lambda *_args, **_kwargs: (True, None),
                open_image_fn=mock.Mock(),
                display_image_fn=mock.Mock(return_value=False),
                sleep_fn=lambda *_: None,
            )

            results = service.apply_pending_incoming_scans(archive_dir, log_info_fn=mock.Mock())

            self.assertEqual(len(results), 2)
            self.assertEqual((archive_dir / "Album_P01_S01.tif").read_text(), "first")
            self.assertEqual((archive_dir / "Album_P02_S01.tif").read_text(), "second")
            self.assertFalse((archive_dir / "incoming_scan0001.tif").exists())
            self.assertFalse((archive_dir / "incoming_scan0002.tif").exists())

    def test_apply_decision_displays_renamed_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            incoming = archive_dir / "incoming_scan.tif"
            incoming.touch()

            display_mock = mock.Mock(return_value=True)
            error_mock = mock.Mock()

            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                process_tiff_fn=mock.Mock(return_value=True),
                validate_stitch_fn=lambda *_args, **_kwargs: (True, None),
                display_image_fn=display_mock,
                log_error_fn=error_mock,
                sleep_fn=lambda *_: None,
            )

            event = service.register_incoming(incoming)
            service.apply_decision(event.id, "Album_P01_S01.tif")

            display_mock.assert_called_once_with(
                archive_dir / "Album_P01_S01.tif",
                title="Renamed scan: Album_P01_S01.tif",
                log_error=error_mock,
            )

    def test_apply_decision_displays_successful_stitch_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            (archive_dir / "Album_P02_S01.tif").touch()
            incoming = archive_dir / "incoming_scan.tif"
            incoming.touch()
            preview = Path(tmp) / "preview.tif"
            preview.touch()

            display_mock = mock.Mock(return_value=True)
            error_mock = mock.Mock()

            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                process_tiff_fn=mock.Mock(return_value=True),
                validate_stitch_fn=mock.Mock(return_value=(True, preview)),
                display_image_fn=display_mock,
                log_error_fn=error_mock,
                sleep_fn=lambda *_: None,
            )

            event = service.register_incoming(incoming)
            service.apply_decision(event.id, "Album_P02_S02.tif")

            display_mock.assert_any_call(
                preview,
                title="Stitched preview: page 02",
                log_error=error_mock,
            )

    def test_stitch_last_scans_uses_last_event_page_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            view_dir = Path(tmp) / "Album_Pages"
            output_path = view_dir / "Album_P12_V.jpg"

            earlier_page_files = [
                archive_dir / "Album_P09_S01.tif",
                archive_dir / "Album_P09_S02.tif",
            ]
            last_page_files = [
                archive_dir / "Album_P12_S01.tif",
                archive_dir / "Album_P12_S02.tif",
                archive_dir / "Album_P12_S03.tif",
                archive_dir / "Album_P12_S04.tif",
            ]
            for path in [*earlier_page_files, *last_page_files]:
                path.touch()
            for path in earlier_page_files:
                os.utime(path, (2000, 2000))
            for path in last_page_files:
                os.utime(path, (1000, 1000))

            display_mock = mock.Mock()
            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                validate_stitch_fn=lambda *_args, **_kwargs: (True, None),
                display_image_fn=display_mock,
                sleep_fn=lambda *_: None,
            )
            service.rebuild()
            service._events["last-event"] = scanwatch.ScanEvent(
                id="last-event",
                archive_dir=str(archive_dir.resolve(strict=False)),
                incoming_path=str(archive_dir / "incoming_scan.tif"),
                status="completed",
                target_name="Album_P12_S04.tif",
                page_num=12,
            )

            with (
                mock.patch("stitch_oversized_pages.get_view_dirname", return_value=str(view_dir)),
                mock.patch("stitch_oversized_pages._view_page_output_path", return_value=output_path),
                mock.patch("stitch_oversized_pages.stitch", return_value=True) as stitch_mock,
            ):
                result = service.stitch_last_scans()

            stitch_mock.assert_called_once_with([str(path) for path in last_page_files], str(view_dir), force=True)
            self.assertEqual(result["page_num"], 12)
            self.assertEqual(result["output_path"], str(output_path))
            display_mock.assert_not_called()

    def test_stitch_last_scans_falls_back_to_newest_mtime_without_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            view_dir = Path(tmp) / "Album_Pages"
            output_path = view_dir / "Album_P09_V.jpg"

            newer_mtime_files = [
                archive_dir / "Album_P09_S01.tif",
                archive_dir / "Album_P09_S02.tif",
            ]
            higher_page_files = [
                archive_dir / "Album_P12_S01.tif",
                archive_dir / "Album_P12_S02.tif",
            ]
            for path in [*newer_mtime_files, *higher_page_files]:
                path.touch()
            for path in newer_mtime_files:
                os.utime(path, (2000, 2000))
            for path in higher_page_files:
                os.utime(path, (1000, 1000))

            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                validate_stitch_fn=lambda *_args, **_kwargs: (True, None),
                display_image_fn=mock.Mock(),
                sleep_fn=lambda *_: None,
            )
            service.rebuild()

            with (
                mock.patch("stitch_oversized_pages.get_view_dirname", return_value=str(view_dir)),
                mock.patch("stitch_oversized_pages._view_page_output_path", return_value=output_path),
                mock.patch("stitch_oversized_pages.stitch", return_value=True) as stitch_mock,
            ):
                result = service.stitch_last_scans()

            stitch_mock.assert_called_once_with([str(path) for path in newer_mtime_files], str(view_dir), force=True)
            self.assertEqual(result["page_num"], 9)
            self.assertEqual(result["output_path"], str(output_path))

    def test_stitch_last_scans_requires_multiple_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp) / "Album_Archive"
            archive_dir.mkdir()
            (archive_dir / "Album_P01_S01.tif").touch()

            service = scanwatch.ScanWatchService(
                root=Path(tmp),
                validate_stitch_fn=lambda *_args, **_kwargs: (True, None),
                display_image_fn=mock.Mock(),
                sleep_fn=lambda *_: None,
            )
            service.rebuild()

            with self.assertRaisesRegex(ValueError, "need at least 2"):
                service.stitch_last_scans()


if __name__ == "__main__":
    unittest.main()
