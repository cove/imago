import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanwatch  # noqa: E402


class TestScanWatch(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
