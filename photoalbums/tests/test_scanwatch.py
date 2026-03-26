import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanwatch


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
                sleep_fn=lambda *_: None,
            )

            status = service.rebuild()

            self.assertEqual(status["pending_event_count"], 1)
            self.assertEqual(len(service.list_events()), 1)
            self.assertEqual(service.list_rescans(), [])


if __name__ == "__main__":
    unittest.main()
