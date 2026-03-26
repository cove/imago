import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyEvent:
    def __init__(self, src_path: str, is_directory: bool = False):
        self.src_path = src_path
        self.is_directory = is_directory


class TestIncomingScansWatcher(unittest.TestCase):
    def test_on_created_registers_pending_event(self):
        import incoming_scans_watcher

        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp) / "Album_Archive"
            watch_dir.mkdir()
            incoming = watch_dir / "incoming_scan.tif"
            incoming.touch()
            event = DummyEvent(str(incoming))

            info_mock = mock.Mock()
            alert_mock = mock.Mock()
            service = incoming_scans_watcher.ScanWatchService(
                root=Path(tmp),
                alert_fn=alert_mock,
                sleep_fn=lambda *_: None,
            )
            handler = incoming_scans_watcher.IncomingScanHandler(
                service,
                sleep_fn=lambda *_: None,
                log_info_fn=info_mock,
            )

            handler.on_created(event)

            status = service.status()
            self.assertEqual(status["pending_event_count"], 1)
            info_mock.assert_called_once()
            alert_mock.assert_called_once()

    def test_on_created_ignores_other_files(self):
        import incoming_scans_watcher

        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp) / "Album_Archive"
            watch_dir.mkdir()
            other = watch_dir / "notes.txt"
            other.touch()
            event = DummyEvent(str(other))

            alert_mock = mock.Mock()
            service = incoming_scans_watcher.ScanWatchService(
                root=Path(tmp),
                alert_fn=alert_mock,
                sleep_fn=lambda *_: None,
            )
            handler = incoming_scans_watcher.IncomingScanHandler(
                service,
                sleep_fn=lambda *_: None,
                log_info_fn=mock.Mock(),
            )

            handler.on_created(event)

            self.assertEqual(service.status()["pending_event_count"], 0)
            alert_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
