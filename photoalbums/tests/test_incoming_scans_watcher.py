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
    def test_on_created_success_no_stitch(self):
        import incoming_scans_watcher

        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp) / "Album_Archive"
            watch_dir.mkdir()
            event = DummyEvent(str(watch_dir / "incoming_scan.tif"))

            rename_mock = mock.Mock(return_value=True)
            process_mock = mock.Mock(return_value=True)
            validate_mock = mock.Mock(return_value=(True, None))
            list_mock = mock.Mock(return_value=[str(watch_dir / "Album_P02_S01.tif")])
            log_ok = mock.Mock()
            log_error = mock.Mock()

            handler = incoming_scans_watcher.IncomingScanHandler(
                get_next_filename_fn=lambda *_: "Album_P02_S01.tif",
                list_page_scans_fn=list_mock,
                rename_fn=rename_mock,
                process_tiff_fn=process_mock,
                validate_stitch_fn=validate_mock,
                open_image_fn=mock.Mock(),
                sleep_fn=lambda *_: None,
                log_ok_fn=log_ok,
                log_error_fn=log_error,
                alert_fn=mock.Mock(),
            )

            handler.on_created(event)

            rename_mock.assert_called_once()
            process_mock.assert_called_once()
            validate_mock.assert_not_called()
            log_ok.assert_called_once()
            log_error.assert_not_called()

    def test_on_created_stitch_fail_sets_retry(self):
        import incoming_scans_watcher

        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp) / "Album_Archive"
            watch_dir.mkdir()
            event = DummyEvent(str(watch_dir / "incoming_scan.tif"))

            list_mock = mock.Mock(
                return_value=[
                    str(watch_dir / "Album_P02_S01.tif"),
                    str(watch_dir / "Album_P02_S02.tif"),
                ]
            )
            alert_mock = mock.Mock()
            log_error = mock.Mock()

            handler = incoming_scans_watcher.IncomingScanHandler(
                get_next_filename_fn=lambda *_: "Album_P02_S02.tif",
                list_page_scans_fn=list_mock,
                rename_fn=mock.Mock(return_value=True),
                process_tiff_fn=mock.Mock(return_value=True),
                validate_stitch_fn=mock.Mock(return_value=(False, None)),
                open_image_fn=mock.Mock(),
                sleep_fn=lambda *_: None,
                log_ok_fn=mock.Mock(),
                log_error_fn=log_error,
                alert_fn=alert_mock,
            )

            handler.on_created(event)

            alert_mock.assert_called_once()
            log_error.assert_called()
            self.assertIn(str(watch_dir), handler.retry_pages)

    def test_on_created_rename_fails(self):
        import incoming_scans_watcher

        with tempfile.TemporaryDirectory() as tmp:
            watch_dir = Path(tmp)
            event = DummyEvent(str(watch_dir / "incoming_scan.tif"))

            process_mock = mock.Mock(return_value=True)
            handler = incoming_scans_watcher.IncomingScanHandler(
                get_next_filename_fn=lambda *_: "Album_P02_S01.tif",
                rename_fn=mock.Mock(return_value=False),
                process_tiff_fn=process_mock,
                sleep_fn=lambda *_: None,
            )

            handler.on_created(event)

            process_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
