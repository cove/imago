import sys
import threading
from pathlib import Path

_VHS_ROOT = str(Path(__file__).parents[1].resolve())
_VHS_TEST = str(Path(__file__).parent.resolve())

# Add vhs/ to sys.path so tests can import vhs_pipeline, common, libs, apps, etc.
if _VHS_ROOT not in sys.path:
    sys.path.insert(0, _VHS_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "playwright: mark test as requiring a live browser (Playwright)")


try:
    import pytest
    from http.server import ThreadingHTTPServer
    from apps.plain_html_wizard.server import WizardHandler

    @pytest.fixture(scope="session")
    def live_server():
        server = ThreadingHTTPServer(("127.0.0.1", 0), WizardHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}"
        server.shutdown()

except Exception:
    pass


def pytest_collect_file(parent, file_path):
    """
    When running from the monorepo root, photoalbums/common.py may get cached in
    sys.modules as 'common' before vhs tests are collected. This hook fires
    immediately before each vhs test file is imported, ensuring vhs/common.py
    takes precedence.
    """
    if not str(file_path).startswith(_VHS_TEST):
        return None
    if _VHS_ROOT in sys.path:
        sys.path.remove(_VHS_ROOT)
    sys.path.insert(0, _VHS_ROOT)
    _cached = sys.modules.get("common")
    if _cached is not None:
        _cached_file = getattr(_cached, "__file__", "") or ""
        if not _cached_file.startswith(_VHS_ROOT):
            del sys.modules["common"]
    return None
