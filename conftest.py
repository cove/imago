import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
_VHS = str(_ROOT / "vhs")
_PHOTOALBUMS = str(_ROOT / "photoalbums")

# Pre-import photoalbums/common.py and save the module object.
# This preserves its identity so mock.patch("common.X") and test-level
# "import common" references all point to the same object, even after
# vhs/test/conftest.py evicts it during vhs test collection.
if _PHOTOALBUMS not in sys.path:
    sys.path.insert(0, _PHOTOALBUMS)
import common as _pa_common  # noqa: E402

_common_modules: dict = {_PHOTOALBUMS: _pa_common}


def pytest_configure(config):
    if os.name != "nt":
        return

    user_root = os.environ.get("USERPROFILE")
    username = os.environ.get("USERNAME")
    if user_root:
        user_root_path = Path(user_root)
        temp_root = (
            user_root_path
            if _is_windows_temp_descendant(user_root_path)
            else user_root_path / "AppData" / "Local" / "Temp"
        )
    elif username:
        temp_root = Path("C:/Users") / username / "AppData" / "Local" / "Temp"
    else:
        return

    temp_root.mkdir(parents=True, exist_ok=True)
    temp_path = str(temp_root)
    os.environ["TEMP"] = temp_path
    os.environ["TMP"] = temp_path
    tempfile.tempdir = temp_path


def _is_windows_temp_descendant(user_root: Path) -> bool:
    parts = tuple(part.lower() for part in user_root.parts)
    return parts[-3:] == ("appdata", "local", "temp") or parts[-4:-1] == ("appdata", "local", "temp")


# vhs common is captured lazily (after vhs/test/conftest.py imports it).
def pytest_collect_file(parent, file_path):
    current = sys.modules.get("common")
    if current is not None:
        f = getattr(current, "__file__", "") or ""
        if f.startswith(_VHS) and _VHS not in _common_modules:
            _common_modules[_VHS] = current
    return None


def pytest_runtest_setup(item):
    """Restore the original common module object before each test runs."""
    path = str(item.fspath)
    if _PHOTOALBUMS in path:
        _restore(_PHOTOALBUMS)
    elif _VHS in path:
        _restore(_VHS)


def _restore(project_root: str) -> None:
    mod = _common_modules.get(project_root)
    if mod is None:
        return
    sys.modules["common"] = mod
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)
