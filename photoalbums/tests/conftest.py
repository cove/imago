import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from photoalbums.lib import ai_index


class _WritableTemporaryDirectory:
    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False):
        root = Path(dir or tempfile.gettempdir())
        name = f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
        self.name = str(root / name)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        Path(self.name).mkdir(parents=True, exist_ok=False)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)


@pytest.fixture(autouse=True)
def use_writable_temporary_directory(monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr(tempfile, "TemporaryDirectory", _WritableTemporaryDirectory)


@pytest.fixture(autouse=True)
def isolate_photoalbums_ai_index_defaults(monkeypatch):
    cast_store = Path(os.environ.get("TEMP", ".")) / "imago-test-cast-store" / uuid.uuid4().hex
    cast_store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ai_index, "DEFAULT_CAST_STORE", cast_store)
