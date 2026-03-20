from pathlib import Path

import pytest

from photoalbums.lib import ai_index


@pytest.fixture(autouse=True)
def isolate_photoalbums_ai_index_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_index, "DEFAULT_CAST_STORE", Path(tmp_path) / "cast_data")
    monkeypatch.setattr(
        ai_index,
        "DEFAULT_MANIFEST_PATH",
        Path(tmp_path) / "ai_index_manifest.jsonl",
    )
