from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODES_ROOT = REPO_ROOT / "modes"
PHOTOALBUMS_MODEL_DIR = MODES_ROOT / "photoalbums"
YOLO_MODEL_DIR = PHOTOALBUMS_MODEL_DIR / "yolo"
HF_MODEL_CACHE_DIR = PHOTOALBUMS_MODEL_DIR / "hf"
