from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


def display_inline_image(
    path: str | Path,
    *,
    title: str = "",
    log_error: Callable[[str], None] | None = None,
    max_width: int = 120,
    max_height: int = 36,
) -> bool:
    chafa = shutil.which("chafa")
    if chafa is None:
        return False

    image_path = Path(path)
    size = shutil.get_terminal_size(fallback=(100, 40))
    width = max(20, min(size.columns, max_width))
    height = max(8, min(size.lines - 4, max_height))

    args = [
        chafa,
        "--animate=off",
        f"--size={width}x{height}",
    ]
    if sys.platform.startswith("win") and os.environ.get("WT_SESSION"):
        args.append("--format=sixels")
    args.append(str(image_path))

    try:
        if title:
            print(title, flush=True)
        subprocess.run(args, check=True)
        print(flush=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        if log_error is not None:
            log_error(f"Inline image display failed for {image_path}: {exc}")
        return False
