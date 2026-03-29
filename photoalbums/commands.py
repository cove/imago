from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def _call_main(func) -> int:
    try:
        result = func()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return int(code)
        print(code)
        return 1
    if isinstance(result, int):
        return int(result)
    return 0


def run_ai_index(argv: list[str]) -> int:
    from .lib import ai_index

    return int(ai_index.run(argv) or 0)


def run_apply_metadata() -> int:
    import apply_metadata

    return _call_main(apply_metadata.main)


def run_create_metadata_tsv() -> int:
    import create_metadata_tsv

    return _call_main(create_metadata_tsv.main)


def run_compress_tiff() -> int:
    import compress_tiff

    return _call_main(compress_tiff.main)


def run_render() -> int:
    import stitch_oversized_pages

    return _call_main(stitch_oversized_pages.main)


def run_stitch_validate() -> int:
    import stitch_oversized_pages_validate

    return _call_main(stitch_oversized_pages_validate.main)


def run_watch_incoming() -> int:
    import incoming_scans_watcher

    return _call_main(incoming_scans_watcher.main)


def run_checksum_tree(*, base_dir: str, verify: bool) -> int:
    import sha3_tree_hashes

    argv = [str(base_dir)]
    if verify:
        argv.append("--verify")
    return int(sha3_tree_hashes.run(argv) or 0)


if __name__ == "__main__":
    raise SystemExit("Internal module. Run: uv run python photoalbums.py ...")
