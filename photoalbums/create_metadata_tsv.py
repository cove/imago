#!/usr/bin/env python3
"""Deprecated metadata TSV exporter."""

from __future__ import annotations

import sys

DEPRECATION_MESSAGE = (
    "photoalbums metadata tsv is deprecated and unsupported because this repo no longer uses metadata.tsv."
)


def main() -> int:
    print(DEPRECATION_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
