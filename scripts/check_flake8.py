#!/usr/bin/env python
"""Run flake8 (style, unused imports, McCabe complexity) on all sub-packages."""

import sys

if len(sys.argv) == 1:
    sys.argv += ["photoalbums", "vhs", "cast"]

from flake8.main.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
