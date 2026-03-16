#!/usr/bin/env python
"""Run pylint duplicate-code check. Defaults to scanning all three sub-packages."""
import sys

if len(sys.argv) == 1:
    sys.argv += ["photoalbums", "vhs", "cast"]

from pylint import lint  # noqa: E402

lint.Run(sys.argv[1:])
