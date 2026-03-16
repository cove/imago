#!/usr/bin/env python
"""Run pylint duplicate-code check on viewer package."""

import sys

if len(sys.argv) == 1:
    sys.argv += ["viewer"]

from pylint import lint  # noqa: E402

lint.Run(sys.argv[1:])