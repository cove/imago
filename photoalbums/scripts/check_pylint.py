#!/usr/bin/env python
"""Run pylint duplicate-code check on photoalbums package."""

import sys

if len(sys.argv) == 1:
    sys.argv += ["photoalbums"]

from pylint import lint  # noqa: E402

lint.Run(sys.argv[1:])