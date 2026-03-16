#!/usr/bin/env python
"""Run pylint duplicate-code check on vhs package."""

import sys

if len(sys.argv) == 1:
    sys.argv += ["vhs"]

from pylint import lint  # noqa: E402

lint.Run(sys.argv[1:])