#!/usr/bin/env python
"""Run vulture with a raised recursion limit to handle deeply nested files."""

import sys

sys.setrecursionlimit(5000)

from vulture.core import main  # noqa: E402

if len(sys.argv) == 1:
    sys.argv += ["photoalbums", "vhs", "cast", "--min-confidence", "80"]

main()
