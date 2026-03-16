#!/usr/bin/env python
"""Run radon cyclomatic complexity check on all sub-packages (grade C and above)."""

import subprocess
import sys

dirs = sys.argv[1:] or ["photoalbums", "vhs", "cast"]
result = subprocess.run(
    [sys.executable, "-m", "radon", "cc", "--min", "C", "--show-complexity"] + dirs,
    check=False,
)
sys.exit(result.returncode)
