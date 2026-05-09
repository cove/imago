#!/usr/bin/env python
"""Run basedpyright type checker (uses basedpyrightconfig.json at repo root)."""

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "basedpyright", "--project", "basedpyrightconfig.json"] + sys.argv[1:],
    check=False,
)
sys.exit(result.returncode)
