#!/usr/bin/env python
"""Run Ruff lint checks on all sub-packages."""

from __future__ import annotations

import subprocess
import sys


paths = sys.argv[1:] or ["photoalbums", "vhs", "cast"]
result = subprocess.run([sys.executable, "-m", "ruff", "check", *paths], check=False)
sys.exit(result.returncode)
