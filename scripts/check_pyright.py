#!/usr/bin/env python
"""Run pyright type checker (uses pyrightconfig.json at repo root)."""

import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "pyright"] + sys.argv[1:], check=False)
sys.exit(result.returncode)
