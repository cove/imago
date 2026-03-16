#!/usr/bin/env python
"""Run skylos dead-code finder on all sub-packages."""

import subprocess
import sys
from pathlib import Path

dirs = sys.argv[1:] or ["photoalbums", "vhs", "cast"]
skylos = Path(sys.executable).with_name(
    "skylos.exe" if sys.platform == "win32" else "skylos"
)
result = subprocess.run([str(skylos), "--no-upload"] + dirs, check=False)
sys.exit(result.returncode)
