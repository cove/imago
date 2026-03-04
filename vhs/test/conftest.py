import sys
from pathlib import Path

# Add vhs/ to sys.path so tests can import vhs_pipeline, common, libs, apps, etc.
sys.path.insert(0, str(Path(__file__).parents[1]))
