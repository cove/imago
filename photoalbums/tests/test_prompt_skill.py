import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import _prompt_skill


class TestPromptSkill(unittest.TestCase):
    def test_base_skill_reloads_when_skill_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_path = Path(tmp) / "SKILL.md"
            skill_path.write_text("## Preamble Location\n- First value\n", encoding="utf-8")
            _prompt_skill._cached_base_skill.cache_clear()

            with mock.patch.object(_prompt_skill, "BASE_SKILL_FILE", skill_path):
                first = _prompt_skill.base_skill()
                self.assertEqual(first["Preamble Location"], ["- First value"])

                time.sleep(0.01)
                skill_path.write_text("## Preamble Location\n- Second value\n", encoding="utf-8")

                second = _prompt_skill.base_skill()
                self.assertEqual(second["Preamble Location"], ["- Second value"])


if __name__ == "__main__":
    unittest.main()
