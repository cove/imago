import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.prompt_debug import PromptDebugSession


class TestPromptDebug(unittest.TestCase):
    def test_prompt_debug_session_builds_single_image_artifact(self):
        session = PromptDebugSession("C:/photos/Album_A/Photo_01.jpg")
        session.record(
            step="caption",
            engine="lmstudio",
            model="qwen",
            prompt="Describe this image",
            system_prompt="Return JSON",
            source_path="C:/photos/Album_A/Photo_01.jpg",
            prompt_source="skill",
            metadata={"request_photo_regions": False},
        )

        artifact = session.to_artifact()

        self.assertEqual(artifact["kind"], "photoalbums_prompts")
        self.assertEqual(artifact["image_path"], str(Path("C:/photos/Album_A/Photo_01.jpg")))
        self.assertEqual(artifact["step_count"], 1)
        self.assertEqual(artifact["steps"][0]["step"], "caption")
        self.assertEqual(artifact["steps"][0]["prompt"], "Describe this image")
        self.assertEqual(artifact["steps"][0]["system_prompt"], "Return JSON")


if __name__ == "__main__":
    unittest.main()
