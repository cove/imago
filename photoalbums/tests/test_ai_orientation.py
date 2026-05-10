import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import ai_orientation


class FakeOrientationEngine:
    effective_model_name = "vision-model"

    def __init__(self, right_side_up):
        self._result = ai_orientation.OrientationResult(engine="lmstudio", right_side_up=right_side_up)

    def analyze(self, *_args, **_kwargs) -> ai_orientation.OrientationResult:
        return self._result


class TestAIOrientation(unittest.TestCase):
    def test_parse_orientation_response(self):
        result = ai_orientation._parse_orientation_response('{"right_side_up": false}')

        self.assertFalse(result.right_side_up)

    def test_correct_orientation_rotates_when_ai_says_not_right_side_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "scan.tif"
            image_path.write_bytes(b"placeholder")

            with mock.patch.object(ai_orientation, "rotate_image_180_in_place") as rotate_mock:
                result = ai_orientation.correct_orientation_after_scan(
                    image_path,
                    engine=FakeOrientationEngine(False),
                )

            rotate_mock.assert_called_once_with(image_path)
            self.assertTrue(result["right_side_up"])
            self.assertFalse(result["ai_right_side_up"])
            self.assertEqual(result["rotation_applied_degrees"], 180)

    def test_correct_orientation_leaves_upright_image_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "scan.tif"
            image_path.write_bytes(b"placeholder")

            with mock.patch.object(ai_orientation, "rotate_image_180_in_place") as rotate_mock:
                result = ai_orientation.correct_orientation_after_scan(
                    image_path,
                    engine=FakeOrientationEngine(True),
                )

            rotate_mock.assert_not_called()
            self.assertTrue(result["right_side_up"])
            self.assertEqual(result["rotation_applied_degrees"], 0)


if __name__ == "__main__":
    unittest.main()
