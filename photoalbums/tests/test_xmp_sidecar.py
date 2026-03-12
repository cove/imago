import sys
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import xmp_sidecar


class TestXMPSidecar(unittest.TestCase):
    def test_write_xmp_sidecar_outputs_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.xmp"
            xmp_sidecar.write_xmp_sidecar(
                out,
                creator_tool="imago-test",
                person_names=["Alice", "Bob", "Alice"],
                subjects=["dog", "park", "dog"],
                description="People: Alice, Bob. Objects: dog, park.",
                source_text="Family_2020_B01_P01_S01.tif; Family_2020_B01_P01_S02.tif",
                ocr_text="Welcome to the park",
                detections_payload={"objects": [{"label": "dog", "score": 0.9}]},
                subphotos=[
                    {
                        "index": 1,
                        "bounds": {"x": 10, "y": 20, "width": 300, "height": 200},
                        "description": "A dog in the park.",
                        "ocr_text": "park sign",
                        "people": ["Alice"],
                        "subjects": ["dog", "park"],
                        "detections": {"objects": [{"label": "dog", "score": 0.9}]},
                    }
                ],
            )

            self.assertTrue(out.exists())
            root = ET.parse(out).getroot()
            xml = ET.tostring(root, encoding="unicode")
            self.assertIn("Alice", xml)
            self.assertIn("Bob", xml)
            self.assertIn("dog", xml)
            self.assertIn("imago-test", xml)
            self.assertIn("Welcome to the park", xml)
            self.assertIn("Family_2020_B01_P01_S01.tif", xml)
            self.assertIn("SubPhotos", xml)
            self.assertIn("A dog in the park.", xml)
            self.assertIn("park sign", xml)


if __name__ == "__main__":
    unittest.main()
