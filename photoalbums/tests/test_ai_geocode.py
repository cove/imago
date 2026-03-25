import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib import ai_geocode


class TestAIGeocode(unittest.TestCase):
    def test_geocode_queries_nominatim_and_caches_result(self):
        response_payload = [
            {
                "lat": "39.9361",
                "lon": "94.8076",
                "display_name": "Mogao Caves, Dunhuang, Jiuquan, Gansu, China",
            }
        ]

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "geocode_cache.json"

            def fake_urlopen(request, timeout):
                self.assertEqual(timeout, ai_geocode.DEFAULT_GEOCODER_TIMEOUT_SECONDS)
                self.assertTrue(
                    request.full_url.startswith(
                        "https://nominatim.openstreetmap.org/search?"
                    )
                )
                self.assertIn("format=jsonv2", request.full_url)
                self.assertIn("limit=1", request.full_url)
                self.assertIn("accept-language=en", request.full_url)
                self.assertIn(
                    "Mogao+Caves%2C+Dunhuang%2C+Gansu%2C+China", request.full_url
                )
                self.assertEqual(
                    request.headers["User-agent"],
                    ai_geocode.DEFAULT_GEOCODER_USER_AGENT,
                )
                return _FakeResponse()

            geocoder = ai_geocode.NominatimGeocoder(cache_path=cache_path)
            with mock.patch.object(
                ai_geocode.urllib.request, "urlopen", side_effect=fake_urlopen
            ):
                result = geocoder.geocode("Mogao Caves, Dunhuang, Gansu, China")

            self.assertEqual(result.latitude, "39.9361")
            self.assertEqual(result.longitude, "94.8076")
            self.assertEqual(
                result.display_name, "Mogao Caves, Dunhuang, Jiuquan, Gansu, China"
            )
            self.assertTrue(cache_path.exists())
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertIn("mogao caves, dunhuang, gansu, china", cached)

    def test_geocode_uses_cached_result_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "geocode_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "mogao caves, dunhuang, gansu, china": {
                            "query": "Mogao Caves, Dunhuang, Gansu, China",
                            "latitude": "39.9361",
                            "longitude": "94.8076",
                            "display_name": "Mogao Caves, Dunhuang, Jiuquan, Gansu, China",
                            "source": "nominatim",
                            "status": "ok",
                        }
                    }
                ),
                encoding="utf-8",
            )
            geocoder = ai_geocode.NominatimGeocoder(cache_path=cache_path)

            with mock.patch.object(
                ai_geocode.urllib.request, "urlopen"
            ) as urlopen_mock:
                result = geocoder.geocode("Mogao Caves, Dunhuang, Gansu, China")

            urlopen_mock.assert_not_called()
            self.assertEqual(result.latitude, "39.9361")
            self.assertEqual(result.longitude, "94.8076")


if __name__ == "__main__":
    unittest.main()
