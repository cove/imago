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
from photoalbums.lib.ai_location import _resolve_location_payload


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
                self.assertTrue(request.full_url.startswith("https://nominatim.openstreetmap.org/search?"))
                self.assertIn("format=jsonv2", request.full_url)
                self.assertIn("limit=1", request.full_url)
                self.assertIn("accept-language=en", request.full_url)
                self.assertIn("Mogao+Caves%2C+Dunhuang%2C+Gansu%2C+China", request.full_url)
                self.assertEqual(
                    request.headers["User-agent"],
                    ai_geocode.DEFAULT_GEOCODER_USER_AGENT,
                )
                return _FakeResponse()

            geocoder = ai_geocode.NominatimGeocoder(cache_path=cache_path)
            with mock.patch.object(ai_geocode.urllib.request, "urlopen", side_effect=fake_urlopen):
                result = geocoder.geocode("Mogao Caves, Dunhuang, Gansu, China")

            self.assertEqual(result.latitude, "39.9361")
            self.assertEqual(result.longitude, "94.8076")
            self.assertEqual(result.display_name, "Mogao Caves, Dunhuang, Jiuquan, Gansu, China")
            self.assertEqual(result.raw["lat"], "39.9361")
            self.assertTrue(cache_path.exists())
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertIn("mogao caves, dunhuang, gansu, china", cached)
            self.assertEqual(cached["mogao caves, dunhuang, gansu, china"]["raw"]["lon"], "94.8076")

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
                            "raw": {"lat": "39.9361", "lon": "94.8076"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            geocoder = ai_geocode.NominatimGeocoder(cache_path=cache_path)

            with mock.patch.object(ai_geocode.urllib.request, "urlopen") as urlopen_mock:
                result = geocoder.geocode("Mogao Caves, Dunhuang, Gansu, China")

            urlopen_mock.assert_not_called()
            self.assertEqual(result.latitude, "39.9361")
            self.assertEqual(result.longitude, "94.8076")

    def test_geocode_strips_leading_the_from_network_lookup_only(self):
        response_payload = [
            {
                "lat": "51.7520",
                "lon": "-1.2577",
                "display_name": "Oxford, Oxfordshire, England, United Kingdom",
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
                self.assertIn("Oxford%2C+England", request.full_url)
                self.assertNotIn("The+Oxford%2C+England", request.full_url)
                return _FakeResponse()

            geocoder = ai_geocode.NominatimGeocoder(cache_path=cache_path)
            with mock.patch.object(ai_geocode.urllib.request, "urlopen", side_effect=fake_urlopen):
                result = geocoder.geocode("The Oxford, England")

            self.assertEqual(result.query, "The Oxford, England")
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertIn("the oxford, england", cached)
            self.assertEqual(cached["the oxford, england"]["query"], "The Oxford, England")

    def test_geocode_extracts_street_sublocation_from_display_name(self):
        response_payload = [
            {
                "lat": "48.2103",
                "lon": "16.3574",
                "display_name": "Vienna City Hall, 1, Rathausplatz, Regierungsviertel, Innere Stadt, Vienna, 1010, Austria",
                "address": {
                    "city": "Vienna",
                    "country": "Austria",
                },
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
            geocoder = ai_geocode.NominatimGeocoder(cache_path=cache_path)
            with mock.patch.object(ai_geocode.urllib.request, "urlopen", return_value=_FakeResponse()):
                result = geocoder.geocode("Vienna City Hall, Vienna, Austria")

            self.assertIsNotNone(result)
            self.assertEqual(result.sublocation, "1 Rathausplatz")
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(
                cached["vienna city hall, vienna, austria"]["sublocation"],
                "1 Rathausplatz",
            )

    def test_location_payload_falls_back_from_yugoslavia_city_to_modern_country(self):
        class _FakeResult:
            query = "Zagreb, Croatia"
            latitude = "45.8130967"
            longitude = "15.9772795"
            display_name = "Grad Zagreb, Hrvatska"
            source = "nominatim"
            city = "City of Zagreb"
            state = ""
            country = "Croatia"
            sublocation = ""

        class _FakeGeocoder:
            def __init__(self):
                self.queries = []

            def geocode(self, query):
                self.queries.append(query)
                return _FakeResult() if query == "Zagreb, Croatia" else None

        geocoder = _FakeGeocoder()
        result = _resolve_location_payload(
            geocoder=geocoder,
            gps_latitude="",
            gps_longitude="",
            location_name="Hotel InterContinental, Zagreb, Yugoslavia",
        )

        self.assertEqual(
            geocoder.queries,
            [
                "Hotel InterContinental, Zagreb, Yugoslavia",
                "Hotel InterContinental, Zagreb, Croatia",
                "Zagreb, Croatia",
            ],
        )
        self.assertEqual(result["gps_latitude"], 45.8130967)
        self.assertEqual(result["country"], "Croatia")

    def test_location_payload_prefers_modern_full_yugoslavia_query_before_old_city_query(self):
        class _FakeResult:
            query = "Hotel InterContinental, Belgrade, Serbia"
            latitude = "44.8094406"
            longitude = "20.4341261"
            display_name = "Hotel Intercontinental, Belgrade, Serbia"
            source = "nominatim"
            city = "Belgrade"
            state = ""
            country = "Serbia"
            sublocation = ""

        class _FakeGeocoder:
            def __init__(self):
                self.queries = []

            def geocode(self, query):
                self.queries.append(query)
                return _FakeResult() if query == "Hotel InterContinental, Belgrade, Serbia" else None

        geocoder = _FakeGeocoder()
        result = _resolve_location_payload(
            geocoder=geocoder,
            gps_latitude="",
            gps_longitude="",
            location_name="Hotel InterContinental, Belgrade, Yugoslavia",
        )

        self.assertEqual(
            geocoder.queries,
            [
                "Hotel InterContinental, Belgrade, Yugoslavia",
                "Hotel InterContinental, Belgrade, Serbia",
            ],
        )
        self.assertEqual(result["gps_latitude"], 44.8094406)
        self.assertEqual(result["country"], "Serbia")

    def test_location_payload_includes_raw_nominatim_result(self):
        class _FakeResult:
            query = "Karnten, Austria"
            latitude = "46.75"
            longitude = "13.8333333"
            display_name = "KARNTEN, AUSTRIA"
            source = "nominatim"
            city = ""
            state = ""
            country = ""
            sublocation = ""
            raw = {"place_id": 123, "lat": "46.75", "lon": "13.8333333"}

        class _FakeGeocoder:
            def geocode(self, query):
                return _FakeResult()

        result = _resolve_location_payload(
            geocoder=_FakeGeocoder(),
            gps_latitude="",
            gps_longitude="",
            location_name="Karnten, Austria",
        )

        self.assertEqual(result["nominatim"]["raw"]["place_id"], 123)


if __name__ == "__main__":
    unittest.main()
