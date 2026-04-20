"""Tests for run_locations_step (Task 2.4)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.ai_location import run_locations_step
from photoalbums.lib.ai_caption import LocationQueryResult


def _make_caption_engine(engine_name="lmstudio", primary_query="", named_queries=None, fallback=False, error=""):
    engine = mock.Mock()
    engine.engine = engine_name
    engine.generate_location_queries.return_value = LocationQueryResult(
        engine=engine_name,
        primary_query=primary_query,
        named_queries=named_queries or [],
        fallback=fallback,
        error=error,
    )
    return engine


def _make_geocoder(latitude=None, longitude=None):
    geocoder = mock.Mock()
    if latitude is not None:
        geo_result = mock.Mock()
        geo_result.latitude = latitude
        geo_result.longitude = longitude
        geo_result.city = "TestCity"
        geo_result.state = ""
        geo_result.country = "France"
        geo_result.sublocation = ""
        geocoder.geocode.return_value = geo_result
    else:
        geocoder.geocode.return_value = None
    return geocoder


class TestRunLocationsStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "a.jpg"
        self.image.write_bytes(b"fake")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_none_when_engine_not_lmstudio(self):
        engine = _make_caption_engine(engine_name="none")
        result = run_locations_step(
            caption_engine=engine,
            image_path=self.image,
            caption_text="",
        )
        self.assertIsNone(result)

    def test_primary_query_resolved_to_gps(self):
        engine = _make_caption_engine(primary_query="Paris, France")
        geocoder = _make_geocoder(latitude=48.8566, longitude=2.3522)

        with mock.patch("photoalbums.lib.ai_location._resolve_location_payload") as payload_mock:
            payload_mock.return_value = {
                "query": "Paris, France",
                "gps_latitude": "48.8566",
                "gps_longitude": "2.3522",
                "city": "Paris",
                "country": "France",
            }
            result = run_locations_step(
                caption_engine=engine,
                image_path=self.image,
                caption_text="A photo from Paris.",
                geocoder=geocoder,
            )

        assert result is not None
        self.assertEqual(result["location"]["city"], "Paris")
        self.assertTrue(result["location_shown_ran"])

    def test_no_primary_query_returns_empty_location(self):
        engine = _make_caption_engine(primary_query="")

        with mock.patch("photoalbums.lib.ai_location._resolve_location_payload") as payload_mock:
            payload_mock.return_value = {}
            result = run_locations_step(
                caption_engine=engine,
                image_path=self.image,
                caption_text="",
            )

        assert result is not None
        self.assertEqual(result["location"], {})
        self.assertTrue(result["location_shown_ran"])

    def test_named_queries_stored_with_coordinates_when_geocoder_available(self):
        engine = _make_caption_engine(
            primary_query="Versailles",
            named_queries=["Palace of Versailles"],
        )
        geocoder = _make_geocoder(latitude=48.8049, longitude=2.1204)

        with mock.patch("photoalbums.lib.ai_location._resolve_location_payload", return_value={}):
            result = run_locations_step(
                caption_engine=engine,
                image_path=self.image,
                caption_text="Visit to Versailles.",
                geocoder=geocoder,
            )

        assert result is not None
        shown = result["locations_shown"]
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown[0]["name"], "Palace of Versailles")
        self.assertIn("gps_latitude", shown[0])
        self.assertEqual(shown[0]["gps_source"], "nominatim")

    def test_named_queries_stored_without_coordinates_when_nominatim_unavailable(self):
        engine = _make_caption_engine(
            primary_query="",
            named_queries=["Eiffel Tower", "Louvre Museum"],
        )
        geocoder = _make_geocoder(latitude=None)  # geocoder returns None

        with mock.patch("photoalbums.lib.ai_location._resolve_location_payload", return_value={}):
            result = run_locations_step(
                caption_engine=engine,
                image_path=self.image,
                caption_text="",
                geocoder=geocoder,
            )

        assert result is not None
        shown = result["locations_shown"]
        self.assertEqual(len(shown), 2)
        self.assertEqual(shown[0]["name"], "Eiffel Tower")
        self.assertNotIn("gps_latitude", shown[0])
        self.assertEqual(shown[1]["name"], "Louvre Museum")

    def test_fallback_returns_empty_location_with_ran_true(self):
        engine = _make_caption_engine(fallback=True, error="model offline")

        result = run_locations_step(
            caption_engine=engine,
            image_path=self.image,
            caption_text="",
        )

        assert result is not None
        self.assertEqual(result["location"], {})
        self.assertEqual(result["locations_shown"], [])
        self.assertTrue(result["location_shown_ran"])


if __name__ == "__main__":
    unittest.main()
