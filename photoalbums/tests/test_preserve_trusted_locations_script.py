from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from photoalbums.lib.xmp_sidecar import read_ai_sidecar_state, write_xmp_sidecar

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preserve_trusted_locations.py"
SPEC = importlib.util.spec_from_file_location("preserve_trusted_locations_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("preserve_trusted_locations_script", MODULE)
SPEC.loader.exec_module(MODULE)


def _write_sidecar(path: Path, *, city: str, state: str, lat: str = "", lon: str = "") -> None:
    write_xmp_sidecar(
        path,
        person_names=[],
        subjects=[],
        description="",
        ocr_text="",
        gps_latitude=lat,
        gps_longitude=lon,
        location_city=city,
        location_state=state,
        location_country="United States",
    )


def test_is_trusted_location_accepts_named_manual_areas_without_gps() -> None:
    assert MODULE._is_trusted_location({"location_city": "San Marino", "location_state": "CA"})
    assert MODULE._is_trusted_location({"location_city": "Altadena", "location_state": "California"})


def test_is_trusted_location_accepts_other_la_area_coordinates() -> None:
    assert MODULE._is_trusted_location(
        {
            "location_city": "Pasadena",
            "location_state": "California",
            "gps_latitude": "34.1478",
            "gps_longitude": "-118.1445",
        }
    )
    assert not MODULE._is_trusted_location(
        {
            "location_city": "San Diego",
            "location_state": "California",
            "gps_latitude": "32.7157",
            "gps_longitude": "-117.1611",
        }
    )


def test_snapshot_locations_keeps_only_trusted_records(tmp_path: Path) -> None:
    trusted = tmp_path / "Family_1953-1960_B03_Pages" / "trusted.xmp"
    untrusted = tmp_path / "Europe_1973_B00_Pages" / "untrusted.xmp"
    trusted.parent.mkdir()
    untrusted.parent.mkdir()
    _write_sidecar(trusted, city="San Marino", state="California", lat="34.11512", lon="-118.10492")
    _write_sidecar(untrusted, city="Paris", state="Île-de-France", lat="48.8566", lon="2.3522")

    records = MODULE.snapshot_locations(tmp_path)

    assert records == [
        {
            "path": "Family_1953-1960_B03_Pages/trusted.xmp",
            "gps_latitude": "34.11512",
            "gps_longitude": "-118.10492",
            "location_city": "San Marino",
            "location_state": "California",
            "location_country": "United States",
            "location_sublocation": "",
            "location_created": "San Marino, California, United States",
        }
    ]


def test_snapshot_locations_skips_title_pages(tmp_path: Path) -> None:
    title_page = tmp_path / "Family_1953-1960_B03_Pages" / "Family_1953-1960_B03_P01_V.xmp"
    title_page.parent.mkdir()
    _write_sidecar(title_page, city="San Marino", state="California", lat="34.11512", lon="-118.10492")

    assert MODULE.snapshot_locations(tmp_path) == []


def test_restore_locations_reapplies_snapshot_over_regenerated_location(tmp_path: Path) -> None:
    sidecar = tmp_path / "Family_1953-1960_B03_Pages" / "trusted.xmp"
    sidecar.parent.mkdir()
    _write_sidecar(sidecar, city="Pasadena", state="California", lat="34.1478", lon="-118.1445")
    record = {
        "path": "Family_1953-1960_B03_Pages/trusted.xmp",
        "gps_latitude": "34.11512",
        "gps_longitude": "-118.10492",
        "location_city": "San Marino",
        "location_state": "California",
        "location_country": "United States",
        "location_sublocation": "",
        "location_created": "San Marino, California, United States",
    }

    restored, missing = MODULE.restore_locations(tmp_path, [record], dry_run=False)

    assert (restored, missing) == (1, 0)
    state = read_ai_sidecar_state(sidecar)
    assert state is not None
    assert state["gps_latitude"] == "34.11512"
    assert state["gps_longitude"] == "-118.10492"
    assert state["location_city"] == "San Marino"
    assert state["location_state"] == "California"
    assert state["location_country"] == "United States"
    detections = state["detections"]
    assert isinstance(detections, dict)
    assert detections["location"] == {
        "gps_latitude": "34.11512",
        "gps_longitude": "-118.10492",
        "city": "San Marino",
        "state": "California",
        "country": "United States",
        "sublocation": "",
    }
