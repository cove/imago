from __future__ import annotations

import json
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GEOCODER_BASE_URL = "https://nominatim.openstreetmap.org"
DEFAULT_GEOCODER_TIMEOUT_SECONDS = 20.0
DEFAULT_GEOCODER_MIN_INTERVAL_SECONDS = 1.0
DEFAULT_GEOCODER_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "geocode_cache.json"
DEFAULT_GEOCODER_USER_AGENT = "imago-photoalbums-ai-index/1.0"


def _clean_query(value: str) -> str:
    return " ".join(str(value or "").split())


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return normalized.encode("ascii", "ignore").decode("ascii")


def _clean_display_part(value: str) -> str:
    return _clean_query(_ascii_fold(value).strip(" ,;"))


def _normalize_display_name(
    *,
    query: str,
    display_name: str,
    city: str = "",
    state: str = "",
    country: str = "",
) -> str:
    clean_display = _clean_query(display_name)
    if clean_display and all(ord(ch) <= 127 for ch in clean_display):
        return clean_display
    parts: list[str] = []
    seen: set[str] = set()
    for raw in (query, city, state, country):
        part = _clean_display_part(raw)
        if not part:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(part)
    if parts:
        return ", ".join(parts)
    return _clean_display_part(display_name) or clean_display


def normalize_geocoder_base_url(value: str) -> str:
    text = str(value or "").strip() or DEFAULT_GEOCODER_BASE_URL
    return text.rstrip("/")


@dataclass(frozen=True)
class GeocodeResult:
    query: str
    latitude: str
    longitude: str
    display_name: str
    source: str = "nominatim"
    city: str = ""
    state: str = ""
    country: str = ""


class NominatimGeocoder:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_GEOCODER_BASE_URL,
        cache_path: str | Path = DEFAULT_GEOCODER_CACHE_PATH,
        timeout_seconds: float = DEFAULT_GEOCODER_TIMEOUT_SECONDS,
        min_interval_seconds: float = DEFAULT_GEOCODER_MIN_INTERVAL_SECONDS,
        user_agent: str = DEFAULT_GEOCODER_USER_AGENT,
    ):
        self.base_url = normalize_geocoder_base_url(base_url)
        self.cache_path = Path(cache_path)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.user_agent = str(user_agent or DEFAULT_GEOCODER_USER_AGENT).strip() or DEFAULT_GEOCODER_USER_AGENT
        self._cache = self._load_cache()
        self._last_request_started_at = 0.0

    def _load_cache(self) -> dict[str, dict[str, str]]:
        if not self.cache_path.is_file():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        rows: dict[str, dict[str, str]] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            rows[key] = {str(field): str(item) for field, item in value.items()}
        return rows

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True)
        self.cache_path.write_text(serialized + "\n", encoding="utf-8")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_started_at
        remaining = float(self.min_interval_seconds) - float(elapsed)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_started_at = time.monotonic()

    def _cache_key(self, query: str) -> str:
        return _clean_query(query).casefold()

    def _cache_result(self, result: GeocodeResult | None) -> None:
        if result is None:
            return
        self._cache[self._cache_key(result.query)] = {
            "query": result.query,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "display_name": result.display_name,
            "source": result.source,
            "city": result.city,
            "state": result.state,
            "country": result.country,
            "status": "ok",
        }
        self._save_cache()

    def _cache_miss(self, query: str) -> None:
        self._cache[self._cache_key(query)] = {
            "query": _clean_query(query),
            "status": "miss",
        }
        self._save_cache()

    def _cached_result(self, query: str) -> GeocodeResult | None:
        row = self._cache.get(self._cache_key(query))
        if not isinstance(row, dict):
            return None
        if str(row.get("status") or "") == "miss":
            return None
        latitude = str(row.get("latitude") or "").strip()
        longitude = str(row.get("longitude") or "").strip()
        if not latitude or not longitude:
            return None
        return GeocodeResult(
            query=str(row.get("query") or _clean_query(query)),
            latitude=latitude,
            longitude=longitude,
            display_name=_normalize_display_name(
                query=str(row.get("query") or _clean_query(query)),
                display_name=str(row.get("display_name") or "").strip(),
                city=str(row.get("city") or "").strip(),
                state=str(row.get("state") or "").strip(),
                country=str(row.get("country") or "").strip(),
            ),
            source=str(row.get("source") or "nominatim"),
            city=str(row.get("city") or "").strip(),
            state=str(row.get("state") or "").strip(),
            country=str(row.get("country") or "").strip(),
        )

    def geocode(self, query: str) -> GeocodeResult | None:
        clean_query = _clean_query(query)
        if not clean_query:
            return None
        cached = self._cached_result(clean_query)
        if cached is not None:
            return cached
        if self._cache.get(self._cache_key(clean_query), {}).get("status") == "miss":
            return None

        params = urllib.parse.urlencode(
            {
                "q": clean_query,
                "format": "jsonv2",
                "limit": "1",
                "addressdetails": "1",
                "accept-language": "en",
            }
        )
        self._throttle()
        request = urllib.request.Request(
            f"{self.base_url}/search?{params}",
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.timeout_seconds)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Nominatim geocoding request failed: {details or f'HTTP {exc.code}'}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Nominatim is unreachable at {self.base_url}: {exc.reason}") from exc

        if not isinstance(payload, list) or not payload:
            self._cache_miss(clean_query)
            return None
        top = payload[0]
        if not isinstance(top, dict):
            self._cache_miss(clean_query)
            return None
        latitude = str(top.get("lat") or "").strip()
        longitude = str(top.get("lon") or "").strip()
        if not latitude or not longitude:
            self._cache_miss(clean_query)
            return None
        address = top.get("address") or {}
        city = str(
            address.get("city") or address.get("town") or address.get("village") or address.get("municipality") or ""
        ).strip()
        result = GeocodeResult(
            query=clean_query,
            latitude=latitude,
            longitude=longitude,
            display_name=_normalize_display_name(
                query=clean_query,
                display_name=str(top.get("display_name") or "").strip(),
                city=city,
                state=str(address.get("state") or "").strip(),
                country=str(address.get("country") or "").strip(),
            ),
            city=city,
            state=str(address.get("state") or "").strip(),
            country=str(address.get("country") or "").strip(),
        )
        self._cache_result(result)
        return result

    def reverse_geocode(self, lat: float | str, lon: float | str) -> GeocodeResult | None:
        lat_str = str(lat).strip()
        lon_str = str(lon).strip()
        if not lat_str or not lon_str:
            return None
        
        query = f"{lat_str},{lon_str}"
        cached = self._cached_result(query)
        if cached is not None:
            return cached
        if self._cache.get(self._cache_key(query), {}).get("status") == "miss":
            return None

        params = urllib.parse.urlencode(
            {
                "lat": lat_str,
                "lon": lon_str,
                "format": "jsonv2",
                "zoom": "18",
                "addressdetails": "1",
                "accept-language": "en",
            }
        )
        self._throttle()
        request = urllib.request.Request(
            f"{self.base_url}/reverse?{params}",
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.timeout_seconds)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Nominatim reverse geocoding request failed: {details or f'HTTP {exc.code}'}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Nominatim is unreachable at {self.base_url}: {exc.reason}") from exc

        if not isinstance(payload, dict):
            self._cache_miss(query)
            return None
        
        latitude = str(payload.get("lat") or "").strip()
        longitude = str(payload.get("lon") or "").strip()
        if not latitude or not longitude:
            self._cache_miss(query)
            return None

        address = payload.get("address") or {}
        city = str(
            address.get("city") or address.get("town") or address.get("village") or address.get("municipality") or ""
        ).strip()
        
        result = GeocodeResult(
            query=query,
            latitude=latitude,
            longitude=longitude,
            display_name=_normalize_display_name(
                query=query,
                display_name=str(payload.get("display_name") or "").strip(),
                city=city,
                state=str(address.get("state") or "").strip(),
                country=str(address.get("country") or "").strip(),
            ),
            city=city,
            state=str(address.get("state") or "").strip(),
            country=str(address.get("country") or "").strip(),
        )
        self._cache_result(result)
        return result
