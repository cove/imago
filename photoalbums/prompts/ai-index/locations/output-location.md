`{"location_name": "", "gps_latitude": "", "gps_longitude": ""}`

- `location_name`: concise geocoding query or empty string.
- Include a country name in every non-empty query, using the album title when the country is not visible on the page.
- `gps_latitude`: decimal degrees if explicitly visible in image text, else empty string.
- `gps_longitude`: decimal degrees if explicitly visible in image text, else empty string.
- Just return the JSON without any extra text or explanation.
