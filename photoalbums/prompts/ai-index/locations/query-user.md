Based on the image and the above context, identify the location(s) shown.
Nominatim accepts free-form place name queries in any language (e.g. "Eiffel Tower, Paris, France", "Cafe Paris, New York"). Do NOT return raw latitude/longitude values - return human-readable place names that Nominatim can resolve.
Every non-empty Nominatim query must include a country name. If the page or OCR text supports a location but does not show the country, choose the single best country from the Album value above. If the Album contains multiple countries, choose the one best supported by the image, caption, or OCR text.
Return:
- primary_query: the single most specific location for the primary GPS, including country (empty string if unknown)
- named_queries: list of named places shown in the image (may be empty)
Each named_queries item should be an object with: name, world_region, country_name, country_code, province_or_state, city, sublocation.
Include country_name for every named query unless no country can be determined from the page, OCR text, caption, or album title. Include other broader geography when supported so the later Nominatim query is less ambiguous.
