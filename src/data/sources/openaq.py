"""
OpenAQ v3 — Global Air Quality data connector.

Fetches PM2.5, NO₂, O₃ and other pollutant measurements from the
OpenAQ REST API v3. Requires a free API key.

In v3 the measurements endpoint is per-sensor:
    GET /v3/sensors/{sensor_id}/measurements
Locations and their sensors are discovered via:
    GET /v3/locations
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests
from loguru import logger

from config import settings
from utils import ISO3_TO_ISO2

# OpenAQ data coverage starts in 2015
_OPENAQ_MIN_YEAR = 2015


class OpenAQSource:
    """
    Connector for OpenAQ v3 air quality measurements.

    Handles pagination, rate limiting, and retries for reliable
    large-scale data retrieval.
    """

    # OpenAQ parameter IDs (v3)
    PARAM_PM25 = 2
    PARAM_NO2 = 7
    PARAM_O3 = 8
    PARAM_PM10 = 1

    PARAMETER_MAP = {
        "pm25": PARAM_PM25,
        "no2": PARAM_NO2,
        "o3": PARAM_O3,
        "pm10": PARAM_PM10,
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.base_url = base_url or settings.data.openaq_base_url
        self.api_key = api_key or settings.data.openaq_api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})
        self._cache_dir = settings.data.processed_dir / "openaq"

    def fetch_locations(
        self,
        countries: list[str] | None = None,
        parameter: str = "pm25",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch monitoring station locations.

        Args:
            countries: List of ISO 3166-1 alpha-2 country codes (e.g., ['US', 'IN']).
            parameter: Filter locations by parameter availability.
            limit: Maximum number of locations to return.

        Returns:
            DataFrame with columns: location_id, name, country, lat, lon,
                                    sensor_ids, last_updated
        """
        if countries:
            countries = [ISO3_TO_ISO2.get(c.upper(), c.upper()) for c in countries]

        logger.info(f"Fetching OpenAQ locations (countries={countries}, param={parameter})")
        param_id = self.PARAMETER_MAP.get(parameter, self.PARAM_PM25)

        params: dict[str, Any] = {
            "limit": min(limit, 1000),
            "parameters_id": param_id,
        }

        data = []
        if countries:
            for country in countries:
                cp = params.copy()
                cp["iso"] = country
                data.extend(self._paginated_get("/locations", cp, max_results=limit))
                time.sleep(0.3)
        else:
            data = self._paginated_get("/locations", params, max_results=limit)

        rows = []
        for loc in data:
            # Extract sensor IDs for the requested parameter
            sensor_ids = []
            for sensor in loc.get("sensors", []):
                sensor_param = sensor.get("parameter", {})
                if sensor_param.get("id") == param_id:
                    sensor_ids.append(sensor["id"])

            rows.append(
                {
                    "location_id": loc.get("id"),
                    "name": loc.get("name", ""),
                    "country": loc.get("country", {}).get("code", ""),
                    "lat": loc.get("coordinates", {}).get("latitude"),
                    "lon": loc.get("coordinates", {}).get("longitude"),
                    "sensor_ids": ",".join(str(s) for s in sensor_ids),
                    "last_updated": loc.get("datetimeLast", {}).get("utc")
                    if isinstance(loc.get("datetimeLast"), dict)
                    else loc.get("datetimeLast"),
                }
            )

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} locations")
        return df

    def fetch_measurements(
        self,
        parameter: str = "pm25",
        countries: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        location_ids: list[int] | None = None,
        limit: int = 10000,
        max_locations: int = 50,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical air quality measurements.

        In OpenAQ v3, measurements are retrieved per-sensor. This method
        first discovers locations/sensors matching the parameter, then
        fetches measurements from each sensor.

        Args:
            parameter: One of 'pm25', 'no2', 'o3', 'pm10'.
            countries: List of ISO country codes.
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).
            location_ids: List of specific location IDs.
            limit: Maximum total measurements.
            max_locations: Maximum number of locations to query sensors from.
            use_cache: Use cached data if available.

        Returns:
            DataFrame with columns: time, location_id, lat, lon, parameter,
                                    value, unit
        """
        if countries:
            countries = [ISO3_TO_ISO2.get(c.upper(), c.upper()) for c in countries]

        if date_from:
            try:
                dt_from = pd.to_datetime(date_from)
                if dt_from.year < _OPENAQ_MIN_YEAR:
                    logger.info(
                        f"Clamping OpenAQ query start date {date_from} to {_OPENAQ_MIN_YEAR}-01-01"
                    )
                    date_from = f"{_OPENAQ_MIN_YEAR}-01-01"
            except Exception:
                pass

        cache_key = (
            f"{parameter}_{'_'.join(countries) if countries else 'global'}_{date_from}_{date_to}"
        )
        cache_path = self._cache_dir / f"{cache_key}.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached OpenAQ data: {cache_path.name}")
            return pd.read_parquet(cache_path)

        logger.info(
            f"Fetching OpenAQ measurements: {parameter} "
            f"(countries={countries}, {date_from} to {date_to})"
        )

        param_id = self.PARAMETER_MAP.get(parameter)
        if param_id is None:
            raise ValueError(f"Unknown parameter: {parameter}. Use: {list(self.PARAMETER_MAP)}")

        # Step 1: Discover locations and their sensor IDs
        locations_df = self.fetch_locations(
            countries=countries,
            parameter=parameter,
            limit=max_locations,
        )

        if locations_df.empty:
            logger.warning(f"No locations found for {parameter}")
            return pd.DataFrame()

        # Step 2: Fetch measurements from each sensor
        all_rows: list[dict] = []
        measurements_per_sensor = max(limit // max(len(locations_df), 1), 100)

        for _, loc in locations_df.iterrows():
            sensor_ids = [int(s) for s in str(loc["sensor_ids"]).split(",") if s.strip()]
            if not sensor_ids:
                continue

            for sensor_id in sensor_ids:
                if len(all_rows) >= limit:
                    break

                try:
                    sensor_data = self._fetch_sensor_measurements(
                        sensor_id=sensor_id,
                        date_from=date_from,
                        date_to=date_to,
                        limit=min(measurements_per_sensor, limit - len(all_rows)),
                    )

                    for m in sensor_data:
                        period = m.get("period", {})
                        dt_from = period.get("datetimeFrom", {})
                        all_rows.append(
                            {
                                "time": dt_from.get("utc")
                                if isinstance(dt_from, dict)
                                else dt_from,
                                "location_id": loc["location_id"],
                                "lat": loc["lat"],
                                "lon": loc["lon"],
                                "parameter": parameter,
                                "value": m.get("value"),
                                "unit": m.get("parameter", {}).get("units", ""),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Failed to fetch sensor {sensor_id}: {e}")
                    continue

                time.sleep(0.2)  # Rate limiting

            if len(all_rows) >= limit:
                break

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Fetched {len(df):,} measurements, cached to {cache_path.name}")

        return df

    def _fetch_sensor_measurements(
        self,
        sensor_id: int,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch measurements for a specific sensor."""
        params: dict[str, Any] = {"limit": min(limit, 1000)}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        resp = self._get_with_retry(f"/sensors/{sensor_id}/measurements", params)
        return resp.get("results", [])  # type: ignore[no-any-return]

    def _paginated_get(
        self,
        endpoint: str,
        params: dict[str, Any],
        max_results: int = 10000,
    ) -> list[dict]:
        """
        Paginated GET with rate-limit handling and retries.
        """
        all_results: list[dict] = []
        page = 1
        per_page = min(params.get("limit", 1000), 1000)

        while len(all_results) < max_results:
            params["page"] = page
            params["limit"] = per_page

            response = self._get_with_retry(endpoint, params)
            results = response.get("results", [])

            if not results:
                break

            all_results.extend(results)
            page += 1

            # Respect rate limits
            time.sleep(0.2)

        return all_results[:max_results]

    def _get_with_retry(
        self,
        endpoint: str,
        params: dict[str, Any],
        max_retries: int = 3,
    ) -> dict:
        """GET request with exponential backoff."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

            except requests.exceptions.RequestException as e:
                status = getattr(e.response, "status_code", None)
                if status in (401, 403, 404):
                    logger.error(f"OpenAQ {status} error for {endpoint}")
                    raise
                if attempt == max_retries - 1:
                    logger.error(f"OpenAQ request failed after {max_retries} retries: {e}")
                    raise
                wait = 2**attempt
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                time.sleep(wait)

        return {"results": []}
