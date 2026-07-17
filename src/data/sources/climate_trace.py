"""
Climate TRACE — Global Greenhouse Gas Emissions data connector.

Fetches CO₂, CH₄, N₂O emissions inventories from the Climate TRACE
API v7 using the ``/sources`` endpoint. Data coverage starts from 2015.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests
from loguru import logger

from config import settings
from utils import _ISO2_TO_ISO3

# Climate TRACE data coverage starts at 2015
_CLIMATE_TRACE_MIN_YEAR = 2015


class ClimateTraceSource:
    """
    Connector for Climate TRACE emissions data.

    Uses the ``/v7/sources`` endpoint which returns per-asset emission
    records. Results are aggregated by year, sector, and optionally
    country in Python after retrieval.
    """

    SECTORS = [
        "power",
        "transportation",
        "manufacturing",
        "buildings",
        "agriculture",
        "fossil-fuel-operations",
        "waste",
        "forestry-and-land-use",
        "mineral-extraction",
    ]

    GAS_TYPES = ["co2", "ch4", "n2o", "co2e_100yr", "co2e_20yr"]

    def __init__(
        self,
        base_url: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ):
        self.base_url = base_url or settings.data.climate_trace_base_url
        # Clamp start_year to the earliest year with Climate TRACE data
        raw_start = start_year or settings.data.start_year
        self.start_year = max(raw_start, _CLIMATE_TRACE_MIN_YEAR)
        self.end_year = end_year or settings.data.end_year
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._cache_dir = settings.data.processed_dir / "climate_trace"

    def fetch_country_emissions(
        self,
        countries: list[str] | None = None,
        gas: str = "co2e_100yr",
        sectors: list[str] | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch emissions data aggregated by country.

        Args:
            countries: ISO 3166-1 alpha-3 codes (e.g., ['USA', 'CHN', 'IND']).
            gas: Gas type. One of: co2, ch4, n2o, co2e_100yr, co2e_20yr.
            sectors: Sector filter. None = all sectors.
            use_cache: Use cached Parquet if available.

        Returns:
            DataFrame with columns: country, sector, year, gas,
            emissions_quantity, unit
        """
        if countries:
            # Map 2-letter codes (e.g. 'IN') to 3-letter codes (e.g. 'IND')
            countries = [_ISO2_TO_ISO3.get(c.upper(), c.upper()) for c in countries]

        cache_key = f"emissions_{'_'.join(countries or ['all'])}_{gas}"
        cache_path = self._cache_dir / f"{cache_key}.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached Climate TRACE data: {cache_path.name}")
            return pd.read_parquet(cache_path)

        logger.info(
            f"Fetching Climate TRACE emissions: gas={gas}, "
            f"countries={countries}, years={self.start_year}–{self.end_year}"
        )

        all_rows: list[dict[str, Any]] = []

        for year in range(self.start_year, self.end_year + 1):
            try:
                data = self._fetch_sources_for_year(year, gas, countries, sectors)
                all_rows.extend(data)
            except Exception as e:
                logger.warning(f"Failed to fetch {year}: {e}")
                continue

            time.sleep(0.3)

        df = pd.DataFrame(all_rows)

        if not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(df):,} emission records")

        return df

    def fetch_global_totals(
        self,
        gas: str = "co2e_100yr",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch global total emissions by year and sector.

        Returns:
            DataFrame with columns: year, sector, gas, emissions_quantity
        """
        cache_path = self._cache_dir / f"global_totals_{gas}.parquet"
        if use_cache and cache_path.exists():
            logger.info("Loading cached global emission totals")
            return pd.read_parquet(cache_path)

        logger.info(
            f"Fetching Climate TRACE global totals ({gas}) "
            f"for years {self.start_year}–{self.end_year}"
        )

        all_rows: list[dict[str, Any]] = []

        for year in range(self.start_year, self.end_year + 1):
            try:
                sources = self._fetch_sources_for_year(year, gas)
                # Aggregate by sector
                sector_totals: dict[str, float] = {}
                for src in sources:
                    sector = src.get("sector", "unknown")
                    qty = src.get("emissions_quantity", 0)
                    sector_totals[sector] = sector_totals.get(sector, 0) + qty

                for sector, total in sector_totals.items():
                    all_rows.append(
                        {
                            "year": year,
                            "sector": sector,
                            "gas": gas,
                            "emissions_quantity": total,
                        }
                    )

                if sector_totals:
                    logger.info(
                        f"  {year}: {len(sector_totals)} sectors, "
                        f"total {sum(sector_totals.values()):,.0f} tonnes"
                    )

            except Exception as e:
                logger.warning(f"Failed to fetch global totals for {year}: {e}")

            time.sleep(0.3)

        df = pd.DataFrame(all_rows)

        if not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(df)} global emission records")

        return df

    def _fetch_sources_for_year(
        self,
        year: int,
        gas: str,
        countries: list[str] | None = None,
        sectors: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Fetch source-level emissions for a year using /v7/sources.

        Returns a list of dicts with keys: country, sector, year, gas,
        emissions_quantity.
        """
        params: dict[str, Any] = {
            "year": year,
            "gas": gas,
            "limit": limit,
        }
        if countries:
            params["countries"] = ",".join(countries)
        if sectors:
            params["sectors"] = ",".join(sectors)

        data = self._get_with_retry("/sources", params)

        if not isinstance(data, list):
            data = [data] if data else []

        rows = []
        for entry in data:
            rows.append(
                {
                    "country": entry.get("country", "UNK"),
                    "sector": entry.get("sector", "unknown"),
                    "year": entry.get("year", year),
                    "gas": gas,
                    "emissions_quantity": entry.get("emissionsQuantity", 0),
                }
            )

        return rows

    def _get_with_retry(
        self,
        endpoint: str,
        params: dict[str, Any],
        max_retries: int = 3,
    ) -> Any:
        """GET request with exponential backoff."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Climate TRACE rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.RequestException as e:
                status = getattr(e.response, "status_code", None)
                if status in (401, 403, 404):
                    logger.error(
                        f"Climate TRACE {status} error for {endpoint} " f"(params={params})"
                    )
                    return []
                if attempt == max_retries - 1:
                    logger.error(f"Climate TRACE request failed: {e}")
                    raise
                time.sleep(2**attempt)

        return []
