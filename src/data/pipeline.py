"""
Data pipeline orchestrator for ClimatePulse.

Coordinates the full ingestion workflow:
  fetch → clean → engineer features → store
"""

from __future__ import annotations

from loguru import logger

from config import settings
from data.processing.cleaning import (
    clean_emissions_data,
    clean_pollution_data,
    clean_sst_data,
    clean_temperature_anomaly,
)
from data.processing.feature_engineering import (
    add_aqi_column,
    add_rolling_features,
    add_seasonal_features,
    align_emissions_temperature,
    compute_nino_indices,
)
from data.sources.climate_trace import ClimateTraceSource
from data.sources.nasa_gistemp import NASAGISTEMPSource
from data.sources.noaa_ersst import NOAAERSSTSource
from data.sources.openaq import OpenAQSource
from data.storage.db import db


class DataPipeline:
    """
    End-to-end data pipeline for ClimatePulse.

    Orchestrates data ingestion from all four sources, cleaning,
    feature engineering, and storage.
    """

    def __init__(self):
        self.noaa = NOAAERSSTSource()
        self.openaq = OpenAQSource()
        self.gistemp = NASAGISTEMPSource()
        self.climate_trace = ClimateTraceSource()

    def run_all(self, use_cache: bool = True) -> dict:
        """
        Execute the full pipeline for all data sources.

        Args:
            use_cache: If True, skip sources that already have cached data.

        Returns:
            Dictionary with summary statistics for each source.
        """
        settings.ensure_directories()
        results = {}

        logger.info("=" * 60)
        logger.info("ClimatePulse Data Pipeline — Starting")
        logger.info("=" * 60)

        # 1. NOAA ERSST
        try:
            results["noaa_ersst"] = self.run_noaa_ersst(use_cache)
        except Exception as e:
            logger.error(f"NOAA ERSST pipeline failed: {e}")
            results["noaa_ersst"] = {"status": "error", "error": str(e)}

        # 2. OpenAQ
        try:
            results["openaq"] = self.run_openaq(use_cache)
        except Exception as e:
            logger.error(f"OpenAQ pipeline failed: {e}")
            results["openaq"] = {"status": "error", "error": str(e)}

        # 3. NASA GISTEMP
        try:
            results["nasa_gistemp"] = self.run_nasa_gistemp(use_cache)
        except Exception as e:
            logger.error(f"NASA GISTEMP pipeline failed: {e}")
            results["nasa_gistemp"] = {"status": "error", "error": str(e)}

        # 4. Climate TRACE
        try:
            results["climate_trace"] = self.run_climate_trace(use_cache)
        except Exception as e:
            logger.error(f"Climate TRACE pipeline failed: {e}")
            results["climate_trace"] = {"status": "error", "error": str(e)}

        logger.info("=" * 60)
        logger.info("ClimatePulse Data Pipeline — Complete")
        logger.info("=" * 60)

        return results

    def run_noaa_ersst(self, use_cache: bool = True) -> dict:
        """Pipeline for NOAA ERSST v5 data."""
        logger.info("--- NOAA ERSST v5 Pipeline ---")

        # Fetch
        sst_df = self.noaa.fetch(use_cache=use_cache)
        sst_df = clean_sst_data(sst_df)

        # Niño 3.4 index (contains the correct anomaly computed vs climatology)
        nino34_df = self.noaa.fetch_nino34(use_cache=use_cache)

        # Niño indices from gridded data
        nino_indices = compute_nino_indices(sst_df)

        # We must use the true anomaly as our target variable
        if "nino34" in nino_indices.columns:
            nino_indices = nino_indices.drop(columns=["nino34"])

        nino_indices = nino_indices.merge(
            nino34_df[["time", "nino34_anomaly"]].rename(columns={"nino34_anomaly": "nino34"}),
            on="time",
            how="left",
        )

        nino_indices = add_rolling_features(nino_indices, "nino34")
        nino_indices = add_seasonal_features(nino_indices)

        # Store processed
        processed_path = settings.data.processed_dir / "nino_features.parquet"
        db.save_parquet(nino_indices, processed_path)

        # Register in metadata
        db.register_source(
            name="noaa_ersst_v5",
            description="NOAA ERSST v5 Sea Surface Temperature (2000–present)",
            record_count=len(sst_df),
            cache_path=str(processed_path),
        )

        return {
            "status": "success",
            "sst_records": len(sst_df),
            "nino34_months": len(nino34_df),
            "feature_months": len(nino_indices),
        }

    def run_openaq(self, use_cache: bool = True) -> dict:
        """Pipeline for OpenAQ v3 data."""
        logger.info("--- OpenAQ v3 Pipeline ---")

        # Fetch PM2.5 data
        pm25_df = self.openaq.fetch_measurements(
            parameter="pm25",
            countries=settings.data.focus_countries,
            date_from=f"{settings.data.start_year}-01-01",
            date_to=f"{settings.data.end_year}-12-31",
            limit=settings.data.openaq_limit,
            max_locations=settings.data.openaq_max_locations,
            use_cache=use_cache,
        )

        if not pm25_df.empty:
            pm25_df = clean_pollution_data(pm25_df)
            pm25_df = add_aqi_column(pm25_df)

        # Fetch locations for graph construction and persist for downstream use
        locations_df = self.openaq.fetch_locations()
        locations_path = settings.data.processed_dir / "openaq" / "locations.parquet"
        db.save_parquet(locations_df, locations_path)

        # Store
        db.register_source(
            name="openaq_v3",
            description="OpenAQ v3 Air Quality Measurements",
            record_count=len(pm25_df),
            cache_path=str(settings.data.processed_dir / "openaq"),
        )

        return {
            "status": "success",
            "pm25_records": len(pm25_df),
            "locations": len(locations_df),
        }

    def run_nasa_gistemp(self, use_cache: bool = True) -> dict:
        """Pipeline for NASA GISTEMP v4 data."""
        logger.info("--- NASA GISTEMP v4 Pipeline ---")

        # Global mean anomaly
        global_mean = self.gistemp.fetch_global_mean(use_cache=use_cache)
        global_mean = clean_temperature_anomaly(global_mean)
        global_mean = add_rolling_features(global_mean, "global_temp_anomaly")
        global_mean = add_seasonal_features(global_mean)

        # Store
        processed_path = settings.data.processed_dir / "gistemp_features.parquet"
        db.save_parquet(global_mean, processed_path)

        db.register_source(
            name="nasa_gistemp_v4",
            description="NASA GISTEMP v4 Global Temperature Anomaly",
            record_count=len(global_mean),
            cache_path=str(processed_path),
        )

        return {
            "status": "success",
            "monthly_records": len(global_mean),
        }

    def run_climate_trace(self, use_cache: bool = True) -> dict:
        """Pipeline for Climate TRACE emissions data."""
        logger.info("--- Climate TRACE Pipeline ---")

        # Global totals (or focused country emissions)
        if settings.data.focus_countries:
            emissions_df = self.climate_trace.fetch_country_emissions(
                countries=settings.data.focus_countries,
                use_cache=use_cache,
            )
        else:
            emissions_df = self.climate_trace.fetch_global_totals(use_cache=use_cache)

        if not emissions_df.empty:
            emissions_df = clean_emissions_data(emissions_df)

        db.register_source(
            name="climate_trace",
            description="Climate TRACE Global Emissions Inventory",
            record_count=len(emissions_df),
            cache_path=str(settings.data.processed_dir / "climate_trace"),
        )

        return {
            "status": "success",
            "emission_records": len(emissions_df),
        }

    def build_regression_dataset(self) -> dict:
        """
        Build the aligned emissions–temperature dataset for regression.

        Requires GISTEMP and Climate TRACE data to be already fetched.
        """
        logger.info("Building regression dataset...")

        global_mean = db.load_parquet(settings.data.processed_dir / "gistemp_features.parquet")

        if settings.data.focus_countries:
            filename = f"emissions_{'_'.join(settings.data.focus_countries)}_{'co2e_100yr'}.parquet"
        else:
            filename = "global_totals_co2e_100yr.parquet"

        emissions = db.load_parquet(settings.data.processed_dir / "climate_trace" / filename)

        aligned = align_emissions_temperature(emissions, global_mean)
        aligned_path = settings.data.processed_dir / "regression_dataset.parquet"
        db.save_parquet(aligned, aligned_path)

        return {
            "status": "success",
            "years": len(aligned),
            "path": str(aligned_path),
        }
