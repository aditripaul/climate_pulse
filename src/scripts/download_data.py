"""
Download and cache all climate data sources.

Usage:
    python -m scripts.download_data [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config import settings
from data.pipeline import DataPipeline


def main():
    parser = argparse.ArgumentParser(description="Download ClimatePulse data sources")
    parser.add_argument("--force", action="store_true", help="Force re-download (skip cache)")
    parser.add_argument(
        "--source",
        choices=["noaa", "openaq", "gistemp", "climate_trace", "all"],
        default="all",
        help="Which source to download",
    )
    args = parser.parse_args()

    settings.ensure_directories()

    pipeline = DataPipeline()
    use_cache = not args.force

    if args.source == "all":
        results = pipeline.run_all(use_cache=use_cache)
    elif args.source == "noaa":
        results = {"noaa_ersst": pipeline.run_noaa_ersst(use_cache)}
    elif args.source == "openaq":
        results = {"openaq": pipeline.run_openaq(use_cache)}
    elif args.source == "gistemp":
        results = {"nasa_gistemp": pipeline.run_nasa_gistemp(use_cache)}
    elif args.source == "climate_trace":
        results = {"climate_trace": pipeline.run_climate_trace(use_cache)}

    logger.info("=" * 50)
    logger.info("Download Results:")
    for source, result in results.items():
        status = result.get("status", "unknown")
        prefix = "[SUCCESS]" if status == "success" else "[FAILED]"
        logger.info(f"  {prefix} {source}: {result}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
