"""
Export TFT lead-1 Nino 3.4 forecasts for Neural Weather Model conditioning.

Uses the trained TFT in the checkpoint directory and the test split, whose
forecasts are out of sample. Writes datasets/processed/nino34_tft_lead1.parquet,
which the NWM reads when `nwm.conditioning` is "forecast".

Usage:
    PYTHONPATH=src python src/scripts/export_tft_forecasts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from models.el_nino.dataset import (  # noqa: E402
    create_tft_dataset,
    prepare_tft_dataframe,
    split_tft_dataset,
)
from models.el_nino.tft_model import (  # noqa: E402
    export_lead1_forecasts,
    find_best_tft_checkpoint,
    load_tft_model,
)
from models.nwm.dataset import nino_conditioning_path  # noqa: E402


def main():
    model = load_tft_model(find_best_tft_checkpoint(settings.model.checkpoint_dir / "el_nino" / "tft"))
    df = prepare_tft_dataframe()
    _, _, test_ds = split_tft_dataset(create_tft_dataset(df), df)
    export_lead1_forecasts(model, test_ds, df, nino_conditioning_path("forecast"))


if __name__ == "__main__":
    main()
