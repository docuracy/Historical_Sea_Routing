import logging
import argparse
from pathlib import Path

import pandas as pd

from process.config import AOIS

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
AOI = AOIS[1]

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Convert .parquet files to .csv if missing (or force overwrite).")
parser.add_argument("--force", action="store_true", help="Force overwrite existing .csv files.")
args = parser.parse_args()

# --- Directories ---
docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / AOI["name"]
geo_output_directory.mkdir(parents=True, exist_ok=True)

# --- Convert .parquet to .csv ---
for parquet_file in geo_output_directory.glob("*.parquet"):
    csv_file = parquet_file.with_suffix(".csv")
    if args.force or not csv_file.exists():
        logger.info(f"{'Overwriting' if csv_file.exists() else 'Converting'} {parquet_file.name} to CSV...")
        try:
            df = pd.read_parquet(parquet_file)
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved CSV: {csv_file.name}")
        except Exception as e:
            logger.error(f"Failed to convert {parquet_file.name}: {e}")
    else:
        logger.info(f"CSV already exists: {csv_file.name} — skipping.")
