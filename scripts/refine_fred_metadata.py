# scripts/refine_fred_metadata.py

import pandas as pd
import json
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------

TIME_SERIES_PATH =  "data/fred_monthly_master_1994.parquet"
METADATA_PATH = "data/raw/fred_full_metadata.parquet"

OUT_PARQUET ="data/fred_metadata_filtered.parquet"
OUT_JSON = "data/fred_id_to_title.json"


# ----------------------------
# LOAD TIME SERIES (SOURCE OF TRUTH)
# ----------------------------
print("\n[1] Loading time series dataset...")
df = pd.read_parquet(TIME_SERIES_PATH)

valid_ids = set(df.columns)

print(valid_ids)

print(f"✔ Valid series count: {len(valid_ids)}")


# ----------------------------
# LOAD METADATA
# ----------------------------
print("\n[2] Loading metadata...")
meta = pd.read_parquet(METADATA_PATH)

# keep only required columns
meta = meta[['id', 'title']].dropna()

print(f"✔ Raw metadata rows: {len(meta)}")


# ----------------------------
# FILTER METADATA
# ----------------------------
print("\n[3] Filtering metadata to match dataset...")

meta_filtered = meta[meta['id'].isin(valid_ids)].copy()

print(f"✔ Filtered rows: {len(meta_filtered)}")


# ----------------------------
# CLEAN (optional but good)
# ----------------------------
print("\n[4] Cleaning titles...")

meta_filtered['title'] = (
    meta_filtered['title']
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# drop duplicates just in case
meta_filtered = meta_filtered.drop_duplicates(subset='id')


# ----------------------------
# SAVE PARQUET
# ----------------------------
print("\n[5] Saving filtered parquet...")

meta_filtered.to_parquet(OUT_PARQUET, index=False)

print(f"✔ Saved: {OUT_PARQUET}")


# ----------------------------
# SAVE JSON MAPPING (FAST LOOKUP)
# ----------------------------
print("\n[6] Saving id → title mapping...")

mapping = dict(zip(meta_filtered['id'], meta_filtered['title']))

with open(OUT_JSON, "w") as f:
    json.dump(mapping, f)

print(f"✔ Saved: {OUT_JSON}")


# ----------------------------
# DONE
# ----------------------------
print("\n🔥 DONE")