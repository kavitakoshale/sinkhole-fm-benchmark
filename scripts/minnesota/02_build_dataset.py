"""Assemble data/processed/minnesota/dataset.parquet = points.gpkg metadata + shared-schema features (scripts/shared_features.py).

Run first:
  python scripts/shared_features.py data/processed/minnesota/points.gpkg data/raw/usgs_3dep_1arcsec/minnesota_dem.vrt \
         data/raw/esa_worldcover/worldcover_minnesota_x4.vrt data/processed/minnesota/features_shared.parquet
"""
import os

import geopandas as gpd
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = f"{ROOT}/data/processed/minnesota"
META = ["pid", "ref_num", "label", "event_year", "accuracy", "true_sink", "field_checked", "county", "neg_strategy", "lon", "lat",
        "x", "y", "block_id", "fold_spatial", "fold_random", "split_temporal"]

pts = pd.DataFrame(gpd.read_file(f"{PROC}/points.gpkg").drop(columns="geometry"))[META]
feat = pd.read_parquet(f"{PROC}/features_shared.parquet")
df = pts.merge(feat, on="pid", how="left", validate="1:1")
df = df[df.elev_m.notna()].reset_index(drop=True)
print("dropped (no DEM):", len(pts) - len(df))
df.to_parquet(f"{PROC}/dataset.parquet", index=False)
print("dataset", df.shape, "->", f"{PROC}/dataset.parquet")
print(df.groupby("neg_strategy").size().to_dict())
print(df.drop(columns=META).describe().T[["mean", "min", "max"]].round(3).to_string())
