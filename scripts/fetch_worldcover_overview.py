"""Fetch ESA WorldCover 2021 (v200) tiles for a lat/lon box at a decimated resolution via COG overviews.

Reads only the overview level (default factor 4 -> ~40 m), so each 3x3-degree tile is ~80 MB uncompressed
instead of 1.3 GB. Mode resampling is implicit in the stored overviews (ESA builds them with 'mode').

Usage:
    python scripts/fetch_worldcover_overview.py <out_dir> --bbox -88 24 -79 31.5 --factor 4
"""
import argparse
import math
import os

import rasterio
from rasterio.enums import Resampling

BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"


def tile_names(w, s, e, n):
    for lat in range(int(math.floor(s / 3) * 3), int(math.ceil(n / 3) * 3), 3):
        for lon in range(int(math.floor(w / 3) * 3), int(math.ceil(e / 3) * 3), 3):
            yield f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--bbox", nargs=4, type=float, default=[-88, 24, -79, 31.5])
    ap.add_argument("--factor", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for t in tile_names(*a.bbox):
        out = os.path.join(a.out_dir, f"worldcover2021_{t}_x{a.factor}.tif")
        if os.path.exists(out):
            continue
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open("/vsicurl/" + BASE.format(t)) as r:
                h, w = r.height // a.factor, r.width // a.factor
                arr = r.read(1, out_shape=(h, w), resampling=Resampling.nearest)
                prof = r.profile
                prof.update(height=h, width=w, transform=r.transform * r.transform.scale(a.factor, a.factor),
                            compress="deflate", tiled=True, blockxsize=512, blockysize=512, driver="GTiff")
                with rasterio.open(out, "w", **prof) as dst:
                    dst.write(arr, 1)
            print("ok", t, arr.shape)
        except Exception as e:  # noqa: BLE001
            print("skip", t, str(e)[:80])


if __name__ == "__main__":
    main()
