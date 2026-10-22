"""Download USGS 3DEP 1 arc-second DEM tiles covering a lat/lon box (only tiles that exist) and build a VRT.

Usage: python scripts/fetch_3dep_tiles.py <out_dir> --bbox W S E N [--vrt name.vrt]
Tiles: https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/n{lat}w{lon}/USGS_1_n{lat}w{lon}.tif
(tile n30w083 covers lat 29-30, lon -83 to -82).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import requests

BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/{t}/USGS_1_{t}.tif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--vrt", default="dem.vrt")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    w, s, e, n = a.bbox
    got, missing = [], []
    for lat in range(math.floor(s) + 1, math.ceil(n) + 1):      # tile named by its north edge
        for lon in range(math.floor(w), math.ceil(e)):           # tile named by its west edge (positive W value)
            t = f"n{lat:02d}w{abs(lon):03d}"
            out = os.path.join(a.out_dir, f"USGS_1_{t}.tif")
            if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
                got.append(out)
                continue
            url = BASE.format(t=t)
            for k in range(4):
                try:
                    r = requests.get(url, stream=True, timeout=300)
                    if r.status_code == 404:
                        missing.append(t)
                        break
                    r.raise_for_status()
                    with open(out + ".part", "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                    os.replace(out + ".part", out)
                    got.append(out)
                    print("ok", t, f"{os.path.getsize(out) / 1e6:.0f} MB", flush=True)
                    break
                except Exception as ex:  # noqa: BLE001
                    print("retry", t, str(ex)[:80], file=sys.stderr)
                    time.sleep(5 * (k + 1))
    with open(os.path.join(a.out_dir, a.vrt.replace(".vrt", ".meta.json")), "w") as fh:
        json.dump({"source": BASE, "bbox": a.bbox, "tiles": [os.path.basename(g) for g in got], "no_tile": missing,
                   "downloaded": time.strftime("%Y-%m-%d")}, fh, indent=1)
    print(f"{len(got)} tiles, {len(missing)} positions without a tile")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "build_vrt.py"),
                    os.path.join(a.out_dir, a.vrt)] + sorted(got), check=True)


if __name__ == "__main__":
    main()
