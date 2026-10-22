"""Download KyFromAbove Phase-1 5-ft DEM tiles (AWS open data, EPSG:3089, Z in US ft) for a set of counties, and
write the tile index + county/tile table that scripts/kentucky/01_extract_depressions.py expects.

Tile naming N{row}E{col}: 5000 x 5000 ft tiles in EPSG:3089 with
    x0 = 3,775,000 + 5000*col,  y_top = 4,365,000 - 5000*row   (checked against the .tfw world files)

Usage: python scripts/fetch_kyfromabove.py [--counties Bourbon Woodford Jessamine Scott] [--pad-ft 2500] [--workers 8]
Outputs under data/raw/kentucky_kyfromabove/: phase1_listing.csv, kyfromabove_phase1_dem_5k_grid.gpkg,
tiles_<tag>.csv (tag = bluegrass4 for the default counties), dem_phase1/<tile>.tif
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = f"{ROOT}/data/raw/kentucky_kyfromabove"
S3 = "s3://kyfromabove/elevation/DEM/Phase1/"
HTTP = "https://kyfromabove.s3.amazonaws.com/elevation/DEM/Phase1/"
TILE = 5000.0


def listing():
    p = f"{D}/phase1_listing.csv"
    if not os.path.exists(p):
        out = subprocess.run(["aws", "s3", "ls", "--no-sign-request", S3], capture_output=True, text=True, check=True).stdout
        rows = [(l.split()[-1], int(l.split()[2])) for l in out.splitlines() if l.strip().endswith("cog.tif")]
        pd.DataFrame(rows, columns=["file", "bytes"]).to_csv(p, index=False)
    return pd.read_csv(p)


def grid(lst):
    m = lst.file.str.extract(r"^N(\d+)E(\d+)_")
    lst = lst.assign(row=m[0].astype(int), col=m[1].astype(int), tilename=lst.file.str.replace("_cog.tif", "", regex=False))
    lst["geometry"] = [box(3_775_000 + TILE * c, 4_365_000 - TILE * (r + 1), 3_775_000 + TILE * (c + 1), 4_365_000 - TILE * r)
                       for r, c in zip(lst.row, lst.col)]
    lst["aws_url"] = HTTP + lst.file
    return gpd.GeoDataFrame(lst, geometry="geometry", crs=3089)


def fetch(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 10_000:
        return "have"
    for k in range(4):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(out + ".part", "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            os.replace(out + ".part", out)
            return "ok"
        except Exception as e:  # noqa: BLE001
            time.sleep(5 * (k + 1))
            err = e
    return f"FAIL {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counties", nargs="*", default=["Bourbon", "Woodford", "Jessamine", "Scott"])
    ap.add_argument("--tag", default=None, help="name for tiles_<tag>.csv (default bluegrass4 for the 4 default counties)")
    ap.add_argument("--pad-ft", type=float, default=2500.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tag = a.tag or ("bluegrass4" if sorted(a.counties) == ["Bourbon", "Jessamine", "Scott", "Woodford"] else "_".join(a.counties))
    os.makedirs(f"{D}/dem_phase1", exist_ok=True)

    g = grid(listing())
    gp = f"{D}/kyfromabove_phase1_dem_5k_grid.gpkg"
    if not os.path.exists(gp):
        g.drop(columns=["row", "col"]).to_file(gp, driver="GPKG")
    # verify the naming formula on one tile's world file
    f0 = g.iloc[0]
    tfw = requests.get(f0.aws_url.replace("_cog.tif", "_cog.tfw"), timeout=60).text.split()
    x0, y0 = float(tfw[4]), float(tfw[5])
    b = f0.geometry.bounds
    assert abs(x0 - b[0]) < 1e-3 and abs(y0 - b[3]) < TILE + 1, (f0.file, x0, y0, b)

    counties = gpd.read_file(f"{ROOT}/data/raw/tiger2023_us_county.gpkg")
    counties = counties[(counties.STATEFP == "21") & counties.NAME.isin(a.counties)].to_crs(3089)
    assert len(counties) == len(a.counties), counties.NAME.tolist()
    rows = []
    for _, c in counties.iterrows():
        hit = g[g.intersects(c.geometry.buffer(a.pad_ft))]
        rows += [{"NAME": c.NAME, "tilename": t} for t in hit.tilename]
    tiles = pd.DataFrame(rows)
    tiles.to_csv(f"{D}/tiles_{tag}.csv", index=False)
    need = g[g.tilename.isin(tiles.tilename)]
    print(f"{tag}: {len(tiles)} county-tile pairs, {len(need)} unique tiles, {need.bytes.sum() / 1e9:.1f} GB", flush=True)
    if a.dry_run:
        return
    jobs = [(u, f"{D}/dem_phase1/{f}") for u, f in zip(need.aws_url, need.file)]
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for (u, out), res in zip(jobs, ex.map(lambda j: fetch(*j), jobs)):
            done += 1
            if res.startswith("FAIL") or done % 50 == 0:
                print(f"  {done}/{len(jobs)} {os.path.basename(out)} {res}", flush=True)
    with open(f"{D}/dem_phase1/meta.json", "w") as fh:
        json.dump({"source": S3, "counties": a.counties, "n_tiles": len(need), "pad_ft": a.pad_ft,
                   "crs": "EPSG:3089 (ft)", "downloaded": time.strftime("%Y-%m-%d")}, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main()
