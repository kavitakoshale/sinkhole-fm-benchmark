"""Replicate the Zhu et al. (2020, J. Hydrology 588:125049) depression dataset for the Kentucky Bluegrass.

1. Sink-fill the KyFromAbove Phase-1 5-ft LiDAR DEM (EPSG:3089, Z in US ft) chunk by chunk
   (morphological reconstruction by erosion == priority-flood fill), with overlap.
2. Depressions = connected components of (filled - dem) > MIN_DEPTH_FT, area >= MIN_AREA_FT2.
3. For each depression compute the 10 morphometric variables of Zhu et al. (2020):
     perimeter, area, compactness (A / area of min circumscribing circle), max depth, mean depth, depth std,
     depth sum, mean slope and slope std in a 9 m (30 ft) ring around the polygon, depth index Dmax / sqrt(A/pi)
   Lengths reported in metres (paper units).
4. Label = 1 if the depression overlaps a KGS LiDAR sinkhole polygon (the KGS inventory was produced by the same
   group with the Zhu et al. 2014 method + visual inspection), else 0.

Usage: python scripts/kentucky/01_extract_depressions.py --county Bourbon [--min-depth-ft 0.5]
Output: data/processed/kentucky/depressions_<county>.gpkg
"""
import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.windows import from_bounds
from scipy import ndimage as ndi
from shapely import minimum_bounding_radius
from shapely.geometry import shape
from skimage.morphology import reconstruction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = f"{ROOT}/data/raw"
OUT = f"{ROOT}/data/processed/kentucky"
FT = 0.3048006096  # US survey ft -> m
CHUNK_FT = 20_000  # 4000 px core
PAD_FT = 2_500  # 500 px overlap so depressions crossing the core edge are complete


def fill_sinks(dem):
    """Priority-flood equivalent: reconstruction by erosion seeded from the border. NaN treated as outlet."""
    valid = np.isfinite(dem)
    lo = np.nanmin(dem) - 1 if valid.any() else 0
    d = np.where(valid, dem, lo)
    seed = d.copy()
    seed[1:-1, 1:-1] = d.max()
    seed[~valid] = lo
    filled = reconstruction(seed, d, method="erosion")
    filled[~valid] = np.nan
    return filled


def slope_deg(dem, cell):
    gy, gx = np.gradient(dem, cell, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def process_chunk(src_files, bounds, core, county_geom, args):
    srcs = [rasterio.open(f) for f in src_files]
    try:
        arr, tr = merge(srcs, bounds=bounds, nodata=np.nan, dtype="float32")
    finally:
        for s in srcs:
            s.close()
    dem = arr[0].astype("float64")
    if np.isfinite(dem).sum() < 1000:
        return []
    cell = abs(tr.a)  # ft
    filled = fill_sinks(dem)
    depth = filled - dem
    mask = np.nan_to_num(depth) > args.min_depth_ft
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    npx = ndi.sum(mask, lab, idx)
    keep = idx[npx * cell * cell >= args.min_area_ft2]
    slope = slope_deg(np.where(np.isfinite(dem), dem, np.nanmean(dem)), cell)
    dmax = ndi.maximum(depth, lab, idx)
    dmean = ndi.mean(depth, lab, idx)
    dstd = ndi.standard_deviation(depth, lab, idx)
    dsum = ndi.sum(depth, lab, idx)

    lab_k = np.where(np.isin(lab, keep), lab, 0).astype("int32")
    rows = []
    ring_px = int(round(30 / cell))  # 9 m ~ 30 ft ring
    objs = ndi.find_objects(lab_k)
    for geom, val in shapes(lab_k, mask=lab_k > 0, transform=tr, connectivity=8):
        val = int(val)
        poly = shape(geom)
        rp = poly.representative_point()
        if not core.contains(rp) or not county_geom.contains(rp):
            continue
        sl = objs[val - 1]
        r0, r1 = max(sl[0].start - ring_px - 1, 0), min(sl[0].stop + ring_px + 1, lab.shape[0])
        c0, c1 = max(sl[1].start - ring_px - 1, 0), min(sl[1].stop + ring_px + 1, lab.shape[1])
        sub = lab_k[r0:r1, c0:c1] == val
        ring = ndi.binary_dilation(sub, iterations=ring_px) & ~sub
        rs = slope[r0:r1, c0:c1][ring]
        A = poly.area
        rad = minimum_bounding_radius(poly)
        i = val - 1
        rows.append({
            "geometry": poly,
            "perimeter_m": poly.length * FT,
            "area_m2": A * FT * FT,
            "compactness": A / (np.pi * rad * rad) if rad > 0 else 1.0,
            "depth_max_m": dmax[i] * FT,
            "depth_mean_m": dmean[i] * FT,
            "depth_std_m": dstd[i] * FT,
            "depth_sum_m": dsum[i] * FT,  # sum of per-cell depths (x cell area = volume), as in the paper
            "ring_slope_mean_deg": float(rs.mean()) if rs.size else np.nan,
            "ring_slope_std_deg": float(rs.std()) if rs.size else np.nan,
            "depth_index": (dmax[i] * FT) / np.sqrt(A * FT * FT / np.pi),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--county", required=True)
    ap.add_argument("--min-depth-ft", type=float, default=0.5)
    ap.add_argument("--min-area-ft2", type=float, default=200.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    d = f"{RAW}/kentucky_kyfromabove"
    tiles = pd.read_csv(f"{d}/tiles_bluegrass4.csv")
    tiles = tiles[tiles.NAME == args.county]
    grid = gpd.read_file(f"{d}/kyfromabove_phase1_dem_5k_grid.gpkg").to_crs(3089)
    grid = grid[grid.tilename.isin(tiles.tilename)]
    grid["path"] = grid.aws_url.str.split("/").str[-1].map(lambda f: f"{d}/dem_phase1/{f}")
    grid = grid[grid.path.map(os.path.exists)]
    county = gpd.read_file(f"{d}/bluegrass4_counties.gpkg").to_crs(3089)
    cgeom = county[county.NAME == args.county].geometry.iloc[0]
    print(args.county, len(grid), "tiles available")

    minx, miny, maxx, maxy = cgeom.bounds
    rows = []
    from shapely.geometry import box
    xs = np.arange(minx, maxx, CHUNK_FT)
    ys = np.arange(miny, maxy, CHUNK_FT)
    for k, (x0, y0) in enumerate([(x, y) for x in xs for y in ys]):
        core = box(x0, y0, x0 + CHUNK_FT, y0 + CHUNK_FT)
        if not core.intersects(cgeom):
            continue
        b = (x0 - PAD_FT, y0 - PAD_FT, x0 + CHUNK_FT + PAD_FT, y0 + CHUNK_FT + PAD_FT)
        files = grid[grid.intersects(box(*b))].path.tolist()
        if not files:
            continue
        r = process_chunk(files, b, core, cgeom, args)
        rows += r
        print(f"  chunk {k + 1}/{len(xs) * len(ys)}: {len(r)} depressions (total {len(rows)})", flush=True)

    dep = gpd.GeoDataFrame(rows, geometry="geometry", crs=3089)
    sk = gpd.read_file(f"{RAW}/kentucky_kgs/lidar_sinkholes_bluegrass4.gpkg").to_crs(3089)
    sk = sk[sk.County.str.lower() == args.county.lower()]
    hit = gpd.sjoin(dep[["geometry"]], sk[["geometry"]], predicate="intersects", how="inner").index.unique()
    dep["label"] = 0
    dep.loc[hit, "label"] = 1
    dep["county"] = args.county
    dep["dep_id"] = [f"{args.county[:3].upper()}_{i}" for i in range(len(dep))]
    c = dep.geometry.centroid.to_crs(4326)
    dep["lon"], dep["lat"] = c.x, c.y
    out = f"{OUT}/depressions_{args.county}.gpkg"
    dep.to_file(out, driver="GPKG")
    print(f"{args.county}: {len(dep)} depressions, {dep.label.sum()} sinkholes ({dep.label.mean():.1%}); "
          f"KGS polygons in county: {len(sk)} -> {out}")


if __name__ == "__main__":
    main()
