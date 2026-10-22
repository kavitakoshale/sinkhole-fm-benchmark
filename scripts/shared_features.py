"""State-agnostic feature schema for the multi-state benchmark: only layers that exist for every region.

  terrain   21 features from a 30 m DEM (3DEP 1 arc-sec here; Copernicus GLO-30 outside the US): elevation, slope,
            curvature, TPI / relief / elev_std / depth_below_max / height_above_min / slope_mean at 150, 300, 600 m.
            Identical definitions to scripts/florida/02_extract_features.py::terrain.
  dep       DEM-derived closed depressions, replacing the Florida FGS "ctds" mapped polygons: a priority-flood fill of a
            5 km window around each point (window edges act as outlets; skimage grayscale reconstruction), depression
            cells = fill depth > DEP_MIN_M. Features: fill depth at the point, inside flag, distance to the nearest
            depression cell, and depression count / area fraction / max depth within 500, 1000, 2000 m.
  lc        ESA WorldCover 2021 class at the point and class fractions within 250 / 1000 m (as Florida).
  ppt       PRISM 1991-2020 annual precipitation normal (CONUS; CHIRPS would be the global stand-in).

Functions take lon/lat arrays plus raster paths and return DataFrames; `extract(points_gpkg, dem_vrt, lc_vrt, out_parquet)`
runs everything in parallel over points (joblib) and writes `<pid, features...>`.
"""
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from pyproj import Transformer
from rasterio.windows import Window
from scipy import ndimage
from skimage.morphology import reconstruction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = f"{ROOT}/data/raw"
PRISM = f"{RAW}/prism/ppt_annual/prism_ppt_us_30s_2020_avg_30y.tif"
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}
DEP_MIN_M = 0.5      # fill depth that counts as a depression (3DEP 1" vertical noise is a few dm)
DEP_HALF_PX = 83     # 5 km half window at 30 m
TERRAIN_HALF_PX = 20
M_PER_ARCSEC = 30.87


def _window(r, lon, lat, R, fill):
    row, col = r.index(lon, lat)
    w = r.read(1, window=Window(col - R, row - R, 2 * R + 1, 2 * R + 1), boundless=True, fill_value=fill)
    return w


def _dxdy(r, lat):
    dy = r.res[1] * 3600 * M_PER_ARCSEC
    return dy * np.cos(np.radians(lat)), dy


def terrain_one(w, dx, dy, R=TERRAIN_HALF_PX):
    c = w[R, R]
    if np.isnan(c):
        return {}
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    dist = np.hypot(xx * dx, yy * dy)
    z3 = w[R - 1:R + 2, R - 1:R + 2]
    gx = ((z3[0, 2] + 2 * z3[1, 2] + z3[2, 2]) - (z3[0, 0] + 2 * z3[1, 0] + z3[2, 0])) / (8 * dx)
    gy = ((z3[2, 0] + 2 * z3[2, 1] + z3[2, 2]) - (z3[0, 0] + 2 * z3[0, 1] + z3[0, 2])) / (8 * dy)
    gyw, gxw = np.gradient(w, dy, dx)
    slope_w = np.degrees(np.arctan(np.hypot(gxw, gyw)))
    d2x = (z3[1, 2] - 2 * c + z3[1, 0]) / dx ** 2
    d2y = (z3[2, 1] - 2 * c + z3[0, 1]) / dy ** 2
    f = {"elev_m": c, "slope_deg": float(np.degrees(np.arctan(np.hypot(gx, gy)))), "laplacian_curv": float(d2x + d2y)}
    for rad in (150, 300, 600):
        m = dist <= rad
        vals = w[m]
        f[f"tpi_{rad}m"] = c - np.nanmean(vals)
        f[f"relief_{rad}m"] = np.nanmax(vals) - np.nanmin(vals)
        f[f"elev_std_{rad}m"] = np.nanstd(vals)
        f[f"depth_below_max_{rad}m"] = np.nanmax(vals) - c
        f[f"height_above_min_{rad}m"] = c - np.nanmin(vals)
        f[f"slope_mean_{rad}m"] = np.nanmean(slope_w[m])
    return f


def depressions_one(w, dx, dy, R=DEP_HALF_PX):
    """Priority-flood style fill via grayscale reconstruction (seed = window with the border kept, interior raised)."""
    if np.isnan(w[R, R]):
        return {}
    z = w.copy()
    nanm = np.isnan(z)
    z[nanm] = np.nanmax(z) + 1e3 if (~nanm).any() else 0  # nodata acts as a wall
    seed = np.full_like(z, z.max())
    seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = z[0, :], z[-1, :], z[:, 0], z[:, -1]
    filled = reconstruction(seed, z, method="erosion")
    depth = filled - z
    depth[nanm] = 0
    dep = depth > DEP_MIN_M
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    dist = np.hypot(xx * dx, yy * dy)
    f = {"dep_fill_depth_m": float(depth[R, R]), "dep_inside": int(dep[R, R])}
    f["dep_dist_m"] = float(dist[dep].min()) if dep.any() else float(dist.max())
    lab, _ = ndimage.label(dep)
    for rad in (500, 1000, 2000):
        m = dist <= rad
        f[f"dep_count_{rad}m"] = int(len(np.unique(lab[m & dep])))
        f[f"dep_area_frac_{rad}m"] = float(dep[m].mean())
        f[f"dep_max_depth_{rad}m"] = float(depth[m].max())
    return f


def landcover_one(w, dx, dy, R):
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    dist = np.hypot(xx * dx, yy * dy)
    f = {"lc_class": WC.get(int(w[R, R]), "other")}
    for rad in (250, 1000):
        m = (dist <= rad) & (w > 0)
        tot = max(m.sum(), 1)
        for k, name in WC.items():
            f[f"lc_{name}_frac_{rad}m"] = float(((w == k) & m).sum() / tot)
    return f


def _chunk(lons, lats, dem_vrt, lc_vrt):
    rows = []
    with rasterio.open(dem_vrt) as dem, rasterio.open(lc_vrt) as lc:
        nod = dem.nodata
        lc_R = int(round(1000 / (lc.res[0] * 3600 * M_PER_ARCSEC))) + 1
        for lo, la in zip(lons, lats):
            dx, dy = _dxdy(dem, la)
            w = _window(dem, lo, la, DEP_HALF_PX, nod).astype("float64")
            w[w == nod] = np.nan
            R = DEP_HALF_PX
            f = terrain_one(w[R - TERRAIN_HALF_PX:R + TERRAIN_HALF_PX + 1, R - TERRAIN_HALF_PX:R + TERRAIN_HALF_PX + 1], dx, dy)
            f.update(depressions_one(w, dx, dy))
            ldx, ldy = _dxdy(lc, la)
            f.update(landcover_one(_window(lc, lo, la, lc_R, 0), ldx, ldy, lc_R))
            rows.append(f)
    return rows


def sample_raster(path, lons, lats):
    with rasterio.open(path) as r:
        tr = Transformer.from_crs(4326, r.crs, always_xy=True)
        px, py = tr.transform(lons, lats)
        vals = np.array([v[0] for v in r.sample(zip(px, py), masked=True)], dtype="float64")
        if r.nodata is not None:
            vals[vals == r.nodata] = np.nan
    return vals


def localize(vrt, cache_dir=None):
    """Materialize a (network-share) VRT as one tiled, uncompressed GeoTIFF on local disk, so that tens of thousands of
    small window reads hit the page cache instead of the share. Returns the local path (reused if present)."""
    cache_dir = cache_dir or os.environ.get("SHARED_FEATURES_CACHE", "/tmp/sinkhole_fm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, os.path.basename(vrt).replace(".vrt", ".tif"))
    stamp = f"{os.path.getsize(vrt)}:{os.path.getmtime(vrt)}"  # share clocks can be ahead of ours, so never compare mtimes directly
    marker = out + ".done"
    if os.path.exists(out) and os.path.exists(marker) and open(marker).read() == stamp:
        return out
    tmp = f"{out}.tmp{os.getpid()}"  # private temp file + atomic rename: concurrent callers never see a half-written raster
    with rasterio.open(vrt) as r:
        prof = r.profile | {"driver": "GTiff", "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": None, "BIGTIFF": "IF_SAFER"}
        with rasterio.open(tmp, "w", **prof) as w:
            for row in range(0, r.height, 1024):
                win = Window(0, row, r.width, min(1024, r.height - row))
                w.write(r.read(window=win), window=win)
    os.replace(tmp, out)
    with open(marker, "w") as fh:
        fh.write(stamp)
    print("localized", vrt, "->", out, f"{os.path.getsize(out) / 1e9:.2f} GB", flush=True)
    return out


def extract(points_gpkg, dem_vrt, lc_vrt, out_parquet, n_jobs=32, chunk=200):
    import geopandas as gpd
    dem_vrt, lc_vrt = localize(dem_vrt), localize(lc_vrt)
    pts = gpd.read_file(points_gpkg)
    lons, lats = pts.lon.to_numpy(), pts.lat.to_numpy()
    idx = [range(i, min(i + chunk, len(pts))) for i in range(0, len(pts), chunk)]
    print(f"{len(pts)} points, {len(idx)} chunks, {n_jobs} jobs", flush=True)
    parts = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_chunk)(lons[list(i)], lats[list(i)], dem_vrt, lc_vrt) for i in idx)
    df = pd.DataFrame([r for p in parts for r in p], index=pts.index)
    df["ppt_annual_mm"] = sample_raster(PRISM, lons, lats)
    df.insert(0, "pid", pts.pid.values)
    df.to_parquet(out_parquet, index=False)
    print("wrote", out_parquet, df.shape, "| NaN elev:", int(df.elev_m.isna().sum()))
    return df


if __name__ == "__main__":
    extract(*sys.argv[1:5])
