"""Extract conditioning factors for every point in data/processed/florida/points.gpkg.

Feature groups (each cached to data/processed/florida/features/<group>.parquet; rerun with --groups):
  terrain     USGS 3DEP 1" DEM: elevation, slope, relief, TPI, curvature, local-minimum depth
  landcover   ESA WorldCover 2021 (40 m overview): class at point + class fractions within 250 m / 1 km
  hydro       FGS FAVA rasters (Mendeley, Youn et al.): SAS depth-to-water, IAS thickness, head difference;
              Floridan recharge/discharge class; DRASTIC Floridan index
  geology     FGS stratigraphic unit/lithology/age, environmental geology, sinkhole-type area, geomorphic district
  depress     FGS closed topographic depressions (ctds_2004): inside, distance, count & area fraction within 1 km
  proximity   distance to FGS swallets, springs, phosphate mined units
  soil        POLARIS 30 m (remote COG sampling): ksat 0-5/30-60 cm, clay & sand 0-5 cm
  climate     PRISM 1991-2020 annual precipitation normal
  expert      FGS sinkhole favorability (WofE) class -- used ONLY as an expert baseline score, never a feature

Final merged table: data/processed/florida/dataset.parquet
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = f"{ROOT}/data/raw"
PROC = f"{ROOT}/data/processed/florida"
FEAT = f"{PROC}/features"
CRS = 3086


def sample_raster(path, xs, ys, src_crs, masked=True):
    """Point-sample band 1; returns float array with NaN for nodata/masked/outside."""
    with rasterio.open(path) as r:
        tr = Transformer.from_crs(src_crs, r.crs, always_xy=True)
        px, py = tr.transform(xs, ys)
        vals = np.array([v[0] for v in r.sample(zip(px, py), masked=masked)], dtype="float64")
        if masked:
            m = np.array([np.ma.getmaskarray(v)[0] for v in r.sample(zip(px, py), masked=True)])
            vals[m] = np.nan
        if r.nodata is not None:
            vals[vals == r.nodata] = np.nan
    return vals


# ---------------------------------------------------------------- terrain
def terrain(pts):
    path = f"{RAW}/usgs_3dep_1arcsec/florida_dem.vrt"
    R = 20  # half window in px (~600 m)
    with rasterio.open(path) as r:
        tr = Transformer.from_crs(CRS, r.crs, always_xy=True)
        lon, lat = tr.transform(pts.x.values, pts.y.values)
        nod = r.nodata
        rows = []
        for lo, la in zip(lon, lat):
            row, col = r.index(lo, la)
            w = r.read(1, window=Window(col - R, row - R, 2 * R + 1, 2 * R + 1), boundless=True, fill_value=nod).astype("float64")
            w[w == nod] = np.nan
            dy = 30.87  # m per arc-second (N-S)
            dx = 30.87 * np.cos(np.radians(la))
            c = w[R, R]
            if np.isnan(c):
                rows.append({})
                continue
            yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
            dist = np.hypot(xx * dx, yy * dy)
            z3 = w[R - 1:R + 2, R - 1:R + 2]
            gx = ((z3[0, 2] + 2 * z3[1, 2] + z3[2, 2]) - (z3[0, 0] + 2 * z3[1, 0] + z3[2, 0])) / (8 * dx)
            gy = ((z3[2, 0] + 2 * z3[2, 1] + z3[2, 2]) - (z3[0, 0] + 2 * z3[0, 1] + z3[0, 2])) / (8 * dy)
            # full-window slope field for neighbourhood stats
            gyw, gxw = np.gradient(w, dy, dx)
            slope_w = np.degrees(np.arctan(np.hypot(gxw, gyw)))
            d2x = (z3[1, 2] - 2 * c + z3[1, 0]) / dx ** 2
            d2y = (z3[2, 1] - 2 * c + z3[0, 1]) / dy ** 2
            f = {"elev_m": c, "slope_deg": float(np.degrees(np.arctan(np.hypot(gx, gy)))),
                 "laplacian_curv": float(d2x + d2y)}
            for rad in (150, 300, 600):
                m = dist <= rad
                vals = w[m]
                f[f"tpi_{rad}m"] = c - np.nanmean(vals)
                f[f"relief_{rad}m"] = np.nanmax(vals) - np.nanmin(vals)
                f[f"elev_std_{rad}m"] = np.nanstd(vals)
                f[f"depth_below_max_{rad}m"] = np.nanmax(vals) - c
                f[f"height_above_min_{rad}m"] = c - np.nanmin(vals)
                f[f"slope_mean_{rad}m"] = np.nanmean(slope_w[m])
            rows.append(f)
    return pd.DataFrame(rows, index=pts.index)


# ---------------------------------------------------------------- landcover
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}


def landcover(pts):
    path = f"{RAW}/esa_worldcover/worldcover_florida_x4.vrt"
    rows = []
    with rasterio.open(path) as r:
        tr = Transformer.from_crs(CRS, r.crs, always_xy=True)
        lon, lat = tr.transform(pts.x.values, pts.y.values)
        R = 28
        for lo, la in zip(lon, lat):
            row, col = r.index(lo, la)
            w = r.read(1, window=Window(col - R, row - R, 2 * R + 1, 2 * R + 1), boundless=True, fill_value=0)
            dy = r.res[1] * 111_320
            dx = r.res[0] * 111_320 * np.cos(np.radians(la))
            yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
            dist = np.hypot(xx * dx, yy * dy)
            f = {"lc_class": int(w[R, R])}
            for rad in (250, 1000):
                m = (dist <= rad) & (w > 0)
                tot = max(m.sum(), 1)
                for k, name in WC.items():
                    f[f"lc_{name}_frac_{rad}m"] = float(((w == k) & m).sum() / tot)
            rows.append(f)
    df = pd.DataFrame(rows, index=pts.index)
    df["lc_class"] = df["lc_class"].map(WC).fillna("other")
    return df


# ---------------------------------------------------------------- hydro
def hydro(pts):
    d = f"{RAW}/florida_mendeley_youn2020"
    df = pd.DataFrame(index=pts.index)
    df["sas_depth_to_water_ft"] = sample_raster(f"{d}/tif/SAS_DTW.tif", pts.x.values, pts.y.values, CRS)
    df["ias_thickness_ft"] = sample_raster(f"{d}/tif/IAS_THK.tif", pts.x.values, pts.y.values, CRS)
    df["head_diff_ft"] = sample_raster(f"{d}/tif/HEAD_DIFF.tif", pts.x.values, pts.y.values, CRS)
    rech = gpd.read_file(f"{d}/rcharg_dec03/rcharg_dec03.shp").to_crs(CRS)[["FLORIDANRE", "geometry"]]
    j = gpd.sjoin(pts[["geometry"]], rech, how="left", predicate="within")
    df["floridan_recharge_class"] = j[~j.index.duplicated()]["FLORIDANRE"].fillna("NA")
    dr = gpd.read_file(f"{RAW}/florida_fgs/layers/drastic_floridan.gpkg").to_crs(CRS)[["INDEX_X", "SETTING", "geometry"]]
    j = gpd.sjoin(pts[["geometry"]], dr, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    idx = pd.to_numeric(j["INDEX_X"], errors="coerce")
    df["drastic_floridan_index"] = idx.where(idx > 0)
    df["drastic_setting"] = j["SETTING"].str.upper().fillna("NA")
    return df


# ---------------------------------------------------------------- geology
def geology(pts):
    L = f"{RAW}/florida_fgs/layers"
    df = pd.DataFrame(index=pts.index)
    specs = [("stratigraphic_geology", {"GEOLOGICUNITCODE": "geo_unit", "LITHOLOGY": "geo_lithology", "GEOLOGICHISTORY": "geo_age"}),
             ("environmental_geology", {"CATEGORY": "envgeo_category"}),
             ("sinkhole_types", {"AREA_DESC": "sinkhole_type_area"}),
             ("geomorphology_districts", {"DISTRICT": "geomorph_district"})]
    for f, cols in specs:
        g = gpd.read_file(f"{L}/{f}.gpkg").to_crs(CRS)[list(cols) + ["geometry"]]
        j = gpd.sjoin(pts[["geometry"]], g, how="left", predicate="within")
        j = j[~j.index.duplicated()]
        miss = j[list(cols)[0]].isna()
        if miss.any():  # points on coast/gaps -> nearest polygon within 2 km
            jn = gpd.sjoin_nearest(pts.loc[miss[miss].index, ["geometry"]], g, how="left", max_distance=2000)
            jn = jn[~jn.index.duplicated()]
            for c in cols:
                j.loc[jn.index, c] = jn[c]
        for c, new in cols.items():
            df[new] = j[c].fillna("NA").astype(str)
    return df


# ---------------------------------------------------------------- closed depressions
def depress(pts):
    g = gpd.read_file(f"{RAW}/florida_mendeley_youn2020/ctds_2004/ctds_2004.shp").to_crs(CRS)[["geometry"]]
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    g["area"] = g.area
    df = pd.DataFrame(index=pts.index)
    jn = gpd.sjoin_nearest(pts[["geometry"]], g, how="left", distance_col="d")
    jn = jn[~jn.index.duplicated()]
    df["ctd_inside"] = (jn["d"] == 0).astype(int)
    df["ctd_dist_m"] = jn["d"]
    for rad in (500, 1000, 2000):
        circ = gpd.GeoDataFrame(geometry=pts.geometry.buffer(rad), crs=CRS)
        ov = gpd.overlay(circ.reset_index(names="pt"), g[["geometry"]], how="intersection", keep_geom_type=True)
        ov["a"] = ov.area
        agg = ov.groupby("pt")["a"].agg(["count", "sum"])
        df[f"ctd_count_{rad}m"] = agg["count"].reindex(pts.index).fillna(0)
        df[f"ctd_area_frac_{rad}m"] = (agg["sum"] / (np.pi * rad ** 2)).reindex(pts.index).fillna(0)
    return df


# ---------------------------------------------------------------- proximity
def proximity(pts):
    L = f"{RAW}/florida_fgs/layers"
    df = pd.DataFrame(index=pts.index)
    for f, name in [("swallets", "dist_swallet_m"), ("springs", "dist_spring_m"), ("phosphate_mined_units_2022", "dist_phosphate_mine_m")]:
        g = gpd.read_file(f"{L}/{f}.gpkg").to_crs(CRS)[["geometry"]]
        jn = gpd.sjoin_nearest(pts[["geometry"]], g, how="left", distance_col="d")
        df[name] = jn[~jn.index.duplicated()]["d"]
    return df


# ---------------------------------------------------------------- soil (POLARIS)
POL = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{var}/mean/{depth}/lat{s}{n}_lon{w}{e}.tif"


def soil(pts):
    tr = Transformer.from_crs(CRS, 4326, always_xy=True)
    lon, lat = tr.transform(pts.x.values, pts.y.values)
    tiles = pd.DataFrame({"lon": lon, "lat": lat, "s": np.floor(lat).astype(int), "w": np.floor(lon).astype(int)}, index=pts.index)
    out = pd.DataFrame(index=pts.index)
    jobs = [("ksat", "0_5"), ("ksat", "30_60"), ("clay", "0_5"), ("sand", "0_5")]

    def run(key):
        (var, depth), (s, w) = key
        sub = tiles[(tiles.s == s) & (tiles.w == w)]
        url = "/vsicurl/" + POL.format(var=var, depth=depth, s=s, n=s + 1, w=w, e=w + 1)
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"), rasterio.open(url) as r:
                v = np.array([x[0] for x in r.sample(zip(sub.lon, sub.lat))], dtype="float64")
                if r.nodata is not None:
                    v[v == r.nodata] = np.nan
        except Exception as e:  # noqa: BLE001
            print("  POLARIS fail", url[-40:], str(e)[:60])
            v = np.full(len(sub), np.nan)
        return f"{var}_{depth}", sub.index, v

    keys = [(j, t) for j in jobs for t in tiles.groupby(["s", "w"]).size().index]
    with ThreadPoolExecutor(12) as ex:
        for col, idx, v in ex.map(run, keys):
            out.loc[idx, f"soil_{col}"] = v
    # POLARIS ksat is log10(cm/hr)
    return out


# ---------------------------------------------------------------- climate
def climate(pts):
    df = pd.DataFrame(index=pts.index)
    df["ppt_annual_mm"] = sample_raster(f"{RAW}/prism/ppt_annual/prism_ppt_us_30s_2020_avg_30y.tif", pts.x.values, pts.y.values, CRS)
    return df


# ---------------------------------------------------------------- expert baseline
def expert(pts):
    v = sample_raster(f"{RAW}/florida_fgs/layers/fgs_sinkhole_favorability_100m.tif", pts.x.values, pts.y.values, CRS, masked=False)
    v[v == 0] = np.nan
    return pd.DataFrame({"fgs_favorability_value": v}, index=pts.index)


GROUPS = {"terrain": terrain, "landcover": landcover, "hydro": hydro, "geology": geology, "depress": depress,
          "proximity": proximity, "soil": soil, "climate": climate, "expert": expert}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(FEAT, exist_ok=True)
    pts = gpd.read_file(f"{PROC}/points.gpkg")
    for gname in a.groups:
        out = f"{FEAT}/{gname}.parquet"
        if os.path.exists(out) and not a.force:
            print("cached", gname)
            continue
        print("extracting", gname, "...", flush=True)
        df = GROUPS[gname](pts)
        df.insert(0, "pid", pts["pid"].values)
        df.to_parquet(out, index=False)
        print("  ", gname, df.shape, "NaN%:", (df.isna().mean() * 100).round(1).to_dict())

    # merge everything available
    base = pd.DataFrame(pts.drop(columns="geometry"))
    for gname in GROUPS:
        p = f"{FEAT}/{gname}.parquet"
        if os.path.exists(p):
            base = base.merge(pd.read_parquet(p), on="pid", how="left")
    base.to_parquet(f"{PROC}/dataset.parquet", index=False)
    print("dataset", base.shape, "->", f"{PROC}/dataset.parquet")


if __name__ == "__main__":
    main()
