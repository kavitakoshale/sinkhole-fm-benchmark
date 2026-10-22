"""Statewide Florida susceptibility maps: foundation model vs XGBoost vs the FGS favourability map.

  grid      build a regular grid over Florida land (cntbnd_sep15), extract exactly the B1 feature groups with the functions
            of 02_extract_features.py (parallel over point chunks and groups) -> data/processed/florida/grid_<res>m.parquet
  predict   fit each model once on the whole headline table (no_landcover × ring, 5,472 rows = the "fixed context"),
            score the grid, write GeoTIFFs + a figure; report map agreement and a sanity AUC of the maps against the
            53 TRUE_SINK=Y reports vs random land cells.

  python scripts/florida/03_predict_raster.py grid --res 1000 --jobs 24
  FM_PRECISION=fp32 python scripts/florida/03_predict_raster.py predict --models tabpfn xgboost xgboost_tuned
Output: results/maps/florida_<model>_<res>m.tif, results/maps/florida_maps.png, results/maps/summary.md
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, f"{ROOT}/scripts")
RAW, PROC, OUT = f"{ROOT}/data/raw", f"{ROOT}/data/processed/florida", f"{ROOT}/results/maps"
CRS = 3086


def extractor():
    """02_extract_features.py as a module, with its RAW redirected to a layout whose big rasters are local copies."""
    spec = importlib.util.spec_from_file_location("ef", f"{ROOT}/scripts/florida/02_extract_features.py")
    ef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef)
    from shared_features import localize
    cache = os.environ.get("SHARED_FEATURES_CACHE", "/tmp/sinkhole_fm_cache")
    fake = f"{cache}/florida_raw_layout"
    for sub, src in (("usgs_3dep_1arcsec/florida_dem.vrt", localize(f"{RAW}/usgs_3dep_1arcsec/florida_dem.vrt")),
                     ("esa_worldcover/worldcover_florida_x4.vrt", localize(f"{RAW}/esa_worldcover/worldcover_florida_x4.vrt")),
                     ("florida_fgs", f"{RAW}/florida_fgs"), ("florida_mendeley_youn2020", f"{RAW}/florida_mendeley_youn2020"),
                     ("prism", f"{RAW}/prism")):
        dst = f"{fake}/{sub}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    ef.RAW = fake
    ef.GROUPS["soil"] = soil_local  # same POLARIS layers, but from tiles downloaded once (147k remote samples are too slow)
    return ef


POLARIS_DIR = f"{RAW}/polaris"
SOIL_JOBS = [("ksat", "0_5"), ("ksat", "30_60"), ("clay", "0_5"), ("sand", "0_5")]


def polaris_tiles(tiles):
    """Download the POLARIS 1°×1° tiles needed (to data/raw/polaris) and mirror them to the local cache; -> local dir."""
    import shutil
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    POL = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{var}/mean/{depth}/lat{s}{n}_lon{w}{e}.tif"
    local = os.path.join(os.environ.get("SHARED_FEATURES_CACHE", "/tmp/sinkhole_fm_cache"), "polaris")

    def get(key):
        (var, depth), (s, w) = key
        name = f"{var}_{depth}/lat{s}{s + 1}_lon{w}{w + 1}.tif"
        dst, loc = f"{POLARIS_DIR}/{name}", f"{local}/{name}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        if not os.path.exists(dst):
            try:
                urllib.request.urlretrieve(POL.format(var=var, depth=depth, s=s, n=s + 1, w=w, e=w + 1), dst + ".part")
                os.rename(dst + ".part", dst)
            except Exception as e:  # noqa: BLE001  (tile outside POLARIS coverage, e.g. open water)
                print("  POLARIS missing", name, str(e)[:60], flush=True)
                return
        if not os.path.exists(loc) or os.path.getsize(loc) != os.path.getsize(dst):
            shutil.copyfile(dst, loc)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, [(j, t) for j in SOIL_JOBS for t in tiles]))
    return local


def soil_local(pts):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(CRS, 4326, always_xy=True).transform(pts.x.values, pts.y.values)
    tiles = pd.DataFrame({"lon": lon, "lat": lat, "s": np.floor(lat).astype(int), "w": np.floor(lon).astype(int)}, index=pts.index)
    local = polaris_tiles(list(tiles.groupby(["s", "w"]).size().index))
    out = pd.DataFrame(index=pts.index)
    for var, depth in SOIL_JOBS:
        col = np.full(len(pts), np.nan)
        for (s, w), sub in tiles.groupby(["s", "w"]):
            f = f"{local}/{var}_{depth}/lat{s}{s + 1}_lon{w}{w + 1}.tif"
            if not os.path.exists(f):
                continue
            with rasterio.open(f) as r:
                v = np.array([x[0] for x in r.sample(zip(sub.lon, sub.lat))], dtype="float64")
                if r.nodata is not None:
                    v[v == r.nodata] = np.nan
            col[pts.index.get_indexer(sub.index)] = v
        out[f"soil_{var}_{depth}"] = col
    return out


def make_grid(res):
    land = gpd.read_file(f"{RAW}/florida_mendeley_youn2020/cntbnd_sep15/cntbnd_sep15.shp").to_crs(CRS)
    x0, y0, x1, y1 = land.total_bounds
    xs = np.arange(np.floor(x0 / res) * res + res / 2, x1, res)
    ys = np.arange(np.floor(y0 / res) * res + res / 2, y1, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame({"x": gx.ravel(), "y": gy.ravel()}, geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=CRS)
    pts = pts.sjoin(land[["geometry"]], how="inner", predicate="within").drop(columns="index_right")
    pts = pts[~pts.index.duplicated()].reset_index(drop=True)
    pts["pid"] = np.arange(len(pts))
    return pts


def _group_chunk(gname, pts_path, lo, hi):
    ef = extractor()
    pts = gpd.read_parquet(pts_path).iloc[lo:hi]
    t0 = time.time()
    df = ef.GROUPS[gname](pts)
    df.insert(0, "pid", pts["pid"].values)
    print(f"  {gname} [{lo}:{hi}] {time.time() - t0:.0f}s", flush=True)
    return df


def grid(a):
    out = f"{PROC}/grid_{a.res}m.parquet"
    pts_path = f"{PROC}/grid_{a.res}m_points.parquet"
    if os.path.exists(pts_path):
        pts = gpd.read_parquet(pts_path)
    else:
        pts = make_grid(a.res)
        pts.to_parquet(pts_path)
    if a.test:
        pts = pts.sample(a.test, random_state=0).sort_index().reset_index(drop=True)
        pts.to_parquet(pts_path.replace(".parquet", "_test.parquet"))
        pts_path = pts_path.replace(".parquet", "_test.parquet")
    print(f"{len(pts)} grid cells at {a.res} m", flush=True)
    feat_dir = f"{PROC}/grid_features_{a.res}m" + ("_test" if a.test else "")
    os.makedirs(feat_dir, exist_ok=True)
    chunk = {"terrain": 2000, "landcover": 2000, "soil": 20000, "depress": 5000}
    extractor()  # localize the big rasters once in the parent, before the parallel chunks
    if "soil" in a.groups:  # fetch the POLARIS tiles once, before the parallel chunks
        from pyproj import Transformer
        lon, lat = Transformer.from_crs(CRS, 4326, always_xy=True).transform(pts.x.values, pts.y.values)
        polaris_tiles(sorted(set(zip(np.floor(lat).astype(int), np.floor(lon).astype(int)))))
    for gname in a.groups:
        fp = f"{feat_dir}/{gname}.parquet"
        if os.path.exists(fp):
            print("cached", gname)
            continue
        c = chunk.get(gname, 50000)
        bounds = [(lo, min(lo + c, len(pts))) for lo in range(0, len(pts), c)]
        t0 = time.time()
        parts = Parallel(n_jobs=min(a.jobs, len(bounds)), backend="loky")(delayed(_group_chunk)(gname, pts_path, lo, hi) for lo, hi in bounds)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(fp, index=False)
        print(f"{gname}: {df.shape} in {time.time() - t0:.0f}s, NaN% {(df.isna().mean() * 100).round(1).to_dict()}", flush=True)
    base = pd.DataFrame(pts.drop(columns="geometry"))
    for fp in sorted(glob.glob(f"{feat_dir}/*.parquet")):
        base = base.merge(pd.read_parquet(fp), on="pid", how="left")
    base.to_parquet(out if not a.test else out.replace(".parquet", "_test.parquet"), index=False)
    print("grid table", base.shape, flush=True)


def predict(a):
    from common import MODELS
    from fm_models import FM_MODELS
    from run_florida_baselines import DATA, feature_cols
    from tuned import TUNED_MODELS
    ALL = {**MODELS, **FM_MODELS, **TUNED_MODELS}
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_parquet(DATA)
    num, cat = feature_cols(d, False, "no_landcover")
    d = d[(d.label == 1) | (d.neg_strategy == "ring")].reset_index(drop=True)
    g = pd.read_parquet(f"{PROC}/grid_{a.res}m.parquet")
    for c in cat:  # grid categories unseen in the table -> "NA" (the table's own missing code)
        g[c] = g[c].where(g[c].isin(set(d[c].astype(str))), "NA").astype(str)
    ok = g[num].notna().mean(axis=1) > 0.5  # cells with features (offshore / nodata cells are dropped)
    pred = g.loc[ok, ["pid", "x", "y", "fgs_favorability_value"]].copy()
    prev = f"{OUT}/florida_grid_predictions_{a.res}m.parquet"
    if os.path.exists(prev):  # resume: keep the models already scored
        old = pd.read_parquet(prev)
        for m in [c for c in old.columns if c in a.models]:
            pred[m] = old.set_index("pid")[m].reindex(pred.pid).to_numpy()
    timing = {}
    for m in a.models:
        if m in pred:
            print("cached", m, flush=True)
            continue
        t0 = time.time()
        est = ALL[m](num, cat, 0)
        est = est.fit(d[num + cat], d.label.to_numpy(), groups=d.block_id.to_numpy()) if getattr(est, "needs_groups", False) \
            else est.fit(d[num + cat], d.label.to_numpy())
        t1 = time.time()
        pred[m] = est.predict_proba(g.loc[ok, num + cat])[:, 1]
        timing[m] = {"fit_s": t1 - t0, "predict_s": time.time() - t1, "n_cells": int(ok.sum())}
        print(m, timing[m], flush=True)
        pred.to_parquet(f"{OUT}/florida_grid_predictions_{a.res}m.parquet", index=False)
    write_rasters(pred, a.res, a.models)
    report(pred, d, a.res, a.models, timing)


def write_rasters(pred, res, models):
    x0, x1, y0, y1 = pred.x.min(), pred.x.max(), pred.y.min(), pred.y.max()
    width, height = int(round((x1 - x0) / res)) + 1, int(round((y1 - y0) / res)) + 1
    col = ((pred.x - x0) / res).round().astype(int)
    row = ((y1 - pred.y) / res).round().astype(int)
    tr = from_origin(x0 - res / 2, y1 + res / 2, res, res)
    prof = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32", crs=f"EPSG:{CRS}", transform=tr,
                nodata=-9999, compress="deflate")
    for m in models + ["fgs_favorability_value"]:
        arr = np.full((height, width), -9999, dtype="float32")
        v = pred[m].to_numpy(dtype="float32").copy()
        v[np.isnan(v)] = -9999
        arr[row, col] = v
        with rasterio.open(f"{OUT}/florida_{m}_{res}m.tif", "w", **prof) as w:
            w.write(arr, 1)


def report(pred, d, res, models, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    land = gpd.read_file(f"{RAW}/florida_mendeley_youn2020/cntbnd_sep15/cntbnd_sep15.shp").to_crs(CRS)
    ver = d[(d.label == 1) & (d.true_sink == "Y")]
    allpos = d[d.label == 1]
    # nearest grid cell of every report, for the sanity AUC (map value at verified sinkholes vs random land cells)
    from scipy.spatial import cKDTree
    tree = cKDTree(pred[["x", "y"]].to_numpy())
    rng = np.random.default_rng(0)
    rows = []
    for m in models + ["fgs_favorability_value"]:
        v = pred[m].to_numpy(dtype=float)
        at = lambda df: v[tree.query(df[["x", "y"]].to_numpy())[1]]  # noqa: E731
        bg = v[rng.choice(len(v), 20000, replace=False)]
        s_ver, s_all = at(ver), at(allpos)
        fin = lambda a, b: (np.concatenate([np.ones(len(a)), np.zeros(len(b))]), np.concatenate([a, b]))  # noqa: E731
        y1, s1 = fin(s_ver[~np.isnan(s_ver)], bg[~np.isnan(bg)])
        y2, s2 = fin(s_all[~np.isnan(s_all)], bg[~np.isnan(bg)])
        r = {"model": m, "cells": int(np.isfinite(v).sum()), "auc_verified53_vs_land": roc_auc_score(y1, s1),
             "auc_all_reports_vs_land": roc_auc_score(y2, s2),
             "share_cells_p>0.5": float(np.nanmean(v > 0.5)) if m != "fgs_favorability_value" else np.nan, **timing.get(m, {})}
        for o in models:
            if o != m and o in pred:
                r[f"spearman_vs_{o}"] = spearmanr(v, pred[o], nan_policy="omit")[0]
        rows.append(r)
    res_df = pd.DataFrame(rows)
    # figure
    n = len(models) + 2
    ncol = int(np.ceil(n / 2))
    fig, axes = plt.subplots(2, ncol, figsize=(5.6 * ncol, 11.5))
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_axis_off()
    for ax, m in zip(axes, models + ["fgs_favorability_value"]):
        sc = ax.scatter(pred.x, pred.y, c=pred[m], s=1.2, marker="s", cmap="viridis" if m != "fgs_favorability_value" else "magma", lw=0)
        land.boundary.plot(ax=ax, lw=.2, color="k")
        ax.scatter(ver.x, ver.y, s=10, facecolors="none", edgecolors="r", lw=.6, label="verified sinkholes (53)")
        ax.set_title(f"{m}" + ("" if m == "fgs_favorability_value" else " (fixed context: 5,472 rows)"))
        ax.set_axis_off()
        plt.colorbar(sc, ax=ax, shrink=.6)
    if len(models) >= 2:
        ax = axes[n - 1]
        diff = pred[models[0]].rank(pct=True) - pred[models[1]].rank(pct=True)
        sc = ax.scatter(pred.x, pred.y, c=diff, s=1.2, marker="s", cmap="RdBu_r", vmin=-.5, vmax=.5, lw=0)
        land.boundary.plot(ax=ax, lw=.2, color="k")
        ax.set_title(f"percentile({models[0]}) − percentile({models[1]})")
        ax.set_axis_off()
        plt.colorbar(sc, ax=ax, shrink=.6)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/florida_maps.png", dpi=110)
    with open(f"{OUT}/summary.md", "w") as fh:
        fh.write(f"# Statewide Florida susceptibility maps ({res} m grid, {len(pred)} land cells)\n\n"
                 "Each model is fit once on the whole headline table (no_landcover × ring: 2,736 reports + 2,736 ring pseudo-absences) "
                 "and scores every grid cell; for the FMs this is the deployment mode (one fixed context, no training). "
                 "`auc_*_vs_land` is a sanity check only: map value at the reports (which were in the context) vs 20,000 random land cells. "
                 "GeoTIFFs: `results/maps/florida_<model>_<res>m.tif` (EPSG:3086). Figure: `results/maps/florida_maps.png`.\n\n" +
                 res_df.round(4).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/summary.md").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "predict"])
    ap.add_argument("--res", type=int, default=1000)
    ap.add_argument("--groups", nargs="*", default=["terrain", "hydro", "geology", "depress", "proximity", "soil", "climate", "expert"])
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--test", type=int, default=0, help="time the extraction on N random cells only")
    ap.add_argument("--models", nargs="*", default=["tabpfn", "xgboost"])
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
