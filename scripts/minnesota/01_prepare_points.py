"""Build the Minnesota sinkhole presence/absence point set (third region). Mirrors scripts/florida/01_prepare_points.py.

Positives: Minnesota Karst Feature Inventory (MGS/DNR, MN Geospatial Commons), feature == "D" (sinkhole), with a known
location method better than "Manual/Unknown"; de-duplicated at 50 m. Carries `field_checked` (field_check_date set)
and `loc_method` (LiDAR / digitized 1:24k / GPS) so the field-checked points can serve as a clean test set.
Ring negatives are paired one-per-positive (uniform in the 500-5000 m annulus) rather than uniform over the ring union.
Study area: Minnesota counties with >= MIN_COUNTY_POS inventory sinkholes (the SE Minnesota karst region). Sampling
"global" negatives statewide would make the task a geography lookup, so both strategies stay inside the study area:
  * "global": uniform over the study-area counties, >= 500 m from any inventory feature (incl. non-sinkhole karst features)
  * "ring":   uniform within a 500-5000 m ring around positives
Splits as Florida: fold_random (stratified 5-fold), block_id / fold_spatial (25 km blocks -> 5 folds), split_temporal
(positives located (geoc_date) >= TEMPORAL_CUT -> "test": essentially the 2014+ LiDAR additions; negatives in proportion).

Output: data/processed/minnesota/points.gpkg (EPSG:26915, NAD83 / UTM 15N)
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KFI = f"{ROOT}/data/raw/minnesota_mgs/karst_feature_inventory_pts.gpkg"
COUNTIES = f"{ROOT}/data/raw/tiger2023_us_county.gpkg"
OUT_DIR = f"{ROOT}/data/processed/minnesota"
CRS = 26915
SEED = 42
MIN_DIST = 500.0
RING_MAX = 5000.0
NEG_RATIO = 1.0
BLOCK = 25_000.0
TEMPORAL_CUT = 2014
MIN_COUNTY_POS = 10
BAD_LOC = ("Unknown", "Manual")
LOC_METHOD = {"LiDAR": "lidar", "Digitized": "digitized", "GPS": "gps"}


def sample_uniform(poly, n, rng, batch=200_000):
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    while sum(len(p) for p in pts) < n:
        xs, ys = rng.uniform(minx, maxx, batch), rng.uniform(miny, maxy, batch)
        cand = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=CRS)
        pts.append(cand[cand.within(poly)])
    return gpd.GeoSeries(pd.concat(pts).iloc[:n].values, crs=CRS)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    kfi = gpd.read_file(KFI, layer="karst_feature_inventory_points").to_crs(CRS)
    kfi = kfi[kfi.geometry.notna() & ~kfi.geometry.is_empty]
    print("inventory features", len(kfi), kfi.feature.value_counts().to_dict())
    counties = gpd.read_file(COUNTIES, where="STATEFP='27'").to_crs(CRS)[["NAME", "geometry"]]

    pos = kfi[(kfi.feature == "D") & ~kfi.gcm.fillna("Unknown").str.startswith(BAD_LOC)].copy()
    pos = gpd.sjoin(pos, counties, how="left", predicate="within").rename(columns={"NAME": "county"})
    pos = pos[pos.county.notna()]
    pos["_gx"], pos["_gy"] = (pos.geometry.x // 50).astype(int), (pos.geometry.y // 50).astype(int)
    pos = pos.drop_duplicates(["_gx", "_gy"]).drop(columns=["_gx", "_gy"])
    keep = pos.county.value_counts()
    study = keep[keep >= MIN_COUNTY_POS].index.tolist()
    pos = pos[pos.county.isin(study)]
    land = counties[counties.NAME.isin(study)].union_all().buffer(0)
    print("positives after filters", len(pos), "| study-area counties", sorted(study))
    print("location method", pos.gcm.str.split(r"[ ;-]").str[0].value_counts().to_dict())
    print("field checked", int(pos.field_check_date.notna().sum()))

    pos_out = gpd.GeoDataFrame({
        "pid": [f"KFI_{r}" for r in pos.relateid.astype(str)],
        "ref_num": pos.relateid.astype(str).values,
        "label": 1,
        "event_year": pd.to_datetime(pos.geoc_date, errors="coerce", utc=True).dt.year.values,
        "accuracy": pos.gcm.str.split(r"[ ;-]").str[0].map(LOC_METHOD).fillna("other").values,
        "true_sink": np.where(pos.field_check_date.notna(), "Y", "U"),
        "field_checked": pos.field_check_date.notna().values.astype(int),
        "county": pos.county.values,
    }, geometry=pos.geometry.values, crs=CRS)

    n_neg = int(round(len(pos_out) * NEG_RATIO))
    # 15k positives / 17k features: nearest-neighbour distances instead of buffer unions (same geometry, minutes faster)
    all_tree, pos_tree = STRtree(kfi.geometry.values), STRtree(pos_out.geometry.values)

    def far_from_any(g):
        return all_tree.query_nearest(g.values, return_distance=True, all_matches=False)[1] >= MIN_DIST

    cand = sample_uniform(land, n_neg * 3, rng)
    cand = cand[far_from_any(cand)].iloc[:n_neg]
    neg_g = gpd.GeoDataFrame({"pid": [f"NEGG_{i}" for i in range(len(cand))], "label": 0}, geometry=cand.values, crs=CRS)
    neg_g["neg_strategy"] = "global"

    # ring negatives are paired: one per positive, uniform in its 500-5000 m annulus (Florida samples the ring *union*
    # uniformly, which here would starve the dense Fillmore karst of negatives because little of it is > 500 m from a feature)
    px, py = pos_out.geometry.x.to_numpy(), pos_out.geometry.y.to_numpy()
    ring_pts = np.full((len(pos_out), 2), np.nan)
    todo = np.arange(len(pos_out))
    for _ in range(40):
        if not len(todo):
            break
        r = np.sqrt(rng.uniform(MIN_DIST ** 2, RING_MAX ** 2, len(todo)))
        th = rng.uniform(0, 2 * np.pi, len(todo))
        cand = gpd.GeoSeries(gpd.points_from_xy(px[todo] + r * np.cos(th), py[todo] + r * np.sin(th)), crs=CRS)
        ok = far_from_any(cand) & cand.within(land).to_numpy()
        ring_pts[todo[ok]] = np.c_[cand.x.to_numpy()[ok], cand.y.to_numpy()[ok]]
        todo = todo[~ok]
    print("positives without a valid ring negative after 40 draws:", len(todo))
    cand = gpd.GeoSeries(gpd.points_from_xy(ring_pts[:, 0], ring_pts[:, 1]), crs=CRS)[np.isfinite(ring_pts[:, 0])]
    neg_r = gpd.GeoDataFrame({"pid": [f"NEGR_{i}" for i in range(len(cand))], "label": 0}, geometry=cand.values, crs=CRS)
    neg_r["neg_strategy"] = "ring"
    print("negatives", len(neg_g), "global,", len(neg_r), "ring")

    pts = gpd.GeoDataFrame(pd.concat([pos_out.assign(neg_strategy="positive"), neg_g, neg_r], ignore_index=True), geometry="geometry", crs=CRS)
    neg = pts.label == 0
    pts.loc[neg, "county"] = gpd.sjoin(pts[neg], counties, how="left", predicate="within")["NAME"].groupby(level=0).first()
    pts["field_checked"] = pts["field_checked"].fillna(0).astype(int)
    ll = pts.to_crs(4326).geometry
    pts["lon"], pts["lat"] = ll.x, ll.y
    pts["x"], pts["y"] = pts.geometry.x, pts.geometry.y

    bx, by = (pts["x"] // BLOCK).astype(int), (pts["y"] // BLOCK).astype(int)
    pts["block_id"] = bx.astype(str) + "_" + by.astype(str)
    blocks = pts.groupby("block_id")["label"].agg(["sum", "count"]).sample(frac=1, random_state=SEED)
    blocks = blocks.sort_values("count", ascending=False, kind="stable")
    load, fold_of = np.zeros(5), {}
    for b, r in blocks.iterrows():
        f = int(np.argmin(load))
        fold_of[b] = f
        load[f] += r["count"]
    pts["fold_spatial"] = pts["block_id"].map(fold_of)

    pts["fold_random"] = -1
    for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.arange(len(pts)), pts["label"])):
        pts.loc[pts.index[te], "fold_random"] = f

    is_pos = pts["label"] == 1
    pts["split_temporal"] = np.where(pts["event_year"] >= TEMPORAL_CUT, "test", "train")
    pts.loc[is_pos & pts["event_year"].isna(), "split_temporal"] = "train"
    frac_test = (pts.loc[is_pos, "split_temporal"] == "test").mean()
    neg_idx = pts.index[~is_pos]
    pts.loc[neg_idx, "split_temporal"] = np.where(rng.random(len(neg_idx)) < frac_test, "test", "train")
    print(f"temporal test fraction {frac_test:.3f}")

    out = f"{OUT_DIR}/points.gpkg"
    pts.to_file(out, driver="GPKG")
    print("wrote", out, len(pts))
    print(pts.groupby(["neg_strategy", "fold_spatial"]).size().unstack())
    print(pts[is_pos].groupby("county").agg(n=("pid", "size"), field_checked=("field_checked", "sum")).sort_values("n", ascending=False))


if __name__ == "__main__":
    main()
