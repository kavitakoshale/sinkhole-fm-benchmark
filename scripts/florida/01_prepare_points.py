"""Build the Florida sinkhole presence/absence point set.

Positives: FGS Subsidence Incident Reports (SIR), filtered for location accuracy and non-sinkhole flags.
Negatives: random land points, two strategies (both written, tagged by `neg_strategy`):
  * "global": uniform over Florida land, >= MIN_DIST m from any SIR report (incl. low-accuracy ones)
  * "ring":   uniform within a 500-5000 m ring around positives (Altarawneh et al. 2026 style buffer),
              which removes the easy "far away from any report" signal.
Each strategy has a 1:1 ratio by default (NEG_RATIO).

Also assigns evaluation splits:
  * fold_random  : stratified random 5-fold
  * block_id / fold_spatial : 25 km grid blocks -> 5 folds (GroupKFold-like, balanced by count)
  * split_temporal: positives with event year >= 2010 -> "test", < 2010 -> "train"; negatives randomly
                    assigned in the same proportion (they are timeless).

Output: data/processed/florida/points.gpkg (EPSG:3086, NAD83 / Florida GDL Albers)
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIR = f"{ROOT}/data/raw/florida_fgs/subsidence_incident_reports.gpkg"
COUNTIES = f"{ROOT}/data/raw/florida_mendeley_youn2020/cntbnd_sep15/cntbnd_sep15.shp"
OUT_DIR = f"{ROOT}/data/processed/florida"
CRS = 3086
SEED = 42
GOOD_ACCURACY = {"PARCEL", "GPSSA", "QTR3", "QTR2"}  # <= ~400 m positional uncertainty
MIN_DIST = 500.0  # m, negatives keep away from any report
RING_MAX = 5000.0
NEG_RATIO = 1.0
BLOCK = 25_000.0  # m
TEMPORAL_CUT = 2010


def sample_uniform(poly, n, rng, batch=200_000):
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    while sum(len(p) for p in pts) < n:
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        cand = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=CRS)
        pts.append(cand[cand.within(poly)])
    return gpd.GeoSeries(pd.concat(pts).iloc[:n].values, crs=CRS)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    sir = gpd.read_file(SIR).to_crs(CRS)
    sir["event_year"] = pd.to_datetime(sir["EVENT_DATE"], unit="ms", errors="coerce").dt.year
    print("SIR total", len(sir))

    land = gpd.read_file(COUNTIES).to_crs(CRS).union_all().buffer(0)

    pos = sir[sir["ACCURACY"].isin(GOOD_ACCURACY) & (sir["TRUE_SINK"] != "N")].copy()
    pos = pos[pos.within(land.buffer(200))]
    # de-duplicate reports closer than 50 m (repeat reports of the same feature)
    pos["_gx"] = (pos.geometry.x // 50).astype(int)
    pos["_gy"] = (pos.geometry.y // 50).astype(int)
    pos = pos.drop_duplicates(["_gx", "_gy"]).drop(columns=["_gx", "_gy"])
    print("positives after filters", len(pos), pos["ACCURACY"].value_counts().to_dict())

    pos_out = gpd.GeoDataFrame({
        "pid": [f"SIR_{r}_{i}" for i, r in enumerate(pos["REF_NUM"].astype(str))],
        "ref_num": pos["REF_NUM"].astype(str).values,
        "label": 1,
        "event_year": pos["event_year"],
        "accuracy": pos["ACCURACY"],
        "true_sink": pos["TRUE_SINK"],
        "county": pos["COUNTY"],
    }, geometry=pos.geometry.values, crs=CRS)

    n_neg = int(round(len(pos_out) * NEG_RATIO))
    all_reports = sir.geometry.union_all()
    excl = all_reports.buffer(MIN_DIST)

    # global negatives
    cand = sample_uniform(land, n_neg * 3, rng)
    cand = cand[~cand.within(excl)].iloc[:n_neg]
    neg_g = gpd.GeoDataFrame({"pid": [f"NEGG_{i}" for i in range(len(cand))], "label": 0}, geometry=cand.values, crs=CRS)
    neg_g["neg_strategy"] = "global"

    # ring negatives: inside 5 km of a positive, outside 500 m of every report
    ring = pos_out.geometry.union_all().buffer(RING_MAX).intersection(land).difference(excl)
    cand = sample_uniform(ring, n_neg, rng)
    neg_r = gpd.GeoDataFrame({"pid": [f"NEGR_{i}" for i in range(len(cand))], "label": 0}, geometry=cand.values, crs=CRS)
    neg_r["neg_strategy"] = "ring"
    print("negatives", len(neg_g), "global,", len(neg_r), "ring")

    pts = pd.concat([pos_out.assign(neg_strategy="positive"), neg_g, neg_r], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=CRS)
    ll = pts.to_crs(4326).geometry
    pts["lon"], pts["lat"] = ll.x, ll.y
    pts["x"], pts["y"] = pts.geometry.x, pts.geometry.y

    # spatial blocks
    bx = (pts["x"] // BLOCK).astype(int)
    by = (pts["y"] // BLOCK).astype(int)
    pts["block_id"] = bx.astype(str) + "_" + by.astype(str)
    # assign blocks to 5 folds, greedily balancing positives (shuffle then fill smallest)
    blocks = pts.groupby("block_id")["label"].agg(["sum", "count"]).sample(frac=1, random_state=SEED)
    blocks = blocks.sort_values("count", ascending=False, kind="stable")
    load = np.zeros(5)
    fold_of = {}
    for b, r in blocks.iterrows():
        f = int(np.argmin(load))
        fold_of[b] = f
        load[f] += r["count"]
    pts["fold_spatial"] = pts["block_id"].map(fold_of)

    # random stratified folds (computed per negative strategy dataset, so positives get one fold)
    pts["fold_random"] = -1
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    idx = np.arange(len(pts))
    for f, (_, te) in enumerate(skf.split(idx, pts["label"])):
        pts.loc[pts.index[te], "fold_random"] = f

    # temporal split
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


if __name__ == "__main__":
    main()
