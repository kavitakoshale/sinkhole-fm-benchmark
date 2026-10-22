"""Replication of Subedi et al. (2019, Sci. Rep. 9:7140): Marion County, FL, logistic regression vs AHP.

Paper design: FGS SIR events Nov-1973..Jun-2018 (n=398), 700 random absences (ratio 1:1.78), 5 factors
(distance to closed topographic depression, surficial geology, soil permeability, distance to flowlines,
distance to active mines). Split: 1973-2007 events -> random 50/50 train/validation; 2008-2018 -> independent set.
Reported: LR AUC 0.808 (validation), 0.826 (2008-2018); AHP 0.730 / 0.796.

Our analogues: ctd_dist_m (FGS ctds_2004), geo_unit (FGS stratigraphic geology), soil_ksat_30_60 (POLARIS),
DEM-derived height_above_min_600m (no NHD flowlines; proxy for incision), dist_phosphate_mine_m.
We also run the full 68-feature GBDTs on the same split to show the gap.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import importlib.util  # noqa: E402
import sys  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import MODELS, metrics  # noqa: E402
from run_florida_baselines import feature_cols  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("fx", f"{ROOT}/scripts/florida/02_extract_features.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)

OUT = f"{ROOT}/results/florida_subedi2019"
SUBEDI5_NUM = ["ctd_dist_m", "soil_ksat_30_60", "height_above_min_600m", "dist_phosphate_mine_m"]
SUBEDI5_CAT = ["geo_unit"]
SEED = 7


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cnty = gpd.read_file(f"{ROOT}/data/raw/florida_mendeley_youn2020/cntbnd_sep15/cntbnd_sep15.shp").to_crs(3086)
    marion = cnty[cnty.NAME.str.upper() == "MARION"].geometry.iloc[0]
    sir = gpd.read_file(f"{ROOT}/data/raw/florida_fgs/subsidence_incident_reports.gpkg").to_crs(3086)
    sir["event_year"] = pd.to_datetime(sir["EVENT_DATE"], unit="ms", errors="coerce").dt.year
    pos = sir[sir.within(marion) & (sir.event_year <= 2018) & (sir.TRUE_SINK != "N")].copy()
    print("Marion SIR events <= 2018:", len(pos), "(paper: 398)")

    # absences: random in county, >= 100 m from any report, 1:1.78
    n_neg = int(round(len(pos) * 1.78))
    excl = sir.geometry.union_all().buffer(100)
    minx, miny, maxx, maxy = marion.bounds
    xs, ys = rng.uniform(minx, maxx, n_neg * 6), rng.uniform(miny, maxy, n_neg * 6)
    cand = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=3086)
    cand = cand[cand.within(marion) & ~cand.within(excl)].iloc[:n_neg]

    pts = gpd.GeoDataFrame({
        "pid": [f"P{i}" for i in range(len(pos))] + [f"N{i}" for i in range(len(cand))],
        "label": [1] * len(pos) + [0] * len(cand),
        "event_year": list(pos.event_year) + [np.nan] * len(cand),
    }, geometry=list(pos.geometry) + list(cand), crs=3086)
    pts["x"], pts["y"] = pts.geometry.x, pts.geometry.y

    cache = f"{OUT}/features.parquet"
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        parts = [pd.DataFrame(pts.drop(columns="geometry"))]
        for g in ["terrain", "landcover", "hydro", "geology", "depress", "proximity", "soil", "climate", "expert"]:
            print("  features:", g, flush=True)
            parts.append(fx.GROUPS[g](pts).reset_index(drop=True))
        df = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
        df.to_parquet(cache, index=False)

    # paper split: <=2007 events -> 50/50 train/val; 2008-2018 -> independent; absences 300/300/100 proportionally
    is_pos = df.label == 1
    old = is_pos & (df.event_year <= 2007)
    new = is_pos & (df.event_year >= 2008)
    split = pd.Series("drop", index=df.index)
    split[old] = np.where(rng.random(old.sum()) < 0.5, "train", "val")
    split[new] = "indep"
    neg_idx = df.index[~is_pos]
    split[neg_idx] = rng.choice(["train", "val", "indep"], size=len(neg_idx), p=[3 / 7, 3 / 7, 1 / 7])
    df["split"] = split
    print(df.groupby(["split", "label"]).size().unstack())

    num_all, cat_all = feature_cols(df.assign(neg_strategy="", fold_spatial=0, fold_random=0, split_temporal="",
                                              block_id="", lon=0, lat=0, accuracy="", true_sink="", county="", ref_num=""))
    num_all = [c for c in num_all if c != "split"]
    rows = []
    for fs, (num, cat) in {"subedi5": (SUBEDI5_NUM, SUBEDI5_CAT), "full68": (num_all, cat_all)}.items():
        tr = df[df.split == "train"]
        for name in (["logreg"] if fs == "subedi5" else []) + ["logreg", "random_forest", "xgboost", "catboost"] * (fs == "full68"):
            m = MODELS[name](num, cat, 0).fit(tr[num + cat], tr.label)
            for te_name in ["val", "indep"]:
                te = df[df.split == te_name]
                r = metrics(te.label, m.predict_proba(te[num + cat])[:, 1])
                rows.append({"features": fs, "model": name, "test": te_name, **r})
    for te_name in ["val", "indep"]:
        te = df[df.split == te_name]
        from sklearn.metrics import roc_auc_score
        ok = te.fgs_favorability_value.notna()
        rows.append({"features": "expert", "model": "fgs_favorability_map", "test": te_name,
                     "roc_auc": roc_auc_score(te.label[ok], te.fgs_favorability_value[ok]), "n_test": int(ok.sum())})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/results.csv", index=False)
    tab = res.pivot_table(index=["features", "model"], columns="test", values="roc_auc").round(3)
    with open(f"{OUT}/summary.md", "w") as fh:
        fh.write("# Subedi et al. (2019) replication, Marion County FL (ROC-AUC)\n\n"
                 "Paper: LR 0.808 (val) / 0.826 (2008-2018 indep); AHP 0.730 / 0.796\n\n")
        fh.write(tab.to_markdown() + "\n")
    print(open(f"{OUT}/summary.md").read())


if __name__ == "__main__":
    main()
