"""Realistic-regime benchmark: the setting a county geologist actually faces.

Regions: florida (committed B1 table; feature sets no_landcover, subedi5, sat_nolc from scripts/feature_sets.py) and
minnesota (shared schema sat_nolc). Ring negatives only. Experiments:

  county      County few-shot. For every county with >= MIN_COUNTY positives: train on k in-county positives + k in-county
              ring negatives (k = 25, 50, 100; 3 seeds), test (a) on the rest of the county and (b) on neighbouring counties
              (rows within NEIGHBOUR_KM of any training point, other counties).
  clean       Clean-label test sets. Test = verified positives (Florida TRUE_SINK == Y, n = 53; Minnesota field_checked, n = 827)
              + one paired ring negative each (nearest). Train = every other row, (i) all of them and (ii) excluding the test
              rows' 25 km blocks (no spatial leakage). Minnesota adds "verified-only": 5-fold spatial CV restricted to the
              field-checked positives and their paired negatives (clean labels on both sides, n ~ 1.6k).
  prevalence  Realistic prevalence. Spatial 5-fold CV where the *test* fold keeps all negatives but only 1/10 (and 1/20) of
              its positives (5 resamples); training folds stay balanced ("balanced") or are thinned the same way
              ("imbalanced"). Reports ROC-AUC, PR-AUC, Brier, log-loss, ECE at the low prevalence.

  python scripts/realistic_benchmark.py county --regions florida minnesota --models logreg xgboost tabpfn ...
  python scripts/realistic_benchmark.py summarize
Outputs: results/realistic/<experiment>/runs_<region>_<fs>.csv and summary.md.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, os.path.dirname(__file__))
from feature_sets import feature_sets as fl_feature_sets  # noqa: E402
from fm_scenarios import ece, fit_predict  # noqa: E402
from run_region_baselines import FEATURE_SETS as SHARED_SETS  # noqa: E402

OUT = f"{ROOT}/results/realistic"
MIN_COUNTY = 100
NEIGHBOUR_KM = 50
DEFAULT_MODELS = ["logreg", "random_forest", "xgboost", "lightgbm", "tabpfn", "tabicl", "kumo_tabular"]
SAVE_PATH = [None]  # incremental checkpoint of the current runs file


def load(region):
    if region == "florida":
        d = pd.read_parquet(f"{ROOT}/data/processed/florida/dataset.parquet")
        sets = {k: v for k, v in fl_feature_sets(d).items() if k in ("no_landcover", "subedi5", "sat_nolc")}
        d["verified"] = (d.true_sink == "Y").astype(int)
        d["county"] = d["county"].str.title()
        miss = d.county.isna()
        if miss.any():  # the committed table only carries the SIR county of positives; negatives get TIGER counties
            import geopandas as gpd
            tiger = gpd.read_file(f"{ROOT}/data/raw/tiger2023_us_county.gpkg", where="STATEFP='12'")[["NAME", "geometry"]]
            pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(d.loc[miss, "lon"], d.loc[miss, "lat"]), crs=4326, index=d.index[miss])
            d.loc[miss, "county"] = gpd.sjoin_nearest(pts.to_crs(tiger.crs), tiger, how="left")["NAME"].groupby(level=0).first()
    else:
        d = pd.read_parquet(f"{ROOT}/data/processed/{region}/dataset.parquet")
        sets = {"sat_nolc": SHARED_SETS["sat_nolc"]}
        d["verified"] = d.get("field_checked", 0)
    d = d[(d.label == 1) | (d.neg_strategy == "ring")].reset_index(drop=True)
    return d, sets


def run(rows, base, name, num, cat, seed, tr, te, d, extra_metrics=False):
    cols = num + cat
    t0 = time.time()
    p = fit_predict(name, num, cat, seed, d.loc[tr, cols], d.loc[tr, "label"].to_numpy(), d.loc[te, cols], d.loc[tr, "block_id"].to_numpy())
    y = d.loc[te, "label"].to_numpy()
    r = {**base, "model": name, "n_train": len(tr), "pos_train": int(d.loc[tr, "label"].sum()), "n_test": len(te),
         "pos_test": int(y.sum()), "roc_auc": np.nan, "pr_auc": np.nan, "seconds": round(time.time() - t0, 1)}
    if p is not None and y.min() != y.max():
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        r.update(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))
        if extra_metrics:
            r.update(brier=brier_score_loss(y, pc), log_loss=log_loss(y, pc), ece=ece(y, pc), mean_p=pc.mean(), prevalence=y.mean())
    rows.append(r)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    if SAVE_PATH[0] and len(rows) % 20 == 0:
        pd.DataFrame(rows).to_csv(SAVE_PATH[0], index=False)


def county(a, region, d, num, cat, fs, rows):
    counts = d[d.label == 1].county.value_counts()
    xy = d[["x", "y"]].to_numpy()
    for c in counts[counts >= MIN_COUNTY].index:
        inc = d.index[d.county == c]
        pos, neg = inc[d.loc[inc, "label"] == 1], inc[d.loc[inc, "label"] == 0]
        for k in a.k:
            if k > len(pos) - 25 or k > len(neg):
                continue
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                tr = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, min(k, len(neg)), replace=False)])
                te_in = inc.difference(tr)
                near = cKDTree(xy[tr]).query(xy, distance_upper_bound=NEIGHBOUR_KM * 1000)[0] < np.inf
                te_nb = d.index[near & (d.county != c)]
                for m in a.models:
                    base = {"region": region, "feature_set": fs, "county": c, "k": k, "rep": seed}
                    run(rows, {**base, "test": "in_county"}, m, num, cat, seed, tr, te_in, d)
                    if len(te_nb) >= 20 and d.loc[te_nb, "label"].nunique() == 2:
                        run(rows, {**base, "test": "neighbours"}, m, num, cat, seed, tr, te_nb, d)


def paired_negatives(d, pos_idx):
    neg = d.index[d.label == 0]
    _, j = cKDTree(d.loc[neg, ["x", "y"]].to_numpy()).query(d.loc[pos_idx, ["x", "y"]].to_numpy())
    return neg[np.unique(j)]


def clean(a, region, d, num, cat, fs, rows):
    ver = d.index[(d.label == 1) & (d.verified == 1)]
    if len(ver) < 20:
        return
    te = ver.union(paired_negatives(d, ver))
    rest = d.index.difference(te)
    base = {"region": region, "feature_set": fs, "n_verified": len(ver)}
    for m in a.models:
        run(rows, {**base, "variant": "train_all_others", "fold": "all", "rep": 0}, m, num, cat, 0, rest, te, d)
        blocks = set(d.loc[te, "block_id"])
        run(rows, {**base, "variant": "train_outside_test_blocks", "fold": "all", "rep": 0}, m, num, cat, 0,
            rest[~d.loc[rest, "block_id"].isin(blocks)], te, d)
    if len(ver) >= 200:  # verified-only spatial CV (clean labels on both sides)
        sub = d.loc[te]
        for f in sorted(sub.fold_spatial.unique()):
            tr, tf = sub.index[sub.fold_spatial != f], sub.index[sub.fold_spatial == f]
            if sub.loc[tf, "label"].nunique() < 2:
                continue
            for m in a.models:
                run(rows, {**base, "variant": "verified_only_spatial5", "fold": f"f{f}", "rep": 0}, m, num, cat, 0, tr, tf, d)


def prevalence(a, region, d, num, cat, fs, rows):
    for f in sorted(d.fold_spatial.unique()):
        tr_all, te_all = d.index[d.fold_spatial != f], d.index[d.fold_spatial == f]
        for ratio in a.ratios:
            for rep in range(a.reps):
                rng = np.random.default_rng(100 * rep + ratio)
                thin = lambda idx: idx[(d.loc[idx, "label"] == 0).to_numpy() | (rng.random(len(idx)) < 1 / ratio)]  # noqa: E731
                te = thin(te_all)
                for train_mode, tr in (("balanced", tr_all), ("imbalanced", thin(tr_all))):
                    for m in a.models:
                        run(rows, {"region": region, "feature_set": fs, "fold": f"f{f}", "neg_per_pos": ratio, "train": train_mode, "rep": rep},
                            m, num, cat, rep, tr, te, d, extra_metrics=True)


EXPERIMENTS = {"county": county, "clean": clean, "prevalence": prevalence}


def summarize():
    txt = ["# Realistic-regime benchmark\n"]
    for exp in EXPERIMENTS:
        files = sorted(glob.glob(f"{OUT}/{exp}/runs_*.csv"))
        if not files:
            continue
        r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        keys = [c for c in r.columns if c not in ("n_train", "pos_train", "n_test", "pos_test", "roc_auc", "pr_auc", "seconds", "brier",
                                                  "log_loss", "ece", "mean_p", "prevalence")]
        r = r.drop_duplicates(subset=keys, keep="first")  # model shards (e.g. tuned baselines run separately) repeat nothing
        txt.append(f"\n# {exp}\n")
        if exp == "county":
            for (reg, fs), g in r.groupby(["region", "feature_set"]):
                piv = g.pivot_table(index=["test", "k"], columns="model", values="roc_auc", aggfunc="mean")
                n = g.groupby(["test", "k"]).county.nunique().rename("counties")
                txt.append(f"\n## {reg} / {fs}: mean AUC over counties x seeds\n\n" + pd.concat([n, piv.round(4)], axis=1).to_markdown() + "\n")
        elif exp == "clean":
            for (reg, fs), g in r.groupby(["region", "feature_set"]):
                piv = g.pivot_table(index="variant", columns="model", values=["roc_auc", "pr_auc"], aggfunc="mean").round(4)
                txt.append(f"\n## {reg} / {fs} (n_verified = {int(g.n_verified.iloc[0])})\n\n" + piv.to_markdown() + "\n")
        else:
            for (reg, fs), g in r.groupby(["region", "feature_set"]):
                piv = g.pivot_table(index=["neg_per_pos", "train"], columns="model",
                                    values=["roc_auc", "pr_auc", "brier", "log_loss", "ece"], aggfunc="mean").round(4)
                for metric in ("roc_auc", "pr_auc", "brier", "log_loss", "ece"):
                    txt.append(f"\n## {reg} / {fs}: {metric}\n\n" + piv[metric].to_markdown() + "\n")
    open(f"{OUT}/summary.md", "w").write("\n".join(txt))
    print("\n".join(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=list(EXPERIMENTS) + ["summarize"])
    ap.add_argument("--regions", nargs="*", default=["florida", "minnesota"])
    ap.add_argument("--feature-sets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--k", nargs="*", type=int, default=[25, 50, 100])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ratios", nargs="*", type=int, default=[10, 20])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--suffix", default="", help="runs file suffix, so model shards (e.g. --models xgboost_tuned) don't overwrite")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.experiment == "summarize":
        return summarize()
    out = f"{OUT}/{a.experiment}"
    os.makedirs(out, exist_ok=True)
    for region in a.regions:
        d, sets = load(region)
        for fs, (num, cat) in sets.items():
            if a.feature_sets and fs not in a.feature_sets:
                continue
            rows = []
            SAVE_PATH[0] = f"{out}/runs_{region}_{fs}{a.suffix}.csv"
            EXPERIMENTS[a.experiment](a, region, d, num, cat, fs, rows)
            pd.DataFrame(rows).to_csv(SAVE_PATH[0], index=False)
    summarize()


if __name__ == "__main__":
    main()
