"""Region-generic susceptibility benchmark on the shared-schema tables: data/processed/<region>/dataset.parquet.

Same protocols and file layout as run_florida_baselines.py (random5 / spatial5 / temporal; neg {global, ring};
oof_<neg>_<protocol>.parquet with the Florida key columns), so compare_models.py works unchanged with `--task <region>`.
Feature sets come from the shared schema (scripts/shared_features.py):
  terrain    21 DEM derivatives
  sat_nolc   terrain + DEM-derived depressions + annual rainfall      (headline: no land cover)
  sat_all    sat_nolc + WorldCover class / fractions

Models: common.MODELS, fm_models.FM_MODELS and tuned.TUNED_MODELS (tuned models get the 25 km block ids as inner groups).

  python baselines/run_region_baselines.py --region minnesota --models logreg xgboost tabpfn --tag fm
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import MODELS, metrics, summarize  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402
from run_florida_baselines import splits  # noqa: E402
from tuned import TUNED_MODELS  # noqa: E402

ALL_MODELS = {**MODELS, **FM_MODELS, **TUNED_MODELS}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TERRAIN = ["elev_m", "slope_deg", "laplacian_curv"] + [f"{s}_{r}m" for r in (150, 300, 600) for s in
           ("tpi", "relief", "elev_std", "depth_below_max", "height_above_min", "slope_mean")]
DEP = ["dep_fill_depth_m", "dep_inside", "dep_dist_m"] + [f"dep_{k}_{r}m" for r in (500, 1000, 2000) for k in ("count", "area_frac", "max_depth")]
LC_NUM = [f"lc_{c}_frac_{r}m" for r in (250, 1000) for c in ("tree", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "mangrove")]
FEATURE_SETS = {
    "terrain": (TERRAIN, []),
    "sat_nolc": (TERRAIN + DEP + ["ppt_annual_mm"], []),
    "sat_all": (TERRAIN + DEP + ["ppt_annual_mm"] + LC_NUM, ["lc_class"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--neg", nargs="*", default=["global", "ring"])
    ap.add_argument("--protocols", nargs="*", default=["random5", "spatial5", "temporal"])
    ap.add_argument("--models", nargs="*", default=list(MODELS), help=f"any of {list(ALL_MODELS)}")
    ap.add_argument("--feature-set", default="sat_nolc", choices=list(FEATURE_SETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = f"{ROOT}/results/{a.region}" + (f"_{a.tag}" if a.tag else "") + (f"_{a.feature_set}" if a.feature_set != "sat_nolc" else "")
    os.makedirs(out, exist_ok=True)

    data = pd.read_parquet(f"{ROOT}/data/processed/{a.region}/dataset.parquet")
    num, cat = FEATURE_SETS[a.feature_set]
    assert all(c in data for c in num + cat), [c for c in num + cat if c not in data]
    print(f"{a.region}: {len(data)} rows, {len(num)} numeric + {len(cat)} categorical features ({a.feature_set})")
    rows = []
    for neg in a.neg:
        df = data[(data.label == 1) | (data.neg_strategy == neg)].reset_index(drop=True)
        for prot in a.protocols:
            oof = df[["pid", "label", "fold_random", "fold_spatial", "split_temporal"]].copy()
            for f, tr, te in splits(df, prot):
                Xtr, ytr, Xte, yte = df.loc[tr, num + cat], df.loc[tr, "label"], df.loc[te, num + cat], df.loc[te, "label"]
                for name in a.models:
                    t0 = time.time()
                    est = ALL_MODELS[name](num, cat, a.seed)
                    est.fit(Xtr, ytr, **({"groups": df.loc[tr, "block_id"].to_numpy()} if getattr(est, "needs_groups", False) else {}))
                    p = est.predict_proba(Xte)[:, 1]
                    dt = time.time() - t0
                    rows.append({"neg_strategy": neg, "protocol": prot, "model": name, "fold": f, **metrics(yte, p), "fit_s": dt})
                    oof.loc[te, name] = p
                    print(f"{neg:6s} {prot:8s} f{f} {name:18s} AUC={rows[-1]['roc_auc']:.4f} PR={rows[-1]['pr_auc']:.4f} ({dt:.1f}s)", flush=True)
            oof.to_parquet(f"{out}/oof_{neg}_{prot}.parquet", index=False)

    long = pd.DataFrame(rows)
    long.to_csv(f"{out}/metrics_long.csv", index=False)
    summ = summarize(long, ["neg_strategy", "protocol", "model"])
    with open(f"{out}/summary.md", "w") as fh:
        fh.write(f"# {a.region.title()} sinkhole baselines (shared schema: {a.feature_set})\n\nfeatures: {len(num)} numeric + {len(cat)} "
                 f"categorical; seed={a.seed}; rows={len(data)}\n\n")
        for (neg, prot), g in summ.groupby(["neg_strategy", "protocol"], sort=False):
            fh.write(f"\n## negatives={neg}, protocol={prot}\n\n" + g.drop(columns=["neg_strategy", "protocol"]).to_markdown(index=False) + "\n")
    with open(f"{out}/run_info.json", "w") as fh:
        import sklearn, xgboost  # noqa: E401
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "python": platform.python_version(), "sklearn": sklearn.__version__,
                   "xgboost": xgboost.__version__, "host": platform.node(),
                   "env": {k: v for k, v in os.environ.items() if k.startswith(("FM_", "TABPFN_MODEL", "TABICL_", "KUMO_", "TUNE_")) or k == "CUDA_VISIBLE_DEVICES"},
                   "args": vars(a), "numeric_features": num, "categorical_features": cat, "n_rows": len(data)}, fh, indent=1)
    print(open(f"{out}/summary.md").read())


if __name__ == "__main__":
    main()
