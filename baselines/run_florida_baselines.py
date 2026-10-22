"""Florida sinkhole susceptibility baselines.

Datasets  : positives (FGS SIR) + negatives from one strategy: "global" or "ring" (500-5000 m)
Protocols : random5   - stratified random 5-fold CV (what most papers report; optimistic)
            spatial5  - 25 km block 5-fold CV (spatial extrapolation)
            temporal  - train on events < 2010, test on events >= 2010 (forecasting future reports)
Models    : baselines/common.py MODELS + expert FGS favorability map (score only)

Outputs (results/florida/):
  metrics_long.csv     one row per (neg_strategy, protocol, model, fold)
  summary.md           mean ± std table
  oof_<neg>_<protocol>.parquet  out-of-fold predictions of every model (for FM comparisons / stats tests)
"""
import os

# more than 4 BLAS/OpenMP threads per process oversubscribe the cores (50-90x slowdown measured)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import MODELS, metrics, score_only_metrics, summarize  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402

ALL_MODELS = {**MODELS, **FM_MODELS}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = f"{ROOT}/data/processed/florida/dataset.parquet"
OUT = f"{ROOT}/results/florida"

META = ["pid", "ref_num", "label", "event_year", "accuracy", "true_sink", "county", "neg_strategy", "lon", "lat",
        "x", "y", "block_id", "fold_spatial", "fold_random", "split_temporal", "fgs_favorability_value"]
CAT = ["lc_class", "floridan_recharge_class", "drastic_setting", "geo_unit", "geo_lithology", "geo_age",
       "envgeo_category", "sinkhole_type_area", "geomorph_district"]


def feature_cols(df, include_xy=False, feature_set="all"):
    """feature_set: 'all' | 'no_landcover' (drops WorldCover features, which mostly encode the urban reporting bias
    of the citizen-reported SIR inventory -> isolates the geologic/hydrogeologic signal)."""
    cat = [c for c in CAT if c in df]
    num = [c for c in df.columns if c not in META and c not in cat]
    if feature_set == "no_landcover":
        cat = [c for c in cat if not c.startswith("lc_")]
        num = [c for c in num if not c.startswith("lc_")]
    if include_xy:
        num += ["x", "y"]
    return num, cat


def splits(df, protocol):
    if protocol == "random5":
        for f in range(5):
            yield f, df.fold_random != f, df.fold_random == f
    elif protocol == "spatial5":
        for f in range(5):
            yield f, df.fold_spatial != f, df.fold_spatial == f
    elif protocol == "temporal":
        yield 0, df.split_temporal == "train", df.split_temporal == "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", nargs="*", default=["global", "ring"])
    ap.add_argument("--protocols", nargs="*", default=["random5", "spatial5", "temporal"])
    ap.add_argument("--models", nargs="*", default=list(MODELS), help=f"any of {list(ALL_MODELS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-xy", action="store_true", help="add projected x/y as features")
    ap.add_argument("--feature-set", default="all", choices=["all", "no_landcover"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = OUT + (f"_{a.tag}" if a.tag else "")
    os.makedirs(out, exist_ok=True)

    data = pd.read_parquet(DATA)
    num, cat = feature_cols(data, a.include_xy, a.feature_set)
    print(f"{len(num)} numeric + {len(cat)} categorical features")
    rows = []
    for neg in a.neg:
        df = data[(data.label == 1) | (data.neg_strategy == neg)].reset_index(drop=True)
        for prot in a.protocols:
            oof = df[["pid", "label", "fold_random", "fold_spatial", "split_temporal"]].copy()
            for f, tr, te in splits(df, prot):
                Xtr, ytr, Xte, yte = df.loc[tr, num + cat], df.loc[tr, "label"], df.loc[te, num + cat], df.loc[te, "label"]
                # expert map: FGS favorability (WofE) class value, higher = more favorable
                m = score_only_metrics(yte.values, df.loc[te, "fgs_favorability_value"].values.astype(float))
                rows.append({"neg_strategy": neg, "protocol": prot, "model": "fgs_favorability_map", "fold": f, **m, "fit_s": 0})
                oof.loc[te, "fgs_favorability_map"] = df.loc[te, "fgs_favorability_value"]
                for name in a.models:
                    t0 = time.time()
                    est = ALL_MODELS[name](num, cat, a.seed).fit(Xtr, ytr)
                    p = est.predict_proba(Xte)[:, 1]
                    dt = time.time() - t0
                    rows.append({"neg_strategy": neg, "protocol": prot, "model": name, "fold": f, **metrics(yte, p), "fit_s": dt})
                    oof.loc[te, name] = p
                    print(f"{neg:6s} {prot:8s} f{f} {name:16s} AUC={rows[-1]['roc_auc']:.4f} PR={rows[-1]['pr_auc']:.4f} ({dt:.1f}s)", flush=True)
            oof.to_parquet(f"{out}/oof_{neg}_{prot}.parquet", index=False)

    long = pd.DataFrame(rows)
    long.to_csv(f"{out}/metrics_long.csv", index=False)
    summ = summarize(long, ["neg_strategy", "protocol", "model"])
    with open(f"{out}/summary.md", "w") as fh:
        fh.write(f"# Florida sinkhole baselines\n\nfeatures: {len(num)} numeric + {len(cat)} categorical; seed={a.seed}; include_xy={a.include_xy}; feature_set={a.feature_set}\n\n")
        for (neg, prot), g in summ.groupby(["neg_strategy", "protocol"], sort=False):
            fh.write(f"\n## negatives={neg}, protocol={prot}\n\n")
            fh.write(g.drop(columns=["neg_strategy", "protocol"]).to_markdown(index=False))
            fh.write("\n")
    with open(f"{out}/run_info.json", "w") as fh:
        import sklearn, xgboost, lightgbm, catboost  # noqa: E401
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "python": platform.python_version(), "sklearn": sklearn.__version__,
                   "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
                   "env": {k: v for k, v in os.environ.items() if k.startswith(("FM_", "TABPFN_MODEL", "TABICL_", "KUMO_")) or k == "CUDA_VISIBLE_DEVICES"},
                   "args": vars(a), "numeric_features": num, "categorical_features": cat, "n_rows": len(data)}, fh, indent=1)
    print(open(f"{out}/summary.md").read())


if __name__ == "__main__":
    main()
