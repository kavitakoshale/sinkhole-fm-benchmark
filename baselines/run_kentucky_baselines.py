"""Kentucky Bluegrass: sinkhole vs non-sinkhole LiDAR depressions (replication of Zhu et al., 2020).

Data: data/processed/kentucky/depressions_<County>.gpkg from scripts/kentucky/01_extract_depressions.py
Features: the paper's 10 morphometric variables.

Protocols
  paper80_20  pooled Bourbon+Woodford+Jessamine, random stratified 80/20 split, repeated 5 seeds (paper protocol)
  loco        leave-one-county-out over the 3 training counties (spatial transfer)
  scott       train on the 3 counties, test on Scott County (the paper's application county)

Paper reference (Table 2 / Figs 3-4, test set): NN AUC 0.950 PR-AUC 0.860; RF AUC 0.947 PR-AUC 0.851;
RUSBoost AUC 0.942; SVM PR-AUC 0.828; LR / NB lower.  All AUCs > 0.87.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import MODELS, PAPER_MODELS, metrics  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = f"{ROOT}/data/processed/kentucky"
OUT = f"{ROOT}/results/kentucky"
FEATS = ["perimeter_m", "area_m2", "compactness", "depth_max_m", "depth_mean_m", "depth_std_m", "depth_sum_m",
         "ring_slope_mean_deg", "ring_slope_std_deg", "depth_index"]
TRAIN_COUNTIES = ["Bourbon", "Woodford", "Jessamine"]
ALL_MODELS = {**PAPER_MODELS, **MODELS, **FM_MODELS}
DEFAULT_MODELS = [m for m in {**PAPER_MODELS, **MODELS} if m != "svm_poly2"]  # SVM is O(n^2) at ~27k rows


def load():
    """Rebuild the table from the per-county gpkgs when present (local pipeline), else use the committed parquet."""
    gpkgs = [f"{PROC}/depressions_{c}.gpkg" for c in TRAIN_COUNTIES + ["Scott"]]
    if not any(os.path.exists(p) for p in gpkgs):
        return pd.read_parquet(f"{PROC}/depressions_all.parquet")
    df = pd.concat([pd.DataFrame(gpd.read_file(p).drop(columns="geometry")) for p in gpkgs if os.path.exists(p)],
                   ignore_index=True)
    df.to_parquet(f"{PROC}/depressions_all.parquet", index=False)
    return df


def splits(df, protocol, n_rep):
    tr3 = df[df.county.isin(TRAIN_COUNTIES)]
    if protocol == "paper80_20":
        for r in range(n_rep):
            tr, te = train_test_split(tr3.index, test_size=0.2, stratify=tr3.label, random_state=r)
            yield f"rep{r}", tr, te
    elif protocol == "loco":
        for c in TRAIN_COUNTIES:
            if (tr3.county == c).any():
                yield f"test_{c}", tr3.index[tr3.county != c], tr3.index[tr3.county == c]
    elif protocol == "scott" and (df.county == "Scott").any():
        yield "test_Scott", tr3.index, df.index[df.county == "Scott"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="*", default=["paper80_20", "loco", "scott"])
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help=f"any of {list(ALL_MODELS)}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-rep", type=int, default=5)
    a = ap.parse_args()
    out = OUT + (f"_{a.tag}" if a.tag else "")
    os.makedirs(out, exist_ok=True)
    df = load()
    print(df.groupby("county").label.agg(["count", "sum", "mean"]))
    rows = []
    for prot in a.protocols:
        oof = []
        for fold, tr, te in splits(df, prot, a.n_rep):
            Xtr, ytr, Xte, yte = df.loc[tr, FEATS], df.loc[tr, "label"], df.loc[te, FEATS], df.loc[te, "label"]
            pred = df.loc[te, ["dep_id", "county", "label"]].assign(fold=fold)
            for name in a.models:
                t0 = time.time()
                p = ALL_MODELS[name](FEATS, [], 0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                yhat = (p >= 0.5).astype(int)
                m = metrics(yte, p)
                m.update(recall=recall_score(yte, yhat), precision=precision_score(yte, yhat, zero_division=0),
                         average_accuracy=balanced_accuracy_score(yte, yhat))
                rows.append({"protocol": prot, "model": name, "fold": fold, **m, "fit_s": time.time() - t0})
                pred[name] = p
                print(f"{prot:10s} {fold:14s} {name:16s} AUC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f}", flush=True)
            oof.append(pred)
        if oof:
            pd.concat(oof).to_parquet(f"{out}/pred_{prot}.parquet", index=False)
    long = pd.DataFrame(rows)
    long.to_csv(f"{out}/metrics_long.csv", index=False)
    cols = ["roc_auc", "pr_auc", "recall", "precision", "accuracy", "average_accuracy", "f1", "brier"]
    g = long.groupby(["protocol", "model"])[cols]
    summ = (g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)).reset_index()
    with open(f"{out}/summary.md", "w") as fh:
        fh.write("# Kentucky Bluegrass depression classification (Zhu et al. 2020 replication)\n\n")
        fh.write(df.groupby("county").label.agg(n="count", sinkholes="sum", frac="mean").round(3).to_markdown() + "\n")
        for prot, gg in summ.groupby("protocol", sort=False):
            fh.write(f"\n## protocol={prot}\n\n" + gg.drop(columns="protocol").to_markdown(index=False) + "\n")
    print(open(f"{out}/summary.md").read())


if __name__ == "__main__":
    main()
