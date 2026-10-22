"""Seed / ensemble-size sensitivity on the headline Florida settings (no_landcover x ring x {spatial5, temporal}).

Pooled OOF AUC + mean per-fold AUC for GBDTs over seeds and FMs over seeds x n_estimators.
Usage: python scripts/fm_sensitivity.py [--seeds 0 1 2] [--n-est 8 32]   -> results/fm_sensitivity/
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
from common import MODELS  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402
from run_florida_baselines import DATA, feature_cols, splits  # noqa: E402

NEST_ENV = {"tabpfn": "TABPFN_N_ESTIMATORS", "kumo_tabular": "KUMO_N_ESTIMATORS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-est", nargs="*", type=int, default=[8, 32])
    ap.add_argument("--protocols", nargs="*", default=["spatial5", "temporal"])
    ap.add_argument("--gbdt", nargs="*", default=["xgboost", "lightgbm"])
    ap.add_argument("--fm", nargs="*", default=["tabpfn", "kumo_tabular"])
    a = ap.parse_args()
    out = f"{ROOT}/results/fm_sensitivity"
    os.makedirs(out, exist_ok=True)

    data = pd.read_parquet(DATA)
    num, cat = feature_cols(data, feature_set="no_landcover")
    df = data[(data.label == 1) | (data.neg_strategy == "ring")].reset_index(drop=True)
    configs = [(m, s, None) for m in a.gbdt for s in a.seeds] + [(m, s, n) for m in a.fm for n in a.n_est for s in a.seeds]
    rows = []
    for prot in a.protocols:
        for model, seed, n_est in configs:
            if n_est:
                os.environ[NEST_ENV[model]] = str(n_est)
            factory = {**MODELS, **FM_MODELS}[model]
            oof, fold_auc, t0 = pd.Series(index=df.index, dtype=float), [], time.time()
            for _, tr, te in splits(df, prot):
                p = factory(num, cat, seed).fit(df.loc[tr, num + cat], df.loc[tr, "label"]).predict_proba(df.loc[te, num + cat])[:, 1]
                oof[te] = p
                fold_auc.append(roc_auc_score(df.loc[te, "label"], p))
            m = oof.notna()
            rows.append({"protocol": prot, "model": model, "n_est": n_est or "-", "seed": seed,
                         "pooled_auc": roc_auc_score(df.label[m], oof[m]), "mean_fold_auc": sum(fold_auc) / len(fold_auc),
                         "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(f"{out}/runs.csv", index=False)
    r = pd.DataFrame(rows)
    g = r.groupby(["protocol", "model", "n_est"], sort=False)[["pooled_auc", "mean_fold_auc", "seconds"]]
    summ = g.mean().round(4).join(g.std().round(4)[["pooled_auc", "mean_fold_auc"]], rsuffix="_sd_over_seeds")
    with open(f"{out}/summary.md", "w") as fh:
        fh.write(f"# Seed / n_estimators sensitivity: Florida no_landcover x ring\n\nseeds={a.seeds}, "
                 f"FM_PRECISION={os.environ.get('FM_PRECISION', 'auto')}\n\n" + summ.reset_index().to_markdown(index=False) + "\n")
    print(open(f"{out}/summary.md").read())


if __name__ == "__main__":
    main()
