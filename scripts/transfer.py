"""Cross-state transfer on the shared feature schema: a fixed context from one state, queries in another.

Regions must have data/processed/<region>/dataset.parquet with the shared columns (scripts/shared_features.py):
florida_shared, minnesota, ... Ring negatives only (the "global" negatives are mostly geography).

For every target spatial fold f (the committed 25 km-block folds of the target table):
  test        = target rows with fold_spatial == f
  local pool  = target rows with fold_spatial != f
  context     = source table (all rows)  +  k local positives and k local ring negatives (k in --k, "all" = whole pool)
Modes written per run: "transfer" (source + k local), "local" (k local only; the few-shot curve without a source),
and k = all with/without the source. GBDTs are refit on each context; the FMs just get a new context.
2 seeds for the local sample when k < all. Feature sets: sat_nolc (34) and sat_nolc_rel (drops the absolute
elevation and rainfall, which differ systematically between states).

  python scripts/transfer.py --source florida_shared --target minnesota --models xgboost random_forest tabpfn tabicl kumo_tabular
  python scripts/transfer.py --summarize
Outputs: results/transfer/runs_<source>_<target>_<fs>.csv, summary.md
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, os.path.dirname(__file__))
from fm_scenarios import run_fit  # noqa: E402
from run_region_baselines import FEATURE_SETS  # noqa: E402

OUT = f"{ROOT}/results/transfer"
ABSOLUTE = {"elev_m", "ppt_annual_mm"}
FEATURE_SETS = {**FEATURE_SETS, "sat_nolc_rel": ([c for c in FEATURE_SETS["sat_nolc"][0] if c not in ABSOLUTE], [])}
ALL_K = 10 ** 9


def load(region):
    d = pd.read_parquet(f"{ROOT}/data/processed/{region}/dataset.parquet")
    return d[(d.label == 1) | (d.neg_strategy == "ring")].reset_index(drop=True)


def sample_local(pool, k, rng):
    pos, neg = pool.index[pool.label == 1], pool.index[pool.label == 0]
    if k >= ALL_K:
        return pool.index
    return np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False), rng.choice(neg, min(k, len(neg)), replace=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="florida_shared")
    ap.add_argument("--target", default="minnesota")
    ap.add_argument("--feature-sets", nargs="*", default=["sat_nolc", "sat_nolc_rel"])
    ap.add_argument("--k", nargs="*", type=int, default=[0, 50, 250, 1000, ALL_K])
    ap.add_argument("--models", nargs="*", default=["logreg", "random_forest", "xgboost", "tabpfn", "tabicl", "kumo_tabular"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.summarize:
        return summarize()

    src, tgt = load(a.source), load(a.target)
    for fs in a.feature_sets:
        num, cat = FEATURE_SETS[fs]
        cols = num + cat
        rows, path = [], f"{OUT}/runs_{a.source}_{a.target}_{fs}.csv"
        for f in a.folds or sorted(tgt.fold_spatial.unique()):
            te = tgt[tgt.fold_spatial == f]
            pool = tgt[tgt.fold_spatial != f]
            for k in a.k:
                for seed in range(a.seeds if 0 < k < ALL_K else 1):
                    loc = sample_local(pool, k, np.random.default_rng(1000 * seed + k))
                    for mode in ("transfer", "local"):
                        if mode == "local" and k == 0:
                            continue
                        ctx = pd.concat([src, pool.loc[loc]]) if mode == "transfer" else pool.loc[loc]
                        if ctx.label.nunique() < 2:
                            continue
                        base = {"source": a.source, "target": a.target, "feature_set": fs, "mode": mode, "k_local": k if k < ALL_K else "all",
                                "fold": f"f{f}", "rep": seed}
                        for m in a.models:
                            run_fit(rows, base, m, num, cat, seed, ctx[cols], ctx.label.to_numpy(), te[cols], te.label.to_numpy())
                    pd.DataFrame(rows).to_csv(path, index=False)
    summarize()


def summarize():
    r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/runs_*.csv"))], ignore_index=True)
    r["k_local"] = r["k_local"].astype(str)
    order = sorted(r.k_local.unique(), key=lambda s: (s == "all", int(s) if s != "all" else 0))
    txt = ["# Cross-state transfer (shared schema, ring negatives, target spatial folds)\n",
           "Mean test ROC-AUC over the target's 5 spatial folds (and 2 local-sample seeds). `transfer` = source table + k local "
           "positives and k local negatives in the context; `local` = the k local rows only. k = all uses the whole local training "
           "fold, i.e. the ordinary spatial CV with (transfer) or without (local) the other state prepended.\n"]
    for (s, t, fs), g in r.groupby(["source", "target", "feature_set"]):
        piv = g.pivot_table(index=["mode", "k_local"], columns="model", values="roc_auc", aggfunc="mean").reindex(
            pd.MultiIndex.from_product([["transfer", "local"], order], names=["mode", "k_local"])).dropna(how="all")
        txt.append(f"\n## {s} -> {t} ({fs})\n\n" + piv.round(4).to_markdown() + "\n")
    open(f"{OUT}/summary.md", "w").write("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
