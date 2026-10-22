"""Tuned classical baselines on the committed B1 / B2 benchmarks, nested spatial CV, parallel over outer folds.

Uses exactly the committed folds of baselines/run_florida_baselines.py and run_kentucky_baselines.py and writes the same
file layout (metrics_long.csv, oof_*.parquet / pred_*.parquet, summary.md, run_info.json) plus tuning.csv (best
parameters, inner score, search time per outer fold), so compare_models.py can merge the tuned columns with the FM runs:

  python scripts/run_tuned_baselines.py --task florida --feature-set no_landcover --jobs 24
  python scripts/run_tuned_baselines.py --task kentucky --jobs 12
  python baselines/compare_models.py florida results/florida_no_landcover results/florida_tuned_no_landcover results/florida_fm_fp32_no_landcover

Budget knobs are env vars documented in baselines/tuned.py (TUNE_TRIALS, TUNE_INNER_FOLDS, TUNE_METRIC, ...). Every
(setting, fold, model) job is independent and cached, so a killed run resumes.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only; otherwise every worker grabs an idle 550 MB CUDA context

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
from common import metrics, summarize  # noqa: E402
from run_florida_baselines import DATA as FL_DATA, feature_cols, splits as fl_splits  # noqa: E402
from run_kentucky_baselines import FEATS as KY_FEATS, TRAIN_COUNTIES, load as ky_load, splits as ky_splits  # noqa: E402
from run_region_baselines import FEATURE_SETS as SHARED_SETS  # noqa: E402
from tuned import TUNED_MODELS, budget  # noqa: E402

DEFAULT_MODELS = ["xgboost_tuned", "xgboost_tuned_bag", "lightgbm_tuned", "lightgbm_tuned_bag", "catboost_tuned", "catboost_tuned_bag",
                  "random_forest_tuned", "logreg_tuned", "svm_rbf_tuned"]
KY_CELL_DEG = 0.05  # ~5 km inner spatial blocks for Kentucky (no block_id column there)


def ky_groups(df):
    return (np.floor(df.lat / KY_CELL_DEG).astype(int) * 10000 + np.floor(df.lon / KY_CELL_DEG).astype(int)).to_numpy()


def one_job(task, key, name, num, cat, Xtr, ytr, groups, Xte, yte, seed):
    t0 = time.time()
    est = TUNED_MODELS[name](num, cat, seed).fit(Xtr, ytr, groups=groups)
    p = est.predict_proba(Xte)[:, 1]
    m = metrics(yte, p)
    if task == "kentucky":
        yhat = (p >= 0.5).astype(int)
        m.update(recall=recall_score(yte, yhat), precision=precision_score(yte, yhat, zero_division=0),
                 average_accuracy=balanced_accuracy_score(yte, yhat))
    s = est.search_
    tune = {**key, "model": name, "inner_cv": est.inner_cv_, "inner_best": s["best_value"], "n_trials": s["n_trials"],
            "n_pruned": s.get("n_pruned"), "search_s": s.get("search_s"), "best_iters": json.dumps(s.get("best_iters")),
            "best_params": json.dumps(s["best_params"])}
    print(f"{' '.join(str(v) for v in key.values()):30s} {name:20s} AUC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} "
          f"inner={s['best_value']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return {**key, "model": name, **m, "fit_s": time.time() - t0}, pd.Series(p, index=Xte.index, name=name), tune


def florida_jobs(a):
    if a.task == "florida":
        data = pd.read_parquet(FL_DATA)
        num, cat = feature_cols(data, False, a.feature_set)
    else:  # any region on the shared schema (minnesota, florida_shared, ...)
        data = pd.read_parquet(f"{ROOT}/data/processed/{a.task}/dataset.parquet")
        num, cat = SHARED_SETS[a.feature_set]
    jobs, frames = [], {}
    for neg in a.neg:
        df = data[(data.label == 1) | (data.neg_strategy == neg)].reset_index(drop=True)
        for prot in a.protocols:
            frames[(neg, prot)] = df[["pid", "label", "fold_random", "fold_spatial", "split_temporal"]].copy()
            for f, tr, te in fl_splits(df, prot):
                for name in a.models:
                    jobs.append(((neg, prot), {"neg_strategy": neg, "protocol": prot, "fold": f}, name, num, cat,
                                 df.loc[tr, num + cat], df.loc[tr, "label"].to_numpy(), df.loc[tr, "block_id"].to_numpy(),
                                 df.loc[te, num + cat], df.loc[te, "label"].to_numpy()))
    return jobs, frames, num, cat, len(data)


def kentucky_jobs(a):
    df = ky_load()
    g = ky_groups(df)
    jobs, frames = [], {}
    for prot in a.protocols:
        frames[prot] = []
        for fold, tr, te in ky_splits(df, prot, a.n_rep):
            frames[prot].append(df.loc[te, ["dep_id", "county", "label"]].assign(fold=fold))
            for name in a.models:
                jobs.append((prot, {"protocol": prot, "fold": fold}, name, KY_FEATS, [], df.loc[tr, KY_FEATS], df.loc[tr, "label"].to_numpy(),
                             g[tr], df.loc[te, KY_FEATS], df.loc[te, "label"].to_numpy()))
    return jobs, frames, KY_FEATS, [], len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="florida | kentucky | <region> with data/processed/<region>/dataset.parquet")
    ap.add_argument("--feature-set", default=None, help="florida: all | no_landcover (default all); regions: terrain | sat_nolc | sat_all (default sat_nolc)")
    ap.add_argument("--neg", nargs="*", default=["global", "ring"], help="not for kentucky")
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help=f"any of {list(TUNED_MODELS)}")
    ap.add_argument("--n-rep", type=int, default=5, help="kentucky paper80_20 repetitions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8, help="parallel outer-fold jobs (each uses OMP_NUM_THREADS cores)")
    ap.add_argument("--tag", default="tuned")
    a = ap.parse_args()
    if a.protocols is None:
        a.protocols = ["paper80_20", "loco", "scott"] if a.task == "kentucky" else ["random5", "spatial5", "temporal"]
    default_fs = {"florida": "all", "kentucky": None}.get(a.task, "sat_nolc")
    a.feature_set = a.feature_set or default_fs
    out = f"{ROOT}/results/{a.task}_{a.tag}" + (f"_{a.feature_set}" if a.feature_set not in (default_fs, None) else "")
    os.makedirs(out, exist_ok=True)

    jobs, frames, num, cat, n_rows = kentucky_jobs(a) if a.task == "kentucky" else florida_jobs(a)
    print(f"{len(jobs)} jobs ({len(num)} numeric + {len(cat)} categorical features) -> {out}", flush=True)
    t0 = time.time()
    # "_bag" variants reuse the cached search of their refit twin, so queue them after all other jobs
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][2].endswith("_bag"))
    res = Parallel(n_jobs=a.jobs, backend="loky", verbose=0)(
        delayed(one_job)(a.task, *jobs[i][1:], a.seed) for i in order)
    res = [r for _, r in sorted(zip(order, res))]

    rows, tunes = [], []
    for (setting, *_), (row, pred, tune) in zip(jobs, res):
        rows.append(row)
        tunes.append(tune)
        if a.task != "kentucky":
            frames[setting].loc[pred.index, pred.name] = pred
        else:
            for fr in frames[setting]:
                if fr["fold"].iloc[0] == row["fold"]:
                    fr.loc[pred.index, pred.name] = pred
    if a.task != "kentucky":
        for (neg, prot), fr in frames.items():
            fr.to_parquet(f"{out}/oof_{neg}_{prot}.parquet", index=False)
    else:
        for prot, frs in frames.items():
            if frs:
                pd.concat(frs).to_parquet(f"{out}/pred_{prot}.parquet", index=False)

    long = pd.DataFrame(rows)
    long.to_csv(f"{out}/metrics_long.csv", index=False)
    pd.DataFrame(tunes).to_csv(f"{out}/tuning.csv", index=False)
    by = ["protocol", "model"] if a.task == "kentucky" else ["neg_strategy", "protocol", "model"]
    summ = summarize(long, by)
    b = {m: budget(m.replace("_tuned_bag", "").replace("_tuned", "")) for m in a.models}
    with open(f"{out}/summary.md", "w") as fh:
        fh.write(f"# {a.task.title()} tuned baselines (nested spatial CV)\n\n{len(num)} numeric + {len(cat)} categorical features; "
                 f"seed={a.seed}; inner CV = {budget('xgboost')['inner_folds']}-fold GroupKFold over spatial blocks; objective = inner "
                 f"{budget('xgboost')['metric']}; trials per outer fold: " + ", ".join(f"{m}={v['n_trials']}" for m, v in b.items()) +
                 f"; wall-clock {time.time() - t0:.0f}s with {a.jobs} parallel jobs.\n")
        for keys, g in summ.groupby(by[:-1], sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            fh.write(f"\n## {', '.join(f'{k}={v}' for k, v in zip(by[:-1], keys))}\n\n")
            fh.write(g.drop(columns=by[:-1]).to_markdown(index=False) + "\n")
    with open(f"{out}/run_info.json", "w") as fh:
        import sklearn, xgboost, lightgbm, catboost, optuna  # noqa: E401
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "python": platform.python_version(), "sklearn": sklearn.__version__,
                   "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
                   "optuna": optuna.__version__, "host": platform.node(), "cpu_count": os.cpu_count(),
                   "env": {k: v for k, v in os.environ.items() if k.startswith(("TUNE_", "OMP_"))}, "budgets": b,
                   "args": vars(a), "numeric_features": num, "categorical_features": cat, "n_rows": n_rows,
                   "wall_s": time.time() - t0}, fh, indent=1)
    print(open(f"{out}/summary.md").read())


if __name__ == "__main__":
    main()
