"""Cost and deployment numbers for the foundation models: what "no training" costs at inference time.

  scaling   context-size scaling on the two largest tasks: Kentucky scott (34k train rows -> 15k test) and Florida
            no_landcover × ring × temporal. Context = stratified subsample of the training fold (n = 500 ... full);
            reports AUC, fit seconds, predict seconds, latency per 10k queries and peak GPU memory per fit.
  cpu       CPU-only timing of the "county model": subedi5 features (5), 500-row context, 500 queries, FM_DEVICE=cpu.
            Run it on a quiet box: it is sensitive to CPU load.

  FM_PRECISION=fp32 python scripts/fm_cost.py scaling
  python scripts/fm_cost.py cpu
Output: results/fm_cost/{scaling.csv, cpu.csv, summary.md}
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, f"{ROOT}/scripts")
from common import MODELS  # noqa: E402
from feature_sets import feature_sets as fl_feature_sets  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402
from fm_scenarios import TASKS, FULL  # noqa: E402

ALL = {**MODELS, **FM_MODELS}
OUT = f"{ROOT}/results/fm_cost"
SIZES = [500, 1000, 2000, 5000, 10000, 20000, FULL]


def timed_fit_predict(name, num, cat, seed, Xtr, ytr, Xte):
    import torch
    gpu = torch.cuda.is_available() and name in FM_MODELS
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    est = ALL[name](num, cat, seed).fit(Xtr, ytr)
    if gpu:
        torch.cuda.synchronize()
    t1 = time.time()
    p = est.predict_proba(Xte)[:, 1]
    if gpu:
        torch.cuda.synchronize()
    t2 = time.time()
    mem = torch.cuda.max_memory_allocated() / 2**30 if gpu else np.nan
    if name in FM_MODELS:
        import gc
        del est
        gc.collect()
        torch.cuda.empty_cache()
    return p, t1 - t0, t2 - t1, mem


def scaling(a):
    rows = []
    # host and precision of the timing run, so a later `summarize` on another machine reports where the numbers come from
    with open(f"{OUT}/run_info.json", "w") as fh:
        json.dump({"host": platform.node(), "FM_PRECISION": os.environ.get("FM_PRECISION", "auto")}, fh, indent=1)
    for task, fold_name in (("kentucky", "test_Scott"), ("florida", "temporal")):
        if task == "kentucky":
            from run_kentucky_baselines import load as ky_load, splits as ky_splits
            d = ky_load().reset_index(drop=True)
            num, cat = TASKS["kentucky"]()[1:3]
            folds = [(f, np.asarray(tr), np.asarray(te)) for f, tr, te in ky_splits(d, "scott", 1)]
        else:
            from run_florida_baselines import DATA, feature_cols, splits
            d = pd.read_parquet(DATA)
            num, cat = feature_cols(d, False, "no_landcover")
            d = d[(d.label == 1) | (d.neg_strategy == "ring")].reset_index(drop=True)
            folds = [("temporal", np.flatnonzero(tr), np.flatnonzero(te)) for _, tr, te in splits(d, "temporal")]
        y = d.label.to_numpy()
        for fold, tr, te in folds:
            for n in SIZES:
                if n != FULL and n >= len(tr):
                    continue
                for rep in range(a.reps if n != FULL else 1):
                    idx = tr if n == FULL else train_test_split(tr, train_size=n, stratify=y[tr], random_state=rep)[0]
                    for m in a.models:
                        try:
                            p, fit_s, pred_s, mem = timed_fit_predict(m, num, cat, rep, d.loc[idx, num + cat], y[idx], d.loc[te, num + cat])
                            auc, pr = roc_auc_score(y[te], p), average_precision_score(y[te], p)
                        except Exception as e:  # noqa: BLE001
                            print(f"FAIL {m} n={n}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                            auc = pr = fit_s = pred_s = mem = np.nan
                        rows.append({"task": task, "fold": fold, "n_context": len(idx), "rep": rep, "model": m, "n_query": len(te),
                                     "roc_auc": auc, "pr_auc": pr, "fit_s": fit_s, "predict_s": pred_s,
                                     "latency_s_per_10k": pred_s / len(te) * 1e4, "peak_gpu_gib": mem})
                        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
                    pd.DataFrame(rows).to_csv(f"{OUT}/scaling.csv", index=False)
    summarize()


def cpu(a):
    """Runs each model in a fresh CPU-only subprocess so CUDA is never initialised."""
    rows = []
    for m in a.models:
        for rep in range(a.reps):
            code = f"""
import os, sys, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["FM_DEVICE"] = "cpu"; os.environ["FM_PRECISION"] = "fp32"
os.environ.setdefault("FM_CPU_THREADS", "{a.threads}"); os.environ["OMP_NUM_THREADS"] = "{a.threads}"
sys.path.insert(0, "{ROOT}/baselines"); sys.path.insert(0, "{ROOT}/scripts")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from common import MODELS
from fm_models import FM_MODELS
from feature_sets import feature_sets
d = pd.read_parquet("{ROOT}/data/processed/florida/dataset.parquet")
d = d[(d.label == 1) | (d.neg_strategy == "ring")].reset_index(drop=True)
num, cat = feature_sets(d)["subedi5"]
tr, te = train_test_split(np.arange(len(d)), train_size=500, test_size=500, stratify=d.label, random_state={rep})
t0 = time.time(); est = {{**MODELS, **FM_MODELS}}["{m}"](num, cat, {rep}).fit(d.loc[tr, num + cat], d.label.to_numpy()[tr]); t1 = time.time()
p = est.predict_proba(d.loc[te, num + cat])[:, 1]; t2 = time.time()
print("RESULT", json.dumps(dict(roc_auc=float(roc_auc_score(d.label.to_numpy()[te], p)), fit_s=t1 - t0, predict_s=t2 - t1)))
"""
            t0 = time.time()
            out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True)
            res = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")]
            r = {"model": m, "rep": rep, "threads": a.threads, "n_context": 500, "n_query": 500, "n_features": 5,
                 "wall_s_incl_import": time.time() - t0}
            if res:
                import json
                r.update(json.loads(res[0][7:]))
            else:
                print(out.stderr[-500:], flush=True)
            rows.append(r)
            print(r, flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/cpu.csv", index=False)
    summarize()


def summarize():
    info = {"host": platform.node(), "FM_PRECISION": os.environ.get("FM_PRECISION", "auto")}
    if os.path.exists(f"{OUT}/run_info.json"):
        with open(f"{OUT}/run_info.json") as fh:
            info = json.load(fh)
    txt = [f"# Cost of inference-time learning\n\nHost: {info['host']}, FM_PRECISION={info['FM_PRECISION']}.\n"]
    if os.path.exists(f"{OUT}/scaling.csv"):
        r = pd.read_csv(f"{OUT}/scaling.csv")
        for task, g in r.groupby("task"):
            piv = g.pivot_table(index="n_context", columns="model", values=["roc_auc", "fit_s", "predict_s", "latency_s_per_10k", "peak_gpu_gib"],
                                aggfunc="mean").round(3)
            txt.append(f"\n## {task}: context-size scaling (n_query = {int(g.n_query.iloc[0])})\n")
            for metric in ("roc_auc", "fit_s", "predict_s", "latency_s_per_10k", "peak_gpu_gib"):
                if metric in piv:
                    txt.append(f"\n### {metric}\n\n" + piv[metric].dropna(axis=1, how="all").to_markdown() + "\n")
    if os.path.exists(f"{OUT}/cpu.csv"):
        c = pd.read_csv(f"{OUT}/cpu.csv")
        txt.append("\n## CPU-only county model (subedi5 features, 500 context rows, 500 queries)\n\n" +
                   c.groupby("model")[[x for x in ("roc_auc", "fit_s", "predict_s", "wall_s_incl_import") if x in c]].mean().round(3).to_markdown() + "\n")
    open(f"{OUT}/summary.md", "w").write("\n".join(txt))
    print("\n".join(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scaling", "cpu", "summarize"])
    ap.add_argument("--models", nargs="*", default=["xgboost", "lightgbm", "tabpfn", "tabicl", "kumo_tabular"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    globals()[a.cmd](a) if a.cmd != "summarize" else summarize()


if __name__ == "__main__":
    main()
