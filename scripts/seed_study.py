"""Seed-aware statistics for the pre-registered headline family.

Re-fits every model with seeds 0..S-1 on the committed folds of the headline settings, stores all predictions, and then
reports, per setting and model:
  * pooled OOF AUC per seed (mean ± sd over seeds) and the AUC of the seed-averaged prediction,
  * ΔAUC vs a reference model with a (spatial block × seed) bootstrap: each resample draws blocks with replacement AND an
    independent seed for the model and for the reference, so seed luck of either side is part of the interval,
  * a paired DeLong test on the seed-averaged pooled OOF predictions, Holm-corrected within the headline family.

Headline family (fixed before the seed runs were made, results/headline_family.md):
  florida   no_landcover × ring × spatial5, no_landcover × ring × temporal
  kentucky  loco
  minnesota sat_nolc × ring × spatial5

  python scripts/seed_study.py run --task florida --models xgboost lightgbm catboost tabpfn tabicl kumo_tabular --seeds 5
  python scripts/seed_study.py run --task florida --models xgboost_tuned --seeds 5 --jobs 10     # CPU, nested CV per seed
  python scripts/seed_study.py summarize
Output: results/seeds/<task>/preds_<models>.parquet (resumable: finished (setting, model, seed) triples are skipped),
        results/seeds/summary.md, results/seeds/summary.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
from common import MODELS  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402
from run_florida_baselines import DATA as FL_DATA, feature_cols, splits as fl_splits  # noqa: E402
from run_kentucky_baselines import FEATS as KY_FEATS, load as ky_load, splits as ky_splits  # noqa: E402
from run_region_baselines import FEATURE_SETS  # noqa: E402
from tuned import TUNED_MODELS  # noqa: E402

ALL = {**MODELS, **FM_MODELS, **TUNED_MODELS}
OUT = f"{ROOT}/results/seeds"
HEADLINE = {"florida": ["ring_spatial5", "ring_temporal"], "kentucky": ["loco"], "minnesota": ["ring_spatial5"]}


def ky_groups(df):
    return (np.floor(df.lat / 0.05).astype(int) * 10000 + np.floor(df.lon / 0.05).astype(int)).to_numpy()


def settings(task):
    """-> {setting: (df, num, cat, [(fold, tr_idx, te_idx)], groups, blocks)}; identical folds to the committed runs."""
    out = {}
    if task == "kentucky":
        d = ky_load().reset_index(drop=True)
        out["loco"] = (d, KY_FEATS, [], [(f, np.asarray(tr), np.asarray(te)) for f, tr, te in ky_splits(d, "loco", 1)],
                       ky_groups(d), np.arange(len(d)), "dep_id")
        return out
    data = pd.read_parquet(FL_DATA if task == "florida" else f"{ROOT}/data/processed/{task}/dataset.parquet")
    num, cat = feature_cols(data, False, "no_landcover") if task == "florida" else FEATURE_SETS["sat_nolc"]
    for s in HEADLINE[task]:
        neg, prot = s.split("_", 1)
        d = data[(data.label == 1) | (data.neg_strategy == neg)].reset_index(drop=True)
        folds = [(f"f{f}", np.flatnonzero(tr), np.flatnonzero(te)) for f, tr, te in fl_splits(d, prot)]
        out[s] = (d, num, cat, folds, d.block_id.to_numpy(), d.block_id.to_numpy(), "pid")
    return out


def fit_one(name, num, cat, seed, Xtr, ytr, groups, Xte):
    est = ALL[name](num, cat, seed)
    est = est.fit(Xtr, ytr, groups=groups) if getattr(est, "needs_groups", False) else est.fit(Xtr, ytr)
    p = est.predict_proba(Xte)[:, 1]
    if name in FM_MODELS:
        import gc
        import torch
        del est
        gc.collect()
        torch.cuda.empty_cache()
    return p


def load_preds(task):
    import glob
    fs = sorted(glob.glob(f"{OUT}/{task}/preds*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def run(a):
    os.makedirs(f"{OUT}/{a.task}", exist_ok=True)
    path = f"{OUT}/{a.task}/preds_{'_'.join(a.models)}.parquet"  # one file per model group: GPU and CPU groups can run concurrently
    done = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    prev = load_preds(a.task)
    have = set(map(tuple, prev[["setting", "model", "seed"]].drop_duplicates().to_numpy())) if len(prev) else set()
    jobs = []
    for s, (d, num, cat, folds, groups, _, key) in settings(a.task).items():
        for m in a.models:
            for seed in range(a.seeds):
                if (s, m, seed) in have:
                    continue
                for f, tr, te in folds:
                    jobs.append((s, m, seed, f, d.loc[tr, num + cat], d.label.to_numpy()[tr], groups[tr], d.loc[te, num + cat],
                                 d.loc[te, key].to_numpy(), d.label.to_numpy()[te], num, cat))
    print(f"{len(jobs)} fits for {a.task} ({len(have)} (setting, model, seed) triples already done)", flush=True)

    def job(s, m, seed, f, Xtr, ytr, g, Xte, ids, yte, num, cat):
        t0 = time.time()
        p = fit_one(m, num, cat, seed, Xtr, ytr, g, Xte)
        print(f"{s:15s} {m:18s} seed={seed} fold={f}: AUC={roc_auc_score(yte, p):.4f} ({time.time() - t0:.0f}s)", flush=True)
        return pd.DataFrame({"setting": s, "model": m, "seed": seed, "fold": str(f), "id": ids, "label": yte, "p": p})

    if a.jobs > 1:
        from joblib import Parallel, delayed
        frames = Parallel(n_jobs=a.jobs, backend="loky")(delayed(job)(*j) for j in jobs)
    else:
        frames = []
        for j in jobs:
            frames.append(job(*j))
            if len(frames) % 10 == 0:  # checkpoint
                pd.concat([done] + frames, ignore_index=True).to_parquet(path, index=False)
    if frames:
        pd.concat([done] + frames, ignore_index=True).to_parquet(path, index=False)


# ---------------------------------------------------------------- statistics
def delong(y, a, b):
    """Paired DeLong test for AUC(a) - AUC(b) (Sun & Xu 2014 fast structural components). -> (delta, se, p)."""
    def components(s):
        pos, neg = s[y == 1], s[y == 0]
        allv = np.concatenate([pos, neg])
        from scipy.stats import rankdata
        r_all, r_pos, r_neg = rankdata(allv), rankdata(pos), rankdata(neg)
        m, n = len(pos), len(neg)
        v10 = (r_all[:m] - r_pos) / n
        v01 = 1 - (r_all[m:] - r_neg) / m
        return v10, v01, (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
    a10, a01, auc_a = components(a)
    b10, b01, auc_b = components(b)
    m, n = len(a10), len(a01)
    s10, s01 = np.cov(np.vstack([a10, b10])), np.cov(np.vstack([a01, b01]))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (auc_a - auc_b) / np.sqrt(var) if var > 0 else 0.0
    return auc_a - auc_b, float(np.sqrt(max(var, 0))), float(2 * norm.sf(abs(z)))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_block_seed(y, P_m, P_r, groups, n, rng):
    """P_m, P_r: (n_seeds, n_rows) prediction matrices. Each resample: blocks with replacement + one seed per model."""
    ug = np.unique(groups)
    gi = {g: np.flatnonzero(groups == g) for g in ug}
    d = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        sm, sr = rng.integers(len(P_m)), rng.integers(len(P_r))
        d.append(roc_auc_score(yy, P_m[sm, idx]) - roc_auc_score(yy, P_r[sr, idx]))
    return np.array(d)


def summarize(a):
    rows, rng = [], np.random.default_rng(0)
    for task in HEADLINE:
        pr = load_preds(task)
        if not len(pr):
            continue
        pr = pr.drop_duplicates(subset=["setting", "model", "seed", "fold", "id"])
        sets = settings(task)
        for s, g in pr.groupby("setting"):
            d, *_, blocks, key = sets[s]
            # wide matrices: rows = test ids (in a fixed order), one column per (model, seed)
            mat = g.pivot_table(index="id", columns=["model", "seed"], values="p")
            lab = g.drop_duplicates("id").set_index("id").label.reindex(mat.index).to_numpy()
            grp = pd.Series(blocks, index=d[key].to_numpy()).reindex(mat.index).to_numpy()
            fold = g.drop_duplicates("id").set_index("id").fold.reindex(mat.index).to_numpy()
            models = list(dict.fromkeys(g.model))
            P = {m: mat[m].dropna(axis=1, how="all").to_numpy().T for m in models}  # (seeds, rows)
            P = {m: v for m, v in P.items() if not np.isnan(v).any()}
            for ref in [r for r in a.refs if r in P]:
                for m in P:
                    aucs = [roc_auc_score(lab, p) for p in P[m]]
                    fold_aucs = [np.mean([roc_auc_score(lab[fold == f], p[fold == f]) for f in np.unique(fold)]) for p in P[m]]
                    avg = P[m].mean(0)
                    r = {"task": task, "setting": s, "ref": ref, "model": m, "n_seeds": len(P[m]), "n_rows": len(lab),
                         "pooled_auc_mean": np.mean(aucs), "pooled_auc_sd": np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0,
                         "fold_auc_mean": np.mean(fold_aucs), "seed_avg_auc": roc_auc_score(lab, avg)}
                    if m != ref:
                        dd = boot_block_seed(lab, P[m], P[ref], grp, a.n_boot, rng)
                        delta, se, p = delong(lab, avg, P[ref].mean(0))
                        r.update(dAUC_mean=np.mean(aucs) - np.mean([roc_auc_score(lab, q) for q in P[ref]]),
                                 ci95_lo=np.percentile(dd, 2.5), ci95_hi=np.percentile(dd, 97.5), p_not_better=float((dd <= 0).mean()),
                                 delong_delta=delta, delong_se=se, delong_p=p)
                    rows.append(r)
                    print(task, s, ref, m, round(r["pooled_auc_mean"], 4), flush=True)
    if not rows:  # never overwrite the committed summary with an empty one
        sys.exit(f"no seed predictions ({OUT}/<task>/preds*.parquet) found; they are in the archived release. Summary left unchanged.")
    res = pd.DataFrame(rows)
    # Holm over the headline family: every (setting, model) vs a given reference, FMs only (the pre-registered claims)
    for ref in res.ref.unique() if "delong_p" in res else []:
        k = (res.ref == ref) & res.model.isin(FM_MODELS) & res.delong_p.notna()
        res.loc[k, "delong_p_holm"] = holm(res.loc[k, "delong_p"])
    res.to_csv(f"{OUT}/summary.csv", index=False)
    with open(f"{OUT}/summary.md", "w") as fh:
        fh.write("# Seed-aware headline statistics\n\n"
                 "Every model refit with seeds 0..S-1 on the committed folds. `pooled_auc_mean ± sd` over seeds; `seed_avg_auc` = AUC "
                 "of the seed-averaged prediction. ΔAUC CI: bootstrap over spatial blocks (Kentucky: rows) **and** an independent seed "
                 "draw for model and reference per resample. DeLong: paired test on the seed-averaged pooled OOF predictions; "
                 "`delong_p_holm` = Holm-corrected over the family {FM × headline setting} for the given reference.\n")
        for (task, s, ref), g in res.groupby(["task", "setting", "ref"], sort=False):
            cols = ["model", "n_seeds", "pooled_auc_mean", "pooled_auc_sd", "fold_auc_mean", "seed_avg_auc", "dAUC_mean", "ci95_lo", "ci95_hi",
                    "p_not_better", "delong_p", "delong_p_holm"]
            fh.write(f"\n## {task} / {s} (ref = {ref})\n\n" + g[[c for c in cols if c in g]].sort_values("pooled_auc_mean", ascending=False)
                     .round(4).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/summary.md").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--task", default="florida", choices=list(HEADLINE))
    ap.add_argument("--models", nargs="*", default=["xgboost", "lightgbm", "catboost", "tabpfn", "tabicl", "kumo_tabular"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1, help="> 1 only for CPU models (tuned baselines)")
    ap.add_argument("--refs", nargs="*", default=["xgboost", "xgboost_tuned"])
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
