"""Scenario experiments: where do tabular foundation models beat GBDTs? All runs reuse the committed folds.

  learning_curve  stratified training subsamples n = 50 ... full, evaluated on the full test fold
                  (Florida headline no_landcover x ring x spatial5, Kentucky loco)
  label_noise     corrupt a fraction of the TRAINING labels, test labels stay clean:
                  symmetric flips, or positives -> negatives (unreported sinkholes among the negatives)
  rotation        Kentucky loco: quantile-normalise the 10 features, then apply a random orthogonal rotation
                  (tests the axis-aligned-split bias of trees; Grinsztajn et al. 2022)
  calibration     Brier / log-loss / ECE / calibration slope, cross-fitted Platt scaling, Kentucky screening,
                  computed from the committed OOF predictions (baselines + results/*_fm_fp32*)
  blend           probability / rank averages of GBDT + FM OOF predictions, bootstrap ΔAUC via compare_models.py

Usage:  python scripts/fm_scenarios.py <experiment> [--tasks florida kentucky] [--models ...] [--reps N]
Output: results/fm_scenarios/<experiment>/ (runs.csv, summary.md, plots)
FM settings come from .env / environment as usual (the committed scenario runs use FM_PRECISION=fp32).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
from common import MODELS, PAPER_MODELS  # noqa: E402
from compare_models import NON_MODEL, load as load_oof  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402
from run_florida_baselines import DATA as FL_DATA, feature_cols, splits as fl_splits  # noqa: E402
from run_kentucky_baselines import FEATS, load as ky_load, splits as ky_splits  # noqa: E402
from tuned import TUNED_MODELS  # noqa: E402

ALL = {**PAPER_MODELS, **MODELS, **FM_MODELS, **TUNED_MODELS}
OUT = f"{ROOT}/results/fm_scenarios"
DEFAULT_MODELS = ["logreg", "random_forest", "xgboost", "lightgbm", "catboost", "tabpfn", "kumo_tabular"]
FULL = 1_000_000  # size key for "whole training fold"


def florida():
    d = pd.read_parquet(FL_DATA)
    num, cat = feature_cols(d, feature_set="no_landcover")
    d = d[(d.label == 1) | (d.neg_strategy == "ring")].reset_index(drop=True)
    return d, num, cat, [(f"f{f}", np.flatnonzero(tr), np.flatnonzero(te)) for f, tr, te in fl_splits(d, "spatial5")]


def kentucky():
    d = ky_load().reset_index(drop=True)
    return d, FEATS, [], [(f, np.asarray(tr), np.asarray(te)) for f, tr, te in ky_splits(d, "loco", 1)]


TASKS = {"florida": florida, "kentucky": kentucky}


def models_for(task, a):
    return a.models or (DEFAULT_MODELS + (["mlp10"] if task == "kentucky" else []))


def groups_of(task, d):
    """Spatial blocks for the inner CV of the tuned baselines (same definition as scripts/run_tuned_baselines.py)."""
    if task == "florida":
        return d.block_id.to_numpy()
    return (np.floor(d.lat / 0.05).astype(int) * 10000 + np.floor(d.lon / 0.05).astype(int)).to_numpy()


def fit_predict(name, num, cat, seed, Xtr, ytr, Xte, groups=None, oom_retries=30):
    """CUDA OOM (shared GPU) -> free memory, wait a minute and retry; any other failure -> NaN row, grid continues."""
    for attempt in range(oom_retries + 1):
        try:
            est = ALL[name](num, cat, seed)
            est = est.fit(Xtr, ytr, groups=groups) if getattr(est, "needs_groups", False) else est.fit(Xtr, ytr)
            return est.predict_proba(Xte)[:, 1]
        except Exception as e:  # noqa: BLE001  (one failed fit must not kill an hour-long grid)
            oom = "out of memory" in str(e).lower()
            print(f"FAIL {name} (attempt {attempt}, oom={oom}): {type(e).__name__}: {str(e)[:200]}", flush=True)
            if not oom or attempt == oom_retries:
                return None
            time.sleep(60)
        finally:
            if name in FM_MODELS:  # don't hoard cached GPU memory between fits
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()


def run_fit(rows, base, name, num, cat, seed, Xtr, ytr, Xte, yte, groups=None):
    t0 = time.time()
    p = fit_predict(name, num, cat, seed, Xtr, ytr, Xte, groups)
    ok = p is not None
    rows.append({**base, "model": name, "n_train": len(ytr), "pos_train": int(np.sum(ytr)),
                 "roc_auc": roc_auc_score(yte, p) if ok else np.nan,
                 "pr_auc": average_precision_score(yte, p) if ok else np.nan, "seconds": round(time.time() - t0, 2)})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)


def fmt_table(df):
    df = df.reset_index()
    if "n" in df:
        df["n"] = df["n"].astype(object).where(df["n"] != FULL, "full")
    return df.to_markdown(index=False)


def paired(r, keys, ref="xgboost"):
    """Mean AUC per key x model, and paired ΔAUC vs `ref` over (fold, rep): mean ± 1.96·SE (share of pairs won)."""
    piv = r.pivot_table(index=keys + ["fold", "rep"], columns="model", values="roc_auc")
    mean = piv.groupby(level=keys).mean()
    if ref not in piv:
        return mean, None
    g = piv.drop(columns=ref).sub(piv[ref], axis=0).groupby(level=keys)
    m, h, w = g.mean(), 1.96 * g.sem(), g.agg(lambda x: (x.dropna() > 0).mean())
    cell = pd.DataFrame({c: [f"{x:+.4f} ± {y:.4f} ({z:.0%})" for x, y, z in zip(m[c], h[c], w[c])] for c in m.columns},
                        index=m.index)
    return mean, cell


def runs_path(out, a):
    """One runs file per process (tasks / fold shards can run as parallel processes, even on several machines);
    summaries combine all runs*.csv."""
    return f"{out}/runs_{'_'.join(a.tasks + (a.folds or []))}{a.suffix}.csv"


def all_runs(out):
    import glob
    r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{out}/runs*.csv"))], ignore_index=True)
    keys = [c for c in r.columns if c not in ("n_train", "pos_train", "roc_auc", "pr_auc", "seconds")]
    return r.drop_duplicates(subset=keys, keep="first")  # overlapping shards


def selected(folds, a):
    return [f for f in folds if not a.folds or f[0] in a.folds]


def write_summary(r, keys, out, title, note="", extra=""):
    mean, delta = paired(r, keys)
    txt = f"# {title}\n\n{note}\n\n## Mean ROC-AUC\n\n{fmt_table(mean.round(4))}\n"
    if delta is not None:
        txt += f"\n## ΔAUC vs xgboost (paired over fold × rep: mean ± 1.96·SE, share of pairs won)\n\n{fmt_table(delta)}\n"
    if "xgboost_tuned" in r.model.unique():
        _, dt = paired(r, keys, ref="xgboost_tuned")
        txt += f"\n## ΔAUC vs xgboost_tuned (nested spatial CV, same pairing)\n\n{fmt_table(dt)}\n"
    with open(f"{out}/summary.md", "w") as fh:
        fh.write(txt + extra)
    return r, mean


# ---------------------------------------------------------------- 1. learning curve
def learning_curve(a):
    out = f"{OUT}/learning_curve"
    os.makedirs(out, exist_ok=True)
    sizes = {"florida": [50, 100, 250, 500, 1000, 2000, FULL], "kentucky": [50, 100, 250, 500, 1000, 2000, 5000, 10000, FULL]}
    rows = []
    for task in a.tasks:
        d, num, cat, folds = TASKS[task]()
        y, g = d.label.to_numpy(), groups_of(task, d)
        for fold, tr, te in selected(folds, a):
            for n in sizes[task]:
                reps = 1 if n == FULL else (a.reps if n <= 1000 else 3)
                for rep in range(reps):
                    idx = tr if n == FULL else train_test_split(tr, train_size=n, stratify=y[tr], random_state=rep)[0]
                    for m in models_for(task, a):
                        run_fit(rows, {"task": task, "fold": fold, "n": n, "rep": rep}, m, num, cat, rep,
                                d.loc[idx, num + cat], y[idx], d.loc[te, num + cat], y[te], g[idx])
            pd.DataFrame(rows).to_csv(runs_path(out, a), index=False)
    summarize_learning_curve(out)


def summarize_learning_curve(out):
    r = all_runs(out)
    write_summary(r, ["task", "n"], out, "Learning curve: AUC vs number of training labels",
                  "Florida = no_landcover × ring × spatial5 (5 folds); Kentucky = loco (3 counties). "
                  "Stratified subsamples of the training fold (5 reps for n ≤ 1000, 3 above, 1 for full); "
                  "test fold always complete. `full` = whole training fold.")
    plot_curve(r, f"{out}/learning_curve.png")


def plot_curve(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = r[r.model != "mlp10"]  # collapses at small n (early stopping on a handful of rows) and squashes the axis
    tasks = r.task.unique()
    names = sorted(r.model.unique())
    colors = dict(zip(names, (plt.cm.tab10 if len(names) <= 10 else plt.cm.tab20).colors))
    fig, axes = plt.subplots(2, len(tasks), figsize=(6.5 * len(tasks), 8), squeeze=False, height_ratios=[3, 2])
    for j, task in enumerate(tasks):
        rt = r[r.task == task]
        x = rt.groupby("n").n_train.mean()
        mean = rt.groupby(["n", "model"]).roc_auc.mean().unstack()
        piv = rt.pivot_table(index=["n", "fold", "rep"], columns="model", values="roc_auc")
        d = piv.drop(columns="xgboost").sub(piv.xgboost, axis=0).groupby(level="n")
        for m in mean.columns:
            fm = m in FM_MODELS
            kw = dict(color=colors[m], marker="o", lw=2.5 if fm else 1.2, ls="-" if fm else "--", label=m)
            axes[0, j].plot(x, mean[m], **kw)
            if m != "xgboost":
                axes[1, j].plot(x, d.mean()[m], **kw)
                if fm or m == "random_forest":
                    axes[1, j].fill_between(x, d.mean()[m] - 1.96 * d.sem()[m], d.mean()[m] + 1.96 * d.sem()[m],
                                            color=colors[m], alpha=.15)
        axes[1, j].axhline(0, color="k", lw=.8)
        axes[0, j].set(xscale="log", ylabel="ROC-AUC (test fold)", title=task)
        axes[1, j].set(xscale="log", xlabel="training labels", ylabel="ΔAUC vs xgboost (±1.96·SE)")
        for ax in axes[:, j]:
            ax.grid(alpha=.3)
        axes[0, j].legend(fontsize=8)
    fig.tight_layout(), fig.savefig(path, dpi=130)


# ---------------------------------------------------------------- 2. label noise
def corrupt(y, kind, rate, rng):
    y = y.copy()
    pool = np.arange(len(y)) if kind == "symmetric" else np.flatnonzero(y == 1)
    flip = rng.choice(pool, int(round(rate * len(pool))), replace=False)
    y[flip] = 1 - y[flip] if kind == "symmetric" else 0
    return y


def label_noise(a):
    out = f"{OUT}/label_noise"
    os.makedirs(out, exist_ok=True)
    kinds, rates = ["symmetric", "pos_to_neg"], [0.0, 0.05, 0.1, 0.2, 0.3]
    rows = []
    for task in a.tasks:
        d, num, cat, folds = TASKS[task]()
        y, g = d.label.to_numpy(), groups_of(task, d)
        for fold, tr, te in selected(folds, a):
            Xtr, Xte = d.loc[tr, num + cat], d.loc[te, num + cat]
            for m in models_for(task, a):  # clean reference, shared by both noise kinds
                run_fit(rows, {"task": task, "fold": fold, "kind": kinds[0], "rate": 0.0, "rep": 0}, m, num, cat, 0,
                        Xtr, y[tr], Xte, y[te], g[tr])
                rows.append({**rows[-1], "kind": kinds[1]})
            for kind in kinds:
                for rate in rates[1:]:
                    for rep in range(a.reps):
                        yn = corrupt(y[tr], kind, rate, np.random.default_rng([rep, int(rate * 100), len(kind)]))
                        for m in models_for(task, a):
                            run_fit(rows, {"task": task, "fold": fold, "kind": kind, "rate": rate, "rep": rep}, m, num,
                                    cat, rep, Xtr, yn, Xte, y[te], g[tr])
            pd.DataFrame(rows).to_csv(runs_path(out, a), index=False)
    summarize_label_noise(out)


def summarize_label_noise(out):
    r = all_runs(out)
    mean = r.groupby(["task", "kind", "rate", "model"]).roc_auc.mean().unstack("model")
    drop = mean - mean.xs(0.0, level="rate").reindex(mean.index.droplevel("rate")).to_numpy()
    write_summary(r, ["task", "kind", "rate"], out, "Label noise in the training labels (test labels clean)",
                  "symmetric = flip `rate` of all training labels; pos_to_neg = relabel `rate` of training positives as "
                  f"negatives (missed/unreported sinkholes). {r.rep.max() + 1} noise draws per rate; Florida headline "
                  "spatial5, Kentucky loco.",
                  f"\n## AUC lost relative to clean labels (mean AUC(rate) − AUC(0))\n\n{fmt_table(drop.round(4))}\n")


# ---------------------------------------------------------------- 3. rotation (axis-aligned bias)
def rotation(a):
    from scipy.stats import special_ortho_group
    from sklearn.preprocessing import QuantileTransformer
    out = f"{OUT}/rotation"
    os.makedirs(out, exist_ok=True)
    d, num, _, folds = kentucky()
    y = d.label.to_numpy()
    rows = []
    for fold, tr, te in folds:
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit(d.loc[tr, num])
        Q = {"tr": qt.transform(d.loc[tr, num]), "te": qt.transform(d.loc[te, num])}
        conds = [("raw", 0, None), ("quantile", 0, None)] + \
                [("rotated", s, special_ortho_group.rvs(len(num), random_state=s)) for s in range(a.reps)]
        for cond, rep, R in conds:
            if cond == "raw":
                Xtr, Xte = d.loc[tr, num], d.loc[te, num]
            else:
                Xtr, Xte = (pd.DataFrame(Q[k] if R is None else Q[k] @ R, columns=num) for k in ("tr", "te"))
            for m in models_for("kentucky", a):
                run_fit(rows, {"task": "kentucky", "fold": fold, "condition": cond, "rep": rep}, m, num, [], rep,
                        Xtr, y[tr], Xte, y[te])
        pd.DataFrame(rows).to_csv(f"{out}/runs_kentucky.csv", index=False)
    summarize_rotation(out)


def summarize_rotation(out):
    r = all_runs(out)
    mean = r.groupby(["condition", "model"]).roc_auc.mean().unstack("model")
    tax = (mean.loc["quantile"] - mean.loc["rotated"]).sort_values().round(4).rename("auc_lost_to_rotation")
    write_summary(r, ["task", "condition"], out, "Rotation test (Kentucky loco)",
                  "raw = original features; quantile = per-feature quantile→normal (monotone, trees are invariant); "
                  f"rotated = quantile + random orthogonal rotation of all 10 features ({r[r.condition == 'rotated'].rep.nunique()} "
                  "rotations). A large AUC loss under rotation means the model relies on axis-aligned structure.",
                  f"\n## AUC lost to rotation (quantile − rotated)\n\n{tax.to_frame().to_markdown()}\n")


# ---------------------------------------------------------------- 4/5. calibration + blends from committed OOFs
BLENDS = {"avg_gbdt3": ["xgboost", "lightgbm", "catboost"], "avg_xgb_tabpfn": ["xgboost", "tabpfn"],
          "avg_xgb_kumo": ["xgboost", "kumo_tabular"], "avg_tabpfn_kumo": ["tabpfn", "kumo_tabular"],
          "avg_xgb_tabpfn_kumo": ["xgboost", "tabpfn", "kumo_tabular"],
          "avg_gbdt3_fm2": ["xgboost", "lightgbm", "catboost", "tabpfn", "kumo_tabular"]}
RANK_BLENDS = {"rank_xgb_tabpfn_kumo": ["xgboost", "tabpfn", "kumo_tabular"]}
SOURCES = [("florida", "florida_no_landcover", "florida_fm_fp32_no_landcover"),
           ("florida", "florida", "florida_fm_fp32"),
           ("kentucky", "kentucky", "kentucky_fm_fp32")]


def fold_col(task, setting):
    if task == "kentucky":
        return None if setting == "scott" else "fold"
    return {"random5": "fold_random", "spatial5": "fold_spatial"}.get(setting.split("_", 1)[1])


def merged_with_blends(task, base, fm):
    out = {}
    for setting, df in load_oof(task, [f"results/{base}", f"results/{fm}"]).items():
        if task == "florida" and "temporal" in setting:
            df = df[df.split_temporal == "test"]
        df = df.reset_index(drop=True)
        fc = fold_col(task, setting)
        for name, ms in BLENDS.items():
            df[name] = df[ms].mean(axis=1)
        for name, ms in RANK_BLENDS.items():  # ranks within each fold so fold-specific score scales don't matter
            df[name] = sum(df.groupby(fc)[m].rank(pct=True) if fc else df[m].rank(pct=True) for m in ms) / len(ms)
        out[setting] = df
    return out


def blend(a):
    out = f"{OUT}/blend"
    for task, base, fm in SOURCES:
        d = f"{out}/{base}"
        os.makedirs(d, exist_ok=True)
        for setting, df in merged_with_blends(task, base, fm).items():
            df.to_parquet(f"{d}/{'oof' if task == 'florida' else 'pred'}_{setting}.parquet", index=False)
        subprocess.run([sys.executable, f"{ROOT}/baselines/compare_models.py", task, d,
                        "--out", f"{out}/comparison_{base}.md"], check=True, stdout=subprocess.DEVNULL)
        print(f"blend {base}: {out}/comparison_{base}.md", flush=True)


def ece(y, p, bins=10):
    order = np.argsort(p)
    return sum(abs(p[g].mean() - y[g].mean()) * len(g) for g in np.array_split(order, bins)) / len(y)


def calib_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p))[:, None], y)
    return {"roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
            "ece": ece(y, p), "slope": lr.coef_[0, 0], "mean_p_minus_rate": p.mean() - y.mean()}


def platt_crossfit(y, p, folds):
    """Recalibrate fold k with a Platt map fitted on the other folds' OOF predictions (no fold sees its own labels)."""
    z, q = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None], np.empty_like(p)
    for f in np.unique(folds):
        k = folds == f
        q[k] = LogisticRegression(C=1e6, max_iter=1000).fit(z[~k], y[~k]).predict_proba(z[k])[:, 1]
    return q


def screening(y, p, thr=0.1):
    order = np.argsort(-p)
    cum = np.cumsum(y[order]) / y.sum()
    flagged = lambda r: (np.searchsorted(cum, r) + 1) / len(y)  # noqa: E731
    return {"flag_frac@p>0.1": (p >= thr).mean(), "recall@p>0.1": y[p >= thr].sum() / y.sum(),
            "flag_frac@recall95": flagged(0.95), "flag_frac@recall97": flagged(0.97)}


def calibration(a):
    out = f"{OUT}/calibration"
    os.makedirs(out, exist_ok=True)
    rows = []
    for task, base, fm in SOURCES:
        for setting, df in merged_with_blends(task, base, fm).items():
            fc = fold_col(task, setting)
            models = [c for c in df.columns if c not in NON_MODEL and c != "fgs_favorability_map" and not c.startswith("rank_")]
            df = df.dropna(subset=models)
            y = df.label.to_numpy()
            for m in models:
                p = df[m].to_numpy(dtype=float)
                r = {"features": base, "setting": setting, "model": m, **calib_metrics(y, p)}
                if fc:
                    folds = df[fc].to_numpy()
                    r["mean_fold_auc"] = np.mean([roc_auc_score(y[folds == f], p[folds == f]) for f in np.unique(folds)])
                    q = platt_crossfit(y, p, folds)
                    r.update({f"platt_{k}": v for k, v in calib_metrics(y, q).items() if k in ("roc_auc", "brier", "log_loss", "ece")})
                if task == "kentucky":
                    r.update(screening(y, p))
                rows.append(r)
    r = pd.DataFrame(rows)
    r.to_csv(f"{out}/calibration.csv", index=False)
    with open(f"{out}/summary.md", "w") as fh:
        fh.write("# Calibration of OOF predictions (FMs in fp32)\n\nece = 10 equal-mass bins; slope = logistic "
                 "calibration slope (1 = ideal, < 1 = over-confident); mean_p_minus_rate > 0 = over-prediction; "
                 "platt_* = after cross-fitted Platt scaling (fitted on the other folds' OOF predictions; "
                 "only for fold-based protocols). Kentucky screening: share of depressions to inspect.\n")
        for (fs, setting), g in r.groupby(["features", "setting"], sort=False):
            fh.write(f"\n## {fs} / {setting}\n\n" + g.drop(columns=["features", "setting"]).dropna(axis=1, how="all")
                     .sort_values("log_loss").round(4).to_markdown(index=False) + "\n")
    print(f"calibration -> {out}/summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["learning_curve", "label_noise", "rotation", "calibration", "blend", "summarize"],
                    help="summarize = rebuild learning_curve / label_noise / rotation summaries from their runs*.csv")
    ap.add_argument("--tasks", nargs="*", default=["florida", "kentucky"])
    ap.add_argument("--models", nargs="*", default=None, help="default: " + " ".join(DEFAULT_MODELS) + " (+ mlp10 on Kentucky)")
    ap.add_argument("--reps", type=int, default=None, help="subsample / noise / rotation repeats")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="learning_curve / label_noise: only these folds (f0..f4, test_<County>), for sharding")
    ap.add_argument("--suffix", default="", help="appended to the runs file name (e.g. _tabicl) so model shards don't collide")
    a = ap.parse_args()
    a.reps = a.reps or {"learning_curve": 5, "label_noise": 2, "rotation": 3}.get(a.experiment, 1)
    os.makedirs(OUT, exist_ok=True)
    if a.experiment == "summarize":
        for e in ("learning_curve", "label_noise", "rotation"):
            if os.path.isdir(f"{OUT}/{e}"):
                globals()[f"summarize_{e}"](f"{OUT}/{e}")
                print(f"{OUT}/{e}/summary.md")
        return
    globals()[a.experiment](a)


if __name__ == "__main__":
    main()
