"""Factor analysis for the foundation models and XGBoost: permutation importance and partial dependence.

Florida headline setting (no_landcover × ring × spatial5). For every outer fold the model is fit on the training fold and
  * permutation importance = AUC drop on the test fold when one feature (or one feature *group*: terrain, hydro, geology,
    depressions, proximity, soil, climate) is shuffled within the test fold (`--reps` shuffles, mean over folds),
  * partial dependence of the top-`--top` numeric features (by mean importance over models): mean prediction on a sample
    of the test fold when the feature is set to each of 11 quantile values (ICE average).

  FM_PRECISION=fp32 python scripts/interpret.py --models xgboost xgboost_tuned tabpfn tabicl kumo_tabular
Output: results/interpret/{importance.csv, group_importance.csv, pdp.csv, summary.md, importance.png, pdp.png}
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, f"{ROOT}/scripts")
from fm_models import FM_MODELS  # noqa: E402
from fm_scenarios import ALL, florida  # noqa: E402

OUT = f"{ROOT}/results/interpret"
GROUPS = {"terrain": lambda c: c.split("_")[0] in ("elev", "slope", "laplacian", "tpi", "relief", "depth", "height") or c.startswith("slope_mean"),
          "hydro": lambda c: c in ("sas_depth_to_water_ft", "ias_thickness_ft", "head_diff_ft", "floridan_recharge_class", "drastic_floridan_index", "drastic_setting"),
          "geology": lambda c: c.startswith(("geo_", "envgeo_", "sinkhole_type", "geomorph")),
          "depressions": lambda c: c.startswith("ctd_"), "proximity": lambda c: c.startswith("dist_"),
          "soil": lambda c: c.startswith("soil_"), "climate": lambda c: c == "ppt_annual_mm"}


def fit(name, num, cat, Xtr, ytr, groups):
    est = ALL[name](num, cat, 0)
    return est.fit(Xtr, ytr, groups=groups) if getattr(est, "needs_groups", False) else est.fit(Xtr, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["xgboost", "xgboost_tuned", "tabpfn", "tabicl", "kumo_tabular"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--pdp-rows", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d, num, cat, folds = florida()
    cols = num + cat
    y = d.label.to_numpy()
    grp = {g: [c for c in cols if f(c)] for g, f in GROUPS.items()}
    assert sorted(sum(grp.values(), [])) == sorted(cols), set(cols) - set(sum(grp.values(), []))
    imp, gimp, models = [], [], {}
    for fold, tr, te in folds:
        Xtr, Xte, yte = d.loc[tr, cols], d.loc[te, cols], y[te]
        for m in a.models:
            t0 = time.time()
            est = fit(m, num, cat, Xtr, y[tr], d.loc[tr, "block_id"].to_numpy())
            models[(fold, m)] = est
            base = roc_auc_score(yte, est.predict_proba(Xte)[:, 1])
            for rep in range(a.reps):
                rng = np.random.default_rng(rep)
                for c in cols:
                    Xp = Xte.copy()
                    Xp[c] = rng.permutation(Xp[c].to_numpy())
                    imp.append({"fold": fold, "model": m, "rep": rep, "feature": c, "auc_drop": base - roc_auc_score(yte, est.predict_proba(Xp)[:, 1])})
                for g, gc in grp.items():
                    Xp = Xte.copy()
                    perm = rng.permutation(len(Xp))
                    Xp[gc] = Xp[gc].to_numpy()[perm]
                    gimp.append({"fold": fold, "model": m, "rep": rep, "group": g, "n_features": len(gc),
                                 "auc_drop": base - roc_auc_score(yte, est.predict_proba(Xp)[:, 1])})
            print(f"{fold} {m}: base AUC {base:.4f}, importance done in {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(imp).to_csv(f"{OUT}/importance.csv", index=False)
        pd.DataFrame(gimp).to_csv(f"{OUT}/group_importance.csv", index=False)
    imp, gimp = pd.DataFrame(imp), pd.DataFrame(gimp)
    mean_imp = imp.groupby(["feature", "model"]).auc_drop.mean().unstack()
    top = [c for c in mean_imp.mean(axis=1).sort_values(ascending=False).index if c in num][:a.top]

    # partial dependence (ICE average) on a sample of each test fold
    pdp = []
    for fold, tr, te in folds:
        rng = np.random.default_rng(0)
        sub = d.loc[rng.choice(te, min(a.pdp_rows, len(te)), replace=False), cols]
        for c in top:
            qs = np.unique(np.nanquantile(d[c], np.linspace(0.02, 0.98, 11)))
            for m in a.models:
                est = models[(fold, m)]
                for q in qs:
                    Xq = sub.copy()
                    Xq[c] = q
                    pdp.append({"fold": fold, "model": m, "feature": c, "value": q, "mean_pred": float(est.predict_proba(Xq)[:, 1].mean())})
        print(f"{fold}: pdp done", flush=True)
    pdp = pd.DataFrame(pdp)
    pdp.to_csv(f"{OUT}/pdp.csv", index=False)

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), width_ratios=[3, 1.4])
    order = mean_imp.mean(axis=1).sort_values().index[-25:]
    mean_imp.loc[order].plot.barh(ax=axes[0], width=.8)
    axes[0].set(title="Permutation importance (AUC drop, mean over 5 spatial folds), top 25", xlabel="AUC drop")
    g = gimp.groupby(["group", "model"]).auc_drop.mean().unstack()
    g.loc[g.mean(axis=1).sort_values().index].plot.barh(ax=axes[1], width=.8, legend=False)
    axes[1].set(title="Feature-group importance", xlabel="AUC drop")
    for ax in axes:
        ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(f"{OUT}/importance.png", dpi=120)
    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 3.8), squeeze=False)
    for ax, c in zip(axes[0], top):
        for m, gm in pdp[pdp.feature == c].groupby("model"):
            cur = gm.groupby("value").mean_pred.mean()
            ax.plot(cur.index, cur.values, marker="o", ms=3, lw=2 if m in FM_MODELS else 1.2, ls="-" if m in FM_MODELS else "--", label=m)
        ax.set(title=c, xlabel=c, ylabel="mean predicted probability")
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pdp.png", dpi=120)
    with open(f"{OUT}/summary.md", "w") as fh:
        fh.write("# Factor analysis (Florida no_landcover × ring × spatial5)\n\nPermutation importance = AUC drop on the test fold when the "
                 f"feature is shuffled ({a.reps} shuffles × 5 folds, mean). Group importance shuffles all features of a source together "
                 "(same row permutation). Partial dependence = mean prediction on a test-fold sample with the feature fixed at 11 quantiles.\n\n"
                 "## Feature-group importance\n\n" + pd.concat([gimp.groupby("group").n_features.first(), g.round(4)], axis=1)
                 .sort_values(g.columns[0], ascending=False).to_markdown() +
                 "\n\n## Top 25 features\n\n" + mean_imp.loc[order[::-1]].round(4).to_markdown() +
                 "\n\n## Rank correlation of feature importance between models (Spearman)\n\n" + mean_imp.corr("spearman").round(3).to_markdown() +
                 "\n\nFigures: `results/interpret/importance.png`, `results/interpret/pdp.png` (top features: " + ", ".join(top) + ").\n")
    print(open(f"{OUT}/summary.md").read())


if __name__ == "__main__":
    main()
