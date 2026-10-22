"""Paired comparison of models on identical out-of-fold predictions (e.g. foundation models vs XGBoost).

Merges the OOF/prediction parquet files of several result directories, then for every (setting, model)
reports pooled ROC-AUC / PR-AUC and a paired bootstrap of the difference vs a reference model.
Florida resamples 25 km spatial blocks (cluster bootstrap, respects spatial autocorrelation);
Kentucky resamples depressions.

Usage:
  python baselines/compare_models.py florida  results/florida results/florida_fm  [--ref xgboost]
  python baselines/compare_models.py florida  results/florida_no_landcover results/florida_fm_no_landcover
  python baselines/compare_models.py kentucky results/kentucky results/kentucky_fm
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEYS = {"florida": ["pid", "label"], "kentucky": ["dep_id", "label", "fold"]}
NON_MODEL = {"pid", "dep_id", "label", "fold", "fold_random", "fold_spatial", "split_temporal", "county", "block_id"}


def kind(task):
    """'florida' layout (oof_*.parquet, pid key, block bootstrap) for every region table except Kentucky."""
    return "kentucky" if task == "kentucky" else "florida"


def load(task, dirs):
    task = kind(task)
    pat = "oof_*.parquet" if task == "florida" else "pred_*.parquet"
    merged = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(ROOT, d, pat) if not os.path.isabs(d) else os.path.join(d, pat))):
            setting = re.sub(r"^(oof|pred)_|\.parquet$", "", os.path.basename(f))
            df = pd.read_parquet(f)
            if setting in merged:
                new = [c for c in df.columns if c not in merged[setting].columns or c in KEYS[task]]
                merged[setting] = merged[setting].merge(df[new], on=KEYS[task], how="inner")
            else:
                merged[setting] = df
    return merged


def boot(y, a, b, groups, n, rng):
    """Paired bootstrap of AUC(a) - AUC(b); resamples whole groups."""
    ug = np.unique(groups)
    gi = {g: np.flatnonzero(groups == g) for g in ug}
    d = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        d.append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))
    return np.array(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="florida | kentucky | any region with data/processed/<region>/dataset.parquet")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default="xgboost")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    merged = load(a.task, a.dirs)
    k = kind(a.task)
    if k == "florida":
        blocks = pd.read_parquet(f"{ROOT}/data/processed/{a.task}/dataset.parquet", columns=["pid", "block_id"])

    rows = []
    for setting, df in merged.items():
        if k == "florida":
            df = df.merge(blocks, on="pid", how="left")
            if "temporal" in setting:  # only the test rows carry predictions
                df = df[df["split_temporal"] == "test"]
            groups = df["block_id"].to_numpy()
        else:
            groups = np.arange(len(df))
        models = [c for c in df.columns if c not in NON_MODEL]
        df = df.dropna(subset=[m for m in models if m != "fgs_favorability_map"])
        y = df["label"].to_numpy()
        if a.ref not in df:
            print(f"skip {setting}: reference {a.ref} missing")
            continue
        for m in models:
            s = df[m].to_numpy(dtype=float)
            ok = ~np.isnan(s)
            r = {"setting": setting, "model": m, "n": int(ok.sum()),
                 "roc_auc": roc_auc_score(y[ok], s[ok]), "pr_auc": average_precision_score(y[ok], s[ok])}
            if m != a.ref and ok.all():
                d = boot(y, s, df[a.ref].to_numpy(dtype=float), groups[ok], a.n_boot, rng)
                r.update({f"dAUC_vs_{a.ref}": r["roc_auc"] - roc_auc_score(y, df[a.ref]),
                          "ci95_lo": np.percentile(d, 2.5), "ci95_hi": np.percentile(d, 97.5),
                          "p_not_better": float((d <= 0).mean())})
            rows.append(r)
    res = pd.DataFrame(rows).sort_values(["setting", "roc_auc"], ascending=[True, False])
    out = a.out or f"{ROOT}/results/comparison_{a.task}_{'_'.join(os.path.basename(d.rstrip('/')) for d in a.dirs)}.md"
    with open(out, "w") as fh:
        fh.write(f"# Paired comparison vs `{a.ref}` ({a.task})\n\nPooled out-of-fold ROC-AUC; ΔAUC 95% CI from "
                 f"{a.n_boot} {'spatial-block' if k == 'florida' else 'row'} bootstrap resamples.\n\n")
        for setting, g in res.groupby("setting", sort=False):
            fh.write(f"\n## {setting}\n\n" + g.drop(columns="setting").round(4).to_markdown(index=False) + "\n")
    res.to_csv(out.replace(".md", ".csv"), index=False)
    print(open(out).read())


if __name__ == "__main__":
    main()
