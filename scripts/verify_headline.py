"""Recompute the headline tables of the paper from the released out-of-fold predictions and compare them with the
committed results. CPU only, no model weights, and nothing committed is overwritten.

Checks
  comparisons  pooled ROC-AUC, PR-AUC and ΔAUC vs hand-set and tuned XGBoost (baselines/compare_models.py) must be
               identical to results/comparison_*.csv. The bootstrap intervals are printed next to the committed ones:
               compare_models.py draws all settings of a table from one random stream, and only the headline settings
               are released, so intervals of settings that follow a missing one differ by Monte-Carlo noise (up to 0.006).
  seeds        five-seed statistics (block x seed bootstrap, DeLong, Holm; scripts/seed_study.py) = results/seeds/summary.csv
  decision     calibration and prioritisation (scripts/decision_analysis.py) = results/decision/summary.csv

  python scripts/verify_headline.py                                      # everything, ~1 h on 4 cores (Kentucky is slow)
  python scripts/verify_headline.py --checks comparisons --tasks florida # a few minutes
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = f"{ROOT}/results"
COMPARISONS = {  # task: (run directories, {reference: committed comparison table})
    "florida": (["florida_no_landcover", "florida_tuned_no_landcover", "florida_fm_fp32_no_landcover", "florida_tabicl_fp32_no_landcover"],
                {"xgboost": "comparison_florida_no_landcover_tuned_fp32", "xgboost_tuned": "comparison_florida_no_landcover_tuned_fp32_ref_xgbtuned"}),
    "kentucky": (["kentucky", "kentucky_tuned", "kentucky_fm_fp32", "kentucky_tabicl_fp32"],
                 {"xgboost": "comparison_kentucky_tuned_fp32", "xgboost_tuned": "comparison_kentucky_tuned_fp32_ref_xgbtuned"}),
    "minnesota": (["minnesota", "minnesota_tuned", "minnesota_fm_fp32"],
                  {"xgboost": "comparison_minnesota_sat_nolc", "xgboost_tuned": "comparison_minnesota_sat_nolc_ref_xgbtuned"}),
}
HEADLINE = {"florida": ["ring_spatial5", "ring_temporal"], "kentucky": ["loco"], "minnesota": ["ring_spatial5"]}
SHOW = ["xgboost", "xgboost_tuned", "random_forest_tuned", "tabpfn", "tabicl", "kumo_tabular"]
TOL = 1e-9
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def has_predictions(dirs):
    return all(any(f.startswith(("oof_", "pred_")) and f.endswith(".parquet") for f in os.listdir(f"{R}/{d}")) for d in dirs)


def max_diff(new, old, keys, cols):
    m = new.merge(old, on=keys, suffixes=("", "_ref"))
    return len(m), {c: float((m[c] - m[f"{c}_ref"]).abs().max()) for c in cols if c in old}


def check_comparisons(tasks, n_boot):
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for task in tasks:
            dirs, refs = COMPARISONS[task]
            if not has_predictions(dirs):
                print(f"[comparisons] {task}: prediction files missing, skipped")
                continue
            for ref, stem in refs.items():
                out = f"{tmp}/{stem}.md"
                subprocess.run([sys.executable, "-W", "ignore", f"{ROOT}/baselines/compare_models.py", task, *[f"{R}/{d}" for d in dirs],
                                "--ref", ref, "--n-boot", str(n_boot), "--out", out], check=True, env=ENV, stdout=subprocess.DEVNULL)
                new, old = pd.read_csv(out.replace(".md", ".csv")), pd.read_csv(f"{R}/{stem}.csv")
                point = ["n", "roc_auc", "pr_auc", f"dAUC_vs_{ref}"]
                n, d = max_diff(new, old, ["setting", "model"], point + ["ci95_lo", "ci95_hi"])
                exact = n == len(new) and all(d[c] <= TOL for c in point)
                ok &= exact
                print(f"[comparisons] {stem}: {n} (setting, model) rows; point estimates {'identical' if exact else 'DIFFER'} "
                      f"(max |Δ| {max(d[c] for c in point):.1e}); CI bounds max |Δ| {max(d['ci95_lo'], d['ci95_hi']):.4f}")
                if ref == "xgboost_tuned":
                    m = new.merge(old, on=["setting", "model"], suffixes=("", "_ref"))
                    m = m[m.setting.isin(HEADLINE[task]) & m.model.isin(SHOW)]
                    for _, r in m.iterrows():
                        if r.model == ref:
                            print(f"    {r.setting:14s} {r.model:20s} AUC {r.roc_auc:.3f}  (reference)")
                            continue
                        print(f"    {r.setting:14s} {r.model:20s} AUC {r.roc_auc:.3f}  ΔAUC {r[f'dAUC_vs_{ref}']:+.3f} "
                              f"[{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}]  committed [{r.ci95_lo_ref:+.3f}, {r.ci95_hi_ref:+.3f}]")
    return ok


def rerun_and_compare(name, cmd, out_dir, keys):
    """Run a script that writes into out_dir, compare its summary.csv with the committed one, then restore out_dir."""
    backup = tempfile.mkdtemp()
    shutil.copytree(out_dir, f"{backup}/d")
    try:
        subprocess.run(cmd, check=True, env=ENV, stdout=subprocess.DEVNULL)
        new = pd.read_csv(f"{out_dir}/summary.csv")
    finally:
        shutil.rmtree(out_dir)
        shutil.copytree(f"{backup}/d", out_dir)
        shutil.rmtree(backup)
    old = pd.read_csv(f"{out_dir}/summary.csv")
    cols = [c for c in old.columns if c not in keys and pd.api.types.is_numeric_dtype(old[c])]
    n, d = max_diff(new, old, keys, cols)
    worst = max(d.values()) if d else float("nan")
    exact = n == len(old) == len(new) and worst <= TOL
    print(f"[{name}] {n} of {len(old)} rows matched; {'identical' if exact else 'DIFFER'} (max |Δ| over {len(cols)} columns {worst:.1e})")
    return exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", nargs="*", default=["comparisons", "seeds", "decision"])
    ap.add_argument("--tasks", nargs="*", default=list(COMPARISONS), help="for the comparisons check")
    ap.add_argument("--n-boot", type=int, default=1000, help="as committed; lower values change the intervals only")
    a = ap.parse_args()
    ok = True
    if "comparisons" in a.checks:
        ok &= check_comparisons(a.tasks, a.n_boot)
    if "seeds" in a.checks:
        ok &= rerun_and_compare("seeds", [sys.executable, "-W", "ignore", f"{ROOT}/scripts/seed_study.py", "summarize"],
                                f"{R}/seeds", ["task", "setting", "ref", "model"])
    if "decision" in a.checks:
        ok &= rerun_and_compare("decision", [sys.executable, "-W", "ignore", f"{ROOT}/scripts/decision_analysis.py"],
                                f"{R}/decision", ["region", "setting", "model"])
    if not ok:
        sys.exit("FAIL: recomputed headline results differ from the committed ones")
    print("OK: headline results recomputed from the released predictions")


if __name__ == "__main__":
    main()
