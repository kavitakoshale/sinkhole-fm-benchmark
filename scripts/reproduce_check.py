"""CPU reproduction check: refit the untuned baselines on the committed Florida headline folds and compare every per-fold
metric with the committed results (results/florida_no_landcover/metrics_long.csv). No GPU, no model weights, no raw data.

Runs XGBoost (hand-set), random forest and logistic regression on no_landcover x ring x {spatial5, temporal} (~1-2 min on
4 cores), writes into a temporary results/florida_repro_check/ that is removed afterwards, and exits non-zero on a mismatch.
On the CPUs used for the paper the metrics are identical; another CPU type can change them in the fifth decimal (BLAS
kernels differ), hence a tolerance of 1e-3, below the precision the paper reports.

  python scripts/reproduce_check.py
  python scripts/reproduce_check.py --models xgboost --protocols temporal     # quicker
"""
import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = f"{ROOT}/results/florida_no_landcover/metrics_long.csv"
TAG = "repro_check"
METRICS = ["roc_auc", "pr_auc", "brier", "log_loss", "accuracy", "f1", "mcc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["xgboost", "random_forest", "logreg"])
    ap.add_argument("--protocols", nargs="*", default=["spatial5", "temporal"])
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    out = f"{ROOT}/results/florida_{TAG}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    try:
        subprocess.run([sys.executable, f"{ROOT}/baselines/run_florida_baselines.py", "--feature-set", "no_landcover", "--neg", "ring",
                        "--protocols", *a.protocols, "--models", *a.models, "--tag", TAG], check=True, env=env, stdout=subprocess.DEVNULL)
        new = pd.read_csv(f"{out}/metrics_long.csv")
    finally:
        shutil.rmtree(out, ignore_errors=True)
    ref = pd.read_csv(REF)
    key = ["neg_strategy", "protocol", "model", "fold"]
    m = new[new.model.isin(a.models)].merge(ref, on=key, suffixes=("", "_ref"))
    if len(m) != (new.model.isin(a.models)).sum():
        sys.exit(f"FAIL: {len(m)} of {len(new)} rows matched the committed results")
    diff = pd.concat({c: (m[c] - m[f"{c}_ref"]).abs() for c in METRICS}, axis=1).max()
    print(f"{len(m)} (protocol, model, fold) rows compared; max |difference| per metric:")
    print(diff.to_string())
    if (diff > a.tol).any():
        sys.exit("FAIL: reproduced metrics differ from the committed results")
    print("OK: baselines reproduce the committed results " + ("exactly" if (diff == 0).all() else f"within {a.tol:g}"))


if __name__ == "__main__":
    main()
