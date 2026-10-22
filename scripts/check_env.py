"""Pre-flight check for a machine that will run the foundation-model experiments.

Prints package versions, GPU info and the resolved FM settings (from env / .env), checks that the
benchmark tables exist, and runs every requested FM on a tiny Florida slice (300 train / 200 test) so
missing weights, licenses or CUDA problems surface in a minute instead of hours into a run.

Usage:  python scripts/check_env.py [--models tabpfn tabicl kumo_tabular] [--no-smoke]
Exit code is non-zero if any requested model fails.
"""
import argparse
import importlib.metadata as md
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "baselines"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["tabpfn", "tabicl", "kumo_tabular"])
    ap.add_argument("--no-smoke", action="store_true")
    a = ap.parse_args()

    from fm_models import FM_MODELS, TABPFN_HF_FILES, TABPFN_HF_REPO, TABPFN_HF_REVISION, device, env  # loads .env

    print("python", sys.version.split()[0])
    for p in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "torch", "tabpfn", "tabicl",
              "structured-data-models"]:
        try:
            print(f"  {p:24s} {md.version(p)}")
        except md.PackageNotFoundError:
            print(f"  {p:24s} MISSING")
    try:
        import torch
        print("cuda available:", torch.cuda.is_available(), "| devices:",
              [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except ImportError:
        print("torch not installed")
    print("resolved FM_DEVICE:", device())
    for k in ["FM_PRECISION", "FM_MAX_CONTEXT", "TABPFN_MODEL_PATH", "TABPFN_MODEL_VERSION", "TABPFN_HF_REPO",
              "TABPFN_HF_REVISION", "TABPFN_HF_FILE", "TABPFN_MODEL_CACHE_DIR",
              "TABICL_MODEL_PATH", "TABICL_CHECKPOINT_VERSION", "KUMO_SIZE", "KUMO_CHECKPOINT_DIR",
              "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE"]:
        v = env(k)
        flag = "" if not v or not k.endswith(("PATH", "DIR", "HOME", "CACHE")) or os.path.exists(v) else "  <-- PATH NOT FOUND"
        print(f"  {k:26s} {v}{flag}")
    print("  TABPFN_TOKEN set:", bool(env("TABPFN_TOKEN")))

    # where the weights come from (nothing is downloaded here; the smoke test below downloads what is missing)
    tp_file = env("TABPFN_HF_FILE") or TABPFN_HF_FILES.get(env("TABPFN_MODEL_VERSION", "v3.5"))
    if env("TABPFN_MODEL_PATH"):
        print("weights tabpfn   local", env("TABPFN_MODEL_PATH"))
    elif tp_file:
        repo, rev = env("TABPFN_HF_REPO", TABPFN_HF_REPO), env("TABPFN_HF_REVISION", TABPFN_HF_REVISION)
        try:
            from huggingface_hub import constants
            cached = os.path.exists(os.path.join(env("TABPFN_MODEL_CACHE_DIR") or os.path.join(constants.HF_HOME, "tabpfn", rev), tp_file))
        except ImportError:
            cached = False
        print(f"weights tabpfn   Hugging Face {repo}@{rev[:7]}/{tp_file} ({'cached' if cached else 'downloaded on first use'})")
    else:
        print("weights tabpfn   tabpfn package download for", env("TABPFN_MODEL_VERSION"), "(needs TABPFN_TOKEN)")
    print("weights tabicl  ", f"local {env('TABICL_MODEL_PATH')}" if env("TABICL_MODEL_PATH") else
          f"Hugging Face jingang/TabICL/{env('TABICL_CHECKPOINT_VERSION', 'tabicl-classifier-v2-20260212.ckpt')}")
    print("weights kumo    ", f"local {env('KUMO_CHECKPOINT_DIR')}" if env("KUMO_CHECKPOINT_DIR") else
          f"Hugging Face nvidia/Kumo-Tabular ({env('KUMO_SIZE', 'large')})")

    tables = {"florida": f"{ROOT}/data/processed/florida/dataset.parquet",
              "kentucky": f"{ROOT}/data/processed/kentucky/depressions_all.parquet"}
    for k, p in tables.items():
        print(f"table {k:9s} {'OK' if os.path.exists(p) else 'MISSING'}  {p}")
    if a.no_smoke:
        return 0

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from run_florida_baselines import feature_cols
    d = pd.read_parquet(tables["florida"])
    d = d[(d.label == 1) | (d.neg_strategy == "ring")]
    num, cat = feature_cols(d)
    tr = d[d.fold_spatial != 0].sample(300, random_state=0)
    te = d[d.fold_spatial == 0].sample(200, random_state=0)
    failed = []
    for m in a.models:
        t0 = time.time()
        try:
            p = FM_MODELS[m](num, cat, 0).fit(tr[num + cat], tr.label).predict_proba(te[num + cat])[:, 1]
            print(f"smoke {m:14s} OK   AUC={roc_auc_score(te.label, p):.3f}  {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            failed.append(m)
            print(f"smoke {m:14s} FAIL {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc(limit=2)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
