"""Positional-accuracy stratification of the Florida headline results.

The SIR reports carry a location-accuracy code (PARCEL / GPSSA = within the parcel or GPS; QTR2/QTR3 = quarter-section
centroids, i.e. hundreds of metres off). For every model the OOF AUC is recomputed with the positives restricted to one
accuracy class (all ring negatives kept), on no_landcover × ring × spatial5 and temporal.

  python scripts/positional_accuracy.py
Output: results/positional_accuracy.md
"""
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
from compare_models import NON_MODEL, load  # noqa: E402

DIRS = ["results/florida_no_landcover", "results/florida_tuned_no_landcover", "results/florida_fm_fp32_no_landcover",
        "results/florida_tabicl_fp32_no_landcover"]
SHOW = ["xgboost", "xgboost_tuned", "random_forest", "tabpfn", "tabicl", "kumo_tabular"]


def main():
    acc = pd.read_parquet(f"{ROOT}/data/processed/florida/dataset.parquet", columns=["pid", "accuracy", "event_year"])
    merged = load("florida", [d for d in DIRS if os.path.isdir(f"{ROOT}/{d}")])
    out = ["# Positional accuracy of the reports vs model AUC (Florida no_landcover × ring)\n",
           "Positives restricted to one SIR accuracy class, all ring negatives kept; pooled OOF AUC.\n"]
    for setting in ("ring_spatial5", "ring_temporal"):
        df = merged[setting].merge(acc, on="pid", how="left")
        if "temporal" in setting:
            df = df[df.split_temporal == "test"]
        models = [m for m in SHOW if m in df.columns]
        df = df.dropna(subset=models)
        neg = df[df.label == 0]
        rows = []
        classes = [("all", df.label == 1), ("PARCEL", df.accuracy == "PARCEL"), ("GPSSA", df.accuracy == "GPSSA"),
                   ("QTR2", df.accuracy == "QTR2"), ("QTR3", df.accuracy == "QTR3"),
                   ("PARCEL+GPSSA", df.accuracy.isin(["PARCEL", "GPSSA"])), ("QTR2+QTR3", df.accuracy.isin(["QTR2", "QTR3"]))]
        for name, mask in classes:
            pos = df[mask & (df.label == 1)]
            if len(pos) < 30:
                continue
            sub = pd.concat([pos, neg])
            rows.append({"positives": name, "n_pos": len(pos), **{m: roc_auc_score(sub.label, sub[m]) for m in models}})
        out.append(f"\n## {setting}\n\n" + pd.DataFrame(rows).round(4).to_markdown(index=False) + "\n")
    open(f"{ROOT}/results/positional_accuracy.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
