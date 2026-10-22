"""Context selection for in-context models: which rows should be in the context when predicting one county?

Florida headline table (no_landcover × ring). For every county with >= MIN_COUNTY positives the county is the query set
(all its positives + ring negatives) and the context is built from the rest of the state in several ways:

  state        every row outside the county (= leave-one-county-out, the largest context)
  neighbours   rows within NEIGHBOUR_KM of the county's points (local context)
  far          rows farther than FAR_KM from every county point (the anti-local control)
  knn_geo_k    the k rows closest (in map distance) to the county's points
  knn_feat_k   the k rows closest in standardised feature space to the county's points (cosine of nearest query point)
  random_k     k random rows (3 draws)

So the question of Saito et al. (GEOAI 2025) -- does a smaller, local context beat the whole database? -- gets tested for
susceptibility classification, for the FMs and for XGBoost refit on the same context.

  FM_PRECISION=fp32 python scripts/context_selection.py --models xgboost tabpfn tabicl kumo_tabular
Output: results/context_selection/{runs.csv, summary.md}
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, f"{ROOT}/scripts")
from fm_scenarios import fit_predict  # noqa: E402
from realistic_benchmark import load  # noqa: E402

OUT = f"{ROOT}/results/context_selection"
MIN_COUNTY, NEIGHBOUR_KM, FAR_KM = 100, 50, 150


def contexts(d, num, inc, rest, ks, rng):
    xy = d[["x", "y"]].to_numpy()
    dist_geo = cKDTree(xy[inc]).query(xy[rest])[0]  # distance of every candidate row to the nearest query point
    Z = StandardScaler().fit_transform(d[num].fillna(d[num].median()).to_numpy())
    dist_feat = cKDTree(Z[inc]).query(Z[rest])[0]
    out = {"state": rest, "neighbours": rest[dist_geo <= NEIGHBOUR_KM * 1000], "far": rest[dist_geo > FAR_KM * 1000]}
    for k in ks:
        out[f"knn_geo_{k}"] = rest[np.argsort(dist_geo)[:k]]
        out[f"knn_feat_{k}"] = rest[np.argsort(dist_feat)[:k]]
        for r in range(3):
            out[f"random_{k}#r{r}"] = rng.choice(rest, k, replace=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["xgboost", "tabpfn", "tabicl", "kumo_tabular"])
    ap.add_argument("--k", nargs="*", type=int, default=[250, 1000])
    ap.add_argument("--feature-set", default="no_landcover")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d, sets = load("florida")
    num, cat = sets[a.feature_set]
    cols = num + cat
    counts = d[d.label == 1].county.value_counts()
    rows, rng = [], np.random.default_rng(0)
    for c in counts[counts >= MIN_COUNTY].index:
        inc = d.index[d.county == c].to_numpy()
        rest = d.index[d.county != c].to_numpy()
        y = d.loc[inc, "label"].to_numpy()
        for name, ctx in contexts(d, num, inc, rest, a.k, rng).items():
            if len(ctx) < 20 or d.loc[ctx, "label"].nunique() < 2:
                continue
            for m in a.models:
                t0 = time.time()
                p = fit_predict(m, num, cat, 0, d.loc[ctx, cols], d.loc[ctx, "label"].to_numpy(), d.loc[inc, cols])
                base, _, rep = name.partition("#")
                rows.append({"county": c, "context": base, "rep": rep or "r0", "n_context": len(ctx), "pos_context": int(d.loc[ctx, "label"].sum()),
                             "median_dist_km": float(np.median(cKDTree(d.loc[inc, ["x", "y"]].to_numpy()).query(d.loc[ctx, ["x", "y"]].to_numpy())[0]) / 1000),
                             "n_query": len(inc), "model": m, "roc_auc": roc_auc_score(y, p) if p is not None else np.nan,
                             "pr_auc": average_precision_score(y, p) if p is not None else np.nan, "seconds": round(time.time() - t0, 1)})
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/runs.csv", index=False)
    summarize()


def summarize():
    r = pd.read_csv(f"{OUT}/runs.csv")
    mean = r.pivot_table(index="context", columns="model", values="roc_auc", aggfunc="mean")
    size = r.groupby("context")[["n_context", "median_dist_km"]].median().round(1)
    per_county = r.pivot_table(index=["county", "context"], columns="model", values="roc_auc", aggfunc="mean")
    piv = r.pivot_table(index=["county", "rep"], columns=["model", "context"], values="roc_auc")
    delta = {}
    for m in r.model.unique():
        base = piv[(m, "state")]
        for ctx in r.context.unique():
            if ctx == "state":
                continue
            x = (piv[(m, ctx)] - base).dropna()
            delta[(ctx, m)] = f"{x.mean():+.4f} ± {1.96 * x.sem():.4f} ({(x > 0).mean():.0%})"
    delta = pd.Series(delta).unstack()
    with open(f"{OUT}/summary.md", "w") as fh:
        fh.write(f"# Context selection (Florida no_landcover × ring; {r.county.nunique()} counties with ≥ {MIN_COUNTY} reports)\n\n"
                 "Query = all rows of one county; context = rows from the rest of the state chosen as described in the header of "
                 "`scripts/context_selection.py`. XGBoost is refit on each context; the FMs just receive it.\n\n"
                 "## Mean AUC over counties (random contexts: 3 draws)\n\n" + pd.concat([size, mean.round(4)], axis=1).to_markdown() +
                 "\n\n## ΔAUC vs the whole-state context (paired over counties: mean ± 1.96·SE, share of counties where it helped)\n\n" +
                 delta.to_markdown() + "\n\n## Per county\n\n" + per_county.round(4).to_markdown() + "\n")
    print(open(f"{OUT}/summary.md").read())


if __name__ == "__main__":
    main()
