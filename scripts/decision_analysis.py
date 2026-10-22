"""Decision analysis from the committed out-of-fold predictions: what the probabilities are worth in the field.

For each headline setting and model:
  * reliability diagram (10 equal-mass bins) with ECE, raw and after cross-fitted Platt scaling,
  * field-prioritisation curve: share of sinkholes found vs share of sites inspected when sites are visited in
    descending score order, re-weighted to a 1:10 sinkhole : non-sinkhole prevalence (the benchmark tables are 1:1),
  * decision curve (net benefit vs threshold probability, Vickers & Elkin 2006) at the same 1:10 prevalence, with the
    probabilities prior-shifted from the 1:1 sample to that prevalence, against inspect-all and inspect-none,
  * a table: AUC, Brier, ECE, calibration slope, recall at 10 % / 20 % inspection budgets, sinkholes found per 100
    inspections at a 10 % budget, and the threshold range in which the model has higher net benefit than inspect-all.

Directories that do not exist yet (e.g. tuned baselines still running) are skipped, so the script can be re-run.

  python scripts/decision_analysis.py
Output: results/decision/<region>_<setting>.png, results/decision/summary.{md,csv}
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, f"{ROOT}/scripts")
from compare_models import NON_MODEL, load  # noqa: E402
from fm_scenarios import calib_metrics, ece, fold_col, platt_crossfit  # noqa: E402

OUT = f"{ROOT}/results/decision"
PREVALENCE = 1 / 11  # 1 sinkhole : 10 non-sinkhole sites
SHOW = ["xgboost", "xgboost_tuned", "random_forest", "logreg", "tabpfn", "tabicl", "kumo_tabular", "fgs_favorability_map"]
SOURCES = [  # (region, setting, dirs)
    ("florida", "ring_spatial5", ["florida_no_landcover", "florida_tuned_no_landcover", "florida_fm_fp32_no_landcover", "florida_tabicl_fp32_no_landcover"]),
    ("florida", "ring_temporal", ["florida_no_landcover", "florida_tuned_no_landcover", "florida_fm_fp32_no_landcover", "florida_tabicl_fp32_no_landcover"]),
    ("kentucky", "loco", ["kentucky", "kentucky_tuned", "kentucky_fm_fp32", "kentucky_tabicl_fp32"]),
    ("kentucky", "scott", ["kentucky", "kentucky_tuned", "kentucky_fm_fp32", "kentucky_tabicl_fp32"]),
    ("minnesota", "ring_spatial5", ["minnesota", "minnesota_tuned", "minnesota_fm_fp32"]),
]


def weights(y, prevalence):
    """Per-row weights that turn the sample prevalence into `prevalence` (positives keep weight 1)."""
    pi0 = y.mean()
    w_neg = (1 - prevalence) / prevalence * pi0 / (1 - pi0)
    return np.where(y == 1, 1.0, w_neg)


def gain_curve(y, p, w):
    """Share of sinkholes found against share of sites inspected, in descending score order. Sites with tied scores form
    one block along which the curve is linear, i.e. the expected value under a random order of the tied sites; otherwise
    the curve would depend on how the CPU's sort breaks ties (it did for the ordinal FGS map and random-forest votes)."""
    o = np.argsort(-p, kind="stable")
    y, w, p = y[o], w[o], p[o]
    end = np.r_[p[1:] != p[:-1], True]  # last site of each tied block
    inspected = np.r_[0, np.cumsum(w)[end] / w.sum()]
    found = np.r_[0, np.cumsum(w * y)[end] / (w * y).sum()]
    return inspected, found


def ece_ties(y, p, bins=10, draws=200, seed=0):
    """fm_scenarios.ece (10 equal-mass bins), averaged over seeded random orders of tied scores, so bin membership of tied
    sites does not depend on the sort implementation. Identical to fm_scenarios.ece when no scores are tied."""
    if len(np.unique(p)) == len(p):
        return ece(y, p, bins)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        o = np.lexsort((rng.random(len(p)), p))
        vals.append(sum(abs(p[g].mean() - y[g].mean()) * len(g) for g in np.array_split(o, bins)) / len(y))
    return float(np.mean(vals))


def recall_at(inspected, found, budget):
    return float(np.interp(budget, inspected, found))


def net_benefit(y, p, w, pts):
    n = w.sum()
    out = []
    for pt in pts:
        flag = p >= pt
        tp, fp = (w * (y == 1) * flag).sum(), (w * (y == 0) * flag).sum()
        out.append(tp / n - fp / n * pt / (1 - pt))
    return np.array(out)


def prior_shift(p, pi_sample, pi_target):
    """Probabilities fitted at prevalence pi_sample, re-expressed at prevalence pi_target (logit shift)."""
    z = np.log(p / (1 - p)) + np.log(pi_target / (1 - pi_target)) - np.log(pi_sample / (1 - pi_sample))
    return 1 / (1 + np.exp(-z))


def reliability(y, p, bins=10):
    o = np.argsort(p, kind="stable")
    xs, ys = [], []
    for g in np.array_split(o, bins):
        xs.append(p[g].mean())
        ys.append(y[g].mean())
    return np.array(xs), np.array(ys)


def analyse(region, setting, dirs):
    dirs = [f"results/{d}" for d in dirs if os.path.isdir(f"{ROOT}/results/{d}")]
    merged = load(region, dirs)
    if setting not in merged:
        return []
    df = merged[setting]
    if "temporal" in setting and "split_temporal" in df:
        df = df[df.split_temporal == "test"]
    df = df.reset_index(drop=True)
    models = [m for m in SHOW if m in df.columns]
    df = df.dropna(subset=[m for m in models if m != "fgs_favorability_map"])
    y = df.label.to_numpy()
    w = weights(y, PREVALENCE)
    fc = fold_col("kentucky" if region == "kentucky" else "florida", setting)
    folds = df[fc].to_numpy() if fc and fc in df else None
    pts = np.linspace(0.05, 0.8, 76)
    pi = PREVALENCE
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = dict(zip(models, plt.cm.tab10.colors))
    for m in models:
        p = df[m].to_numpy(dtype=float)
        ok = ~np.isnan(p)
        yy, pp, ww = y[ok], p[ok], w[ok]
        if m == "fgs_favorability_map":  # ordinal map classes, not probabilities: rescale to [0, 1] for the ranking curves only
            pp = (pp - pp.min()) / (pp.max() - pp.min() + 1e-9)
        pp = np.clip(pp, 1e-6, 1 - 1e-6)
        cm = calib_metrics(yy, pp)
        ins, fnd = gain_curve(yy, pp, ww)
        nb = net_benefit(yy, prior_shift(pp, yy.mean(), pi), ww, pts)
        nb_all = pi - (1 - pi) * pts / (1 - pts)
        better = pts[nb > nb_all + 1e-9]
        r = {"region": region, "setting": setting, "model": m, "n": int(ok.sum()), "roc_auc": cm["roc_auc"], "brier": cm["brier"],
             "ece": ece_ties(yy, pp), "slope": cm["slope"], "recall@10%": recall_at(ins, fnd, .10), "recall@20%": recall_at(ins, fnd, .20),
             "recall@30%": recall_at(ins, fnd, .30),
             "found_per_100_inspected@10%": recall_at(ins, fnd, .10) * pi * 100 / .10,
             "nb_beats_inspect_all_from_pt": better.min() if len(better) else np.nan, "nb_max": nb.max()}
        if folds is not None and m != "fgs_favorability_map":
            q = platt_crossfit(yy, pp, folds[ok])
            r.update(platt_brier=brier_score_loss(yy, q), platt_ece=ece_ties(yy, q))
        rows.append(r)
        kw = dict(color=colors[m], lw=2.2 if m in ("tabpfn", "tabicl", "kumo_tabular") else 1.3,
                  ls=":" if m == "fgs_favorability_map" else "-", label=m)
        if m != "fgs_favorability_map":
            xs, ys = reliability(yy, pp)
            axes[0].plot(xs, ys, marker="o", ms=3, **{**kw, "label": f"{m} (ECE {r['ece']:.3f})"})
        axes[1].plot(ins, fnd, **kw)
        axes[2].plot(pts, nb, **kw)
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8)
    axes[0].set(xlabel="mean predicted probability", ylabel="observed sinkhole rate", title="Reliability (1:1 sample)")
    axes[1].plot([0, 1], [0, 1], "k--", lw=.8)
    axes[1].set(xlim=(0, .5), xlabel=f"share of sites inspected (prevalence {pi:.3f})", ylabel="share of sinkholes found",
                title="Field prioritisation")
    axes[2].plot(pts, pi - (1 - pi) * pts / (1 - pts), "k--", lw=.8, label="inspect all")
    axes[2].axhline(0, color="k", lw=.8)
    axes[2].set(ylim=(-0.02, pi * 1.1), xlabel="threshold probability p_t", ylabel="net benefit", title="Decision curve")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{region} / {setting}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{region}_{setting}.png", dpi=130)
    plt.close(fig)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for region, setting, dirs in SOURCES:
        rows += analyse(region, setting, dirs)
        print(region, setting, flush=True)
    if not rows:  # never overwrite the committed summary with an empty one
        sys.exit("no out-of-fold predictions (results/*/oof_*.parquet) found; they are in the archived release. Summary left unchanged.")
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/summary.csv", index=False)
    with open(f"{OUT}/summary.md", "w") as fh:
        fh.write("# Decision analysis of OOF predictions\n\n"
                 f"Prioritisation and decision curves are re-weighted to a sinkhole prevalence of {PREVALENCE:.3f} (1:10); reliability, "
                 "Brier and ECE are on the 1:1 benchmark sample. `found_per_100_inspected@10%` = expected sinkholes among 100 inspected "
                 "sites when the top 10 % of sites are visited. `nb_beats_inspect_all_from_pt` = smallest threshold probability from "
                 "which the model's net benefit exceeds inspecting every site (NaN = never within 0.05–0.8; probabilities prior-shifted to the 1:10 "
                 "prevalence before thresholding). `platt_*` = after "
                 "cross-fitted Platt scaling. The FGS favourability map is an ordinal class map (no probabilities), shown for ranking only. "
                 "Tied scores (FGS classes, random-forest vote shares) enter the ranking curves and ECE as the expected value over random "
                 "orders of the tied sites, so the numbers do not depend on how the sort breaks ties.\n")
        for (region, setting), g in res.groupby(["region", "setting"], sort=False):
            fh.write(f"\n## {region} / {setting}  (`results/decision/{region}_{setting}.png`)\n\n" +
                     g.drop(columns=["region", "setting"]).round(4).to_markdown(index=False) + "\n")
    print(open(f"{OUT}/summary.md").read())


if __name__ == "__main__":
    main()
