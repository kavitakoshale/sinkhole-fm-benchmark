"""Spatially separated small-data evaluation (Supplementary Section S20 of the paper).

The committed small-data results are interpolation results: the county few-shot test rows share the county with the training
rows (`realistic_benchmark.py county`) and the learning curve subsamples the training side of a 25 km block CV that enforces no
minimum distance across block boundaries (`fm_scenarios.py learning_curve`). This script answers with

  county          Experiment A. Same design as realistic_benchmark.county (counties with >= 100 positives, k in-county positives +
                  k in-county ring negatives, k in {25, 50, 100}, 3 seeds) with (1) explicit pair handling — `paired` draws k positives
                  and takes their paired (nearest) ring negatives, `independent` draws positives and negatives independently (the
                  current design) — and (2) a buffered test set: the SAME fitted model is scored once, then evaluated on the in-county
                  rows farther than B km from every training row, B in {0, 1, 2.5, 5, 10} (B = 0 reproduces the current result), plus
                  the `neighbours` test (other counties within 50 km). Cells with < 20 positives or < 20 negatives left are skipped
                  and recorded as such. No refit per B.
  curve           Experiment B. For each committed outer fold f of `fold_spatial` (5 folds, both states): test = fold f complete;
                  candidate training rows = rows outside fold f AND farther than B km from every test row, B in {0, 2.5, 5, 10};
                  matched training sets n in {50, 100, 250, 500, 1000, 2000} drawn stratified by label from the candidates
                  (train_test_split(..., stratify=y, random_state=rep); 5 reps for n <= 1000, 3 for 2000) plus `full` = all
                  candidates. Every model gets identical rows. Sampling is `independent` only (pairs are handled in Experiment A).
  kentucky_cells  Experiment C. Paired ΔAUC of tabpfn / tabicl / kumo_tabular vs xgboost_tuned and xgboost on the committed Kentucky
                  loco OOF predictions with a cluster bootstrap over 0.05° cells (seed_study.ky_groups), per held-out county and
                  pooled, next to the existing row bootstrap; plus the (cell × seed) version from results/seeds/kentucky.
  summarize       summary.md (sections 1–4), buffered_curve.png, buffered_county.png, run_info.json.

Data: Florida committed table, label == 1 or ring negatives, feature set no_landcover (49 factors); Minnesota committed table, same
filter, shared schema sat_nolc (34). Pairs: the pair of a positive is its nearest ring negative (cKDTree) in both states and is written
to results/buffered/pairs_<region>.csv. Nothing under results/ outside results/buffered/ is touched.

  python scripts/buffered_smalldata.py kentucky_cells
  python scripts/buffered_smalldata.py county --regions florida minnesota [--models ...] [--suffix _tuned]
  python scripts/buffered_smalldata.py curve --regions florida --folds 0 1 [--buffers 0 5 10] [--suffix _f01]
  python scripts/buffered_smalldata.py summarize
Shards write results/buffered/<exp>/runs_<region>_<fs><suffix>.csv and are resumable (finished cells are skipped on restart);
summarize merges every runs*.csv. FM settings come from .env (run with FM_PRECISION=fp32 as the committed results do).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/baselines")
sys.path.insert(0, os.path.dirname(__file__))
from compare_models import boot  # noqa: E402
from fm_models import FM_MODELS  # noqa: E402
from fm_scenarios import ece, fit_predict  # noqa: E402,F401  (ece re-exported for the calibration columns of future shards)
from realistic_benchmark import MIN_COUNTY, NEIGHBOUR_KM, load  # noqa: E402
from seed_study import boot_block_seed, delong, ky_groups  # noqa: E402

OUT = f"{ROOT}/results/buffered"
FS = {"florida": "no_landcover", "minnesota": "sat_nolc"}
DEFAULT_MODELS = ["xgboost", "xgboost_tuned", "random_forest", "tabpfn", "tabicl", "kumo_tabular"]
BUFFERS_A = [0, 1, 2.5, 5, 10]
BUFFERS_B = [0, 2.5, 5, 10]
SIZES = [50, 100, 250, 500, 1000, 2000]
FULL = 1_000_000  # size key for "all candidates" (same convention as fm_scenarios)
MIN_CLASS = 20  # a test cell needs >= 20 positives and >= 20 negatives
REF = "xgboost_tuned"
RUN_KEYS = {"county": ["region", "feature_set", "county", "k", "rep", "mode", "test", "buffer_km", "model"],
            "curve": ["region", "feature_set", "fold", "buffer_km", "n", "rep", "model"]}
METRIC_COLS = ["n_train", "pos_train", "n_test", "pos_test", "roc_auc", "pr_auc", "seconds", "skipped", "n_candidates",
               "pos_candidates", "n_pair_fallback", "n_test_paired_with_train"]


# ---------------------------------------------------------------- data
def region_data(region):
    """Committed table (label == 1 or ring negatives), headline feature set, projected coordinates, nearest-negative pairs."""
    d, sets = load(region)
    num, cat = sets[FS[region]]
    y = d.label.to_numpy()
    xy = d[["x", "y"]].to_numpy(dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    dist, j = cKDTree(xy[neg]).query(xy[pos])
    pair = np.full(len(d), -1)
    pair[pos] = neg[j]
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({"pid_pos": d.pid.to_numpy()[pos], "pid_neg": d.pid.to_numpy()[neg[j]], "dist_m": dist.round(1),
                  "same_county": (d.county.to_numpy()[pos] == d.county.to_numpy()[neg[j]]),
                  "n_positives_sharing_negative": pd.Series(neg[j]).map(pd.Series(neg[j]).value_counts()).to_numpy()}
                 ).to_csv(f"{OUT}/pairs_{region}.csv", index=False)
    return d, num, cat, y, xy, pair


def metrics(y, p):
    if p is None or len(y) == 0 or y.min() == y.max():
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


class Runs:
    """Incremental, resumable runs file: rows already present (by RUN_KEYS) are skipped."""

    def __init__(self, exp, region, suffix):
        os.makedirs(f"{OUT}/{exp}", exist_ok=True)
        self.path = f"{OUT}/{exp}/runs_{region}_{FS[region]}{suffix}.csv"
        self.keys = RUN_KEYS[exp]
        self.rows = pd.read_csv(self.path).to_dict("records") if os.path.exists(self.path) else []
        self.done = {tuple(str(r[k]) for k in self.keys) for r in self.rows}
        self.n_new = 0

    def has(self, base, model):
        return tuple(str({**base, "model": model}[k]) for k in self.keys) in self.done

    def add(self, row):
        self.rows.append(row)
        self.n_new += 1
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in self.keys or k in METRIC_COLS}, flush=True)
        if self.n_new % 20 == 0:
            self.save()

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


# ---------------------------------------------------------------- Experiment A: county few-shot with a spatial buffer
def draw_training(mode, pos, neg, pair, k, rng):
    """-> (train idx, n_pair_fallback). paired: k positives + their nearest ring negatives (fallback: random in-county ring
    negative when the pair lies outside the county or two positives share one negative); independent: k positives, k negatives."""
    trp = rng.choice(pos, k, replace=False)
    if mode == "independent":
        return np.concatenate([trp, rng.choice(neg, min(k, len(neg)), replace=False)]), 0
    pn = pair[trp]
    trn = np.unique(pn[np.isin(pn, neg)])
    need = min(k, len(neg)) - len(trn)
    if need > 0:
        trn = np.concatenate([trn, rng.choice(np.setdiff1d(neg, trn), need, replace=False)])
    return np.concatenate([trp, trn]), int(need)


def county(a, region):
    d, num, cat, y, xy, pair = region_data(region)
    cols, blocks = num + cat, d.block_id.to_numpy()
    runs = Runs("county", region, a.suffix)
    counts = d[d.label == 1].county.value_counts()
    for c in counts[counts >= MIN_COUNTY].index:
        inc = np.flatnonzero(d.county.to_numpy() == c)
        pos, neg = inc[y[inc] == 1], inc[y[inc] == 0]
        for k in a.k:
            if k > len(pos) - 25 or k > len(neg):
                continue
            for seed in range(a.seeds):
                for mode in a.modes:
                    rng = np.random.default_rng(seed)  # same seed -> same positives in both modes
                    tr, n_fallback = draw_training(mode, pos, neg, pair, k, rng)
                    te_in = np.setdiff1d(inc, tr)
                    dist_tr = cKDTree(xy[tr]).query(xy)[0]  # distance of every row to the nearest training row
                    te_nb = np.flatnonzero((dist_tr <= NEIGHBOUR_KM * 1000) & (d.county.to_numpy() != c))
                    te_all = np.union1d(te_in, te_nb)
                    paired_with_train = np.unique(np.concatenate([pair[tr[y[tr] == 1]], np.flatnonzero(np.isin(pair, tr))]))
                    base = {"region": region, "feature_set": FS[region], "county": c, "k": k, "rep": seed, "mode": mode}
                    todo = [m for m in a.models if not all(runs.has({**base, "test": "in_county", "buffer_km": B}, m) for B in a.buffers)]
                    for m in todo:
                        t0 = time.time()
                        p = fit_predict(m, num, cat, seed, d.loc[tr, cols], y[tr], d.loc[te_all, cols], blocks[tr])
                        sec = round(time.time() - t0, 2)
                        pm = pd.Series(p, index=te_all) if p is not None else None
                        common = {**base, "model": m, "n_train": len(tr), "pos_train": int(y[tr].sum()), "seconds": sec,
                                  "n_pair_fallback": n_fallback}
                        for B in a.buffers:
                            te = te_in[dist_tr[te_in] > B * 1000]
                            yt = y[te]
                            skipped = (yt.sum() < MIN_CLASS) or ((len(yt) - yt.sum()) < MIN_CLASS)
                            auc, pr = (np.nan, np.nan) if skipped or pm is None else metrics(yt, pm.loc[te].to_numpy())
                            runs.add({**common, "test": "in_county", "buffer_km": B, "n_test": len(te), "pos_test": int(yt.sum()),
                                      "n_test_paired_with_train": int(np.isin(te, paired_with_train).sum()), "skipped": bool(skipped),
                                      "roc_auc": auc, "pr_auc": pr})
                        if len(te_nb) >= MIN_CLASS and len(np.unique(y[te_nb])) == 2:
                            auc, pr = metrics(y[te_nb], None if pm is None else pm.loc[te_nb].to_numpy())
                            runs.add({**common, "test": "neighbours", "buffer_km": np.nan, "n_test": len(te_nb), "pos_test": int(y[te_nb].sum()),
                                      "n_test_paired_with_train": int(np.isin(te_nb, paired_with_train).sum()), "skipped": False,
                                      "roc_auc": auc, "pr_auc": pr})
                    runs.save()
    runs.save()


# ---------------------------------------------------------------- Experiment B: buffered learning curve on the committed folds
def curve(a, region):
    d, num, cat, y, xy, _ = region_data(region)
    cols, blocks = num + cat, d.block_id.to_numpy()
    runs = Runs("curve", region, a.suffix)
    fold_col = d.fold_spatial.to_numpy()
    for f in sorted(np.unique(fold_col)):
        if a.folds and str(f) not in a.folds:
            continue
        te, tr_all = np.flatnonzero(fold_col == f), np.flatnonzero(fold_col != f)
        dist_te = cKDTree(xy[te]).query(xy[tr_all])[0]  # distance of every training-side row to the nearest test row
        Xte, yte = d.loc[te, cols], y[te]
        for B in a.buffers:
            cand = tr_all[dist_te > B * 1000]
            info = {"n_candidates": len(cand), "pos_candidates": int(y[cand].sum()), "n_test": len(te), "pos_test": int(yte.sum())}
            print(f"{region} fold {f} B={B} km: {len(cand)} candidates ({info['pos_candidates']} pos) of {len(tr_all)}", flush=True)
            for n in a.sizes + [FULL]:
                reps = 1 if n == FULL else (a.reps if n <= 1000 else 3)
                for rep in range(reps):
                    base = {"region": region, "feature_set": FS[region], "fold": f"f{f}", "buffer_km": B, "n": n, "rep": rep}
                    if n != FULL and n > len(cand) - 10:
                        continue  # not enough candidates for a matched draw of this size
                    idx = cand if n == FULL else train_test_split(cand, train_size=n, stratify=y[cand], random_state=rep)[0]
                    for m in a.models:
                        if runs.has(base, m):
                            continue
                        t0 = time.time()
                        p = fit_predict(m, num, cat, rep, d.loc[idx, cols], y[idx], Xte, blocks[idx])
                        auc, pr = metrics(yte, p)
                        runs.add({**base, "model": m, "n_train": len(idx), "pos_train": int(y[idx].sum()), **info, "roc_auc": auc,
                                  "pr_auc": pr, "seconds": round(time.time() - t0, 2), "skipped": p is None})
            runs.save()
    runs.save()


# ---------------------------------------------------------------- Experiment C: Kentucky spatial-cell resampling
def load_ky_oof():
    base = pd.read_parquet(f"{ROOT}/results/kentucky/pred_loco.parquet")
    key = ["dep_id", "label", "fold"]
    for f in ["kentucky_tuned", "kentucky_fm_fp32", "kentucky_tabicl_fp32"]:
        e = pd.read_parquet(f"{ROOT}/results/{f}/pred_loco.parquet")
        base = base.merge(e[[c for c in e.columns if c not in base.columns or c in key]], on=key, how="inner")
    ll = pd.read_parquet(f"{ROOT}/data/processed/kentucky/depressions_all.parquet", columns=["dep_id", "lon", "lat"])
    base = base.merge(ll, on="dep_id", how="left")
    base["cell"] = ky_groups(base)
    return base


def kentucky_cells(a):
    out = f"{OUT}/kentucky_cells"
    os.makedirs(out, exist_ok=True)
    df = load_ky_oof()
    fms = [m for m in ["tabpfn", "tabicl", "kumo_tabular"] if m in df]
    refs = [r for r in ["xgboost_tuned", "xgboost"] if r in df]
    cells = df.groupby("fold").agg(n_rows=("dep_id", "size"), n_pos=("label", "sum"), n_cells=("cell", "nunique")).reset_index()
    cells = pd.concat([cells, pd.DataFrame([{"fold": "pooled", "n_rows": len(df), "n_pos": int(df.label.sum()), "n_cells": df.cell.nunique()}])])
    cells.to_csv(f"{out}/cells_per_county.csv", index=False)
    # seed-study predictions (long) -> (seeds, rows) matrices aligned to df
    seeds = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/results/seeds/kentucky/preds_*.parquet"))], ignore_index=True)
    seeds = seeds[seeds.setting == "loco"].drop_duplicates(subset=["model", "seed", "id"])
    mat = seeds.pivot_table(index="id", columns=["model", "seed"], values="p").reindex(df.dep_id.to_numpy())
    P = {m: mat[m].dropna(axis=1, how="all").to_numpy().T for m in mat.columns.get_level_values(0).unique()}
    P = {m: v for m, v in P.items() if not np.isnan(v).any()}
    rng, rows = np.random.default_rng(0), []
    levels = [("pooled", df)] + [(c, g) for c, g in df.groupby("fold")]
    for level, g in levels:
        y, cell = g.label.to_numpy(), g.cell.to_numpy()
        for ref in refs:
            for m in fms:
                pm, pr = g[m].to_numpy(dtype=float), g[ref].to_numpy(dtype=float)
                d_row = boot(y, pm, pr, np.arange(len(g)), a.n_boot, rng)
                d_cell = boot(y, pm, pr, cell, a.n_boot, rng)
                delta, se, pval = delong(y, pm, pr)
                r = {"level": level, "ref": ref, "model": m, "n_rows": len(g), "n_pos": int(y.sum()), "n_cells": len(np.unique(cell)),
                     "auc_model": roc_auc_score(y, pm), "auc_ref": roc_auc_score(y, pr), "dAUC": roc_auc_score(y, pm) - roc_auc_score(y, pr),
                     "row_ci95_lo": np.percentile(d_row, 2.5), "row_ci95_hi": np.percentile(d_row, 97.5), "row_p_not_better": float((d_row <= 0).mean()),
                     "cell_ci95_lo": np.percentile(d_cell, 2.5), "cell_ci95_hi": np.percentile(d_cell, 97.5), "cell_p_not_better": float((d_cell <= 0).mean()),
                     "delong_p": pval}
                if m in P and ref in P:
                    sel = (df.fold.to_numpy() == level) if level != "pooled" else np.ones(len(df), bool)
                    ds = boot_block_seed(y, P[m][:, sel], P[ref][:, sel], cell, a.n_boot, rng)
                    r.update(n_seeds=min(len(P[m]), len(P[ref])), seedcell_dAUC_mean=np.mean([roc_auc_score(y, q) for q in P[m][:, sel]])
                             - np.mean([roc_auc_score(y, q) for q in P[ref][:, sel]]),
                             seedcell_ci95_lo=np.percentile(ds, 2.5), seedcell_ci95_hi=np.percentile(ds, 97.5), seedcell_p_not_better=float((ds <= 0).mean()))
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/deltas.csv", index=False)


# ---------------------------------------------------------------- summaries
def all_runs(exp):
    files = sorted(glob.glob(f"{OUT}/{exp}/runs_*.csv"))
    if not files:
        return pd.DataFrame()
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return r.drop_duplicates(subset=RUN_KEYS[exp], keep="first")


def fmt_n(df):
    df = df.reset_index()
    if "n" in df:
        df["n"] = df["n"].astype(object).where(df["n"] != FULL, "full")
    return df.to_markdown(index=False)


def paired_delta(r, keys, pair_keys, ref=REF):
    """Mean AUC per keys x model and paired ΔAUC vs ref over pair_keys: 'mean ± 1.96·SE (share > 0)' with n pairs."""
    piv = r.pivot_table(index=keys + pair_keys, columns="model", values="roc_auc")
    mean = piv.groupby(level=keys).mean()
    cnt = piv.groupby(level=keys).count()
    mean.insert(0, "cells", cnt.max(axis=1).astype(int))
    if ref not in piv:
        return mean, cnt, None
    g = piv.drop(columns=ref).sub(piv[ref], axis=0).groupby(level=keys)
    m, h, w, n = g.mean(), 1.96 * g.sem(), g.agg(lambda x: (x.dropna() > 0).mean()), g.count()
    cell = pd.DataFrame({c: [f"{x:+.4f} ± {y:.4f} ({z:.0%}, n={int(q)})" if q > 1 else "" for x, y, z, q in zip(m[c], h[c], w[c], n[c])]
                         for c in m.columns}, index=m.index)
    return mean, cnt, cell


def summarize_county(txt):
    r = all_runs("county")
    if r.empty:
        return r
    ok = r[~r.skipped.astype(bool) & r.roc_auc.notna()].copy()
    ok["B"] = ok.buffer_km.where(ok.test == "in_county", np.inf)  # np.inf = neighbours test, so the tables sort by distance
    txt.append("\n## 1. Experiment A — county few-shot with a train–test buffer\n\n"
               f"Counties with ≥ {MIN_COUNTY} positives; training = k in-county positives + k in-county ring negatives; 3 seeds; the same "
               "fitted model is scored once and evaluated on the in-county rows farther than B km from every training row (B = 0 is the "
               "committed design) and on the `neighbours` test (other counties within 50 km of a training row). `paired` = the k positives' "
               "nearest ring negatives (random in-county fallback where the pair is outside the county or shared); `independent` = positives "
               f"and negatives drawn independently. Cells with < {MIN_CLASS} positives or negatives are skipped. Δ bands are descriptive: "
               "pairs share counties and seeds.\n")
    nb = lambda df: df.reset_index().assign(B=lambda x: x.B.astype(object).where(np.isfinite(x.B), "neighbours")).to_markdown(index=False)  # noqa: E731
    for reg, g in ok.groupby("region"):
        cells = r[(r.region == reg) & (r.test == "in_county")].drop_duplicates(subset=["county", "k", "rep", "mode", "buffer_km"])
        skip = cells.groupby(["mode", "k", "buffer_km"]).agg(
            cells=("skipped", "size"), skipped=("skipped", lambda s: int(s.astype(bool).sum())), n_test=("n_test", "mean"),
            pos_test=("pos_test", "mean"), test_rows_paired_with_train=("n_test_paired_with_train", "mean")).round(1)
        txt.append(f"\n### {reg} / {FS[reg]}: in-county test set per cell (mean over counties × seeds) and skipped cells\n\n"
                   + skip.reset_index().to_markdown(index=False) + "\n")
        for mode, gm in g.groupby("mode"):
            mean, cnt, delta = paired_delta(gm, ["k", "B"], ["county", "rep"])
            txt.append(f"\n### {reg} / {FS[reg]} / {mode}: mean in-county AUC over county × seed (B = buffer km; `neighbours` = other counties ≤ 50 km)\n\n"
                       + nb(mean.round(4)) + "\n")
            if delta is not None:
                txt.append(f"\n### {reg} / {FS[reg]} / {mode}: paired ΔAUC vs {REF} (mean ± 1.96 s.e. over county × seed pairs, share > 0; "
                           "descriptive, pairs share counties)\n\n" + nb(delta) + "\n")
        both = g[(g.test == "in_county")].pivot_table(index=["k", "buffer_km", "county", "rep", "model"], columns="mode", values="roc_auc")
        if {"paired", "independent"} <= set(both.columns):
            dd = (both.paired - both.independent).dropna().groupby(level=["k", "buffer_km", "model"]).agg(["mean", "count"])
            dd["cell"] = [f"{m:+.4f} (n={int(n)})" for m, n in zip(dd["mean"], dd["count"])]
            txt.append(f"\n### {reg}: paired − independent sampling (same positives, same seed; cells evaluated in both modes), mean over county × seed\n\n"
                       + dd.reset_index().pivot_table(index=["k", "buffer_km"], columns="model", values="cell", aggfunc="first")
                       .reset_index().to_markdown(index=False) + "\n")
    return ok


def summarize_curve(txt):
    r = all_runs("curve")
    if r.empty:
        return r
    ok = r[~r.skipped.astype(bool) & r.roc_auc.notna()]
    txt.append("\n## 2. Experiment B — buffered learning curve on the committed spatial folds\n\n"
               "Test = complete outer fold of the committed 25 km block CV; training candidates = rows outside the fold and farther than "
               "B km from every test row; matched stratified draws of n rows (5 reps for n ≤ 1000, 3 for 2000, identical rows for every "
               "model), `full` = all candidates. Sampling is independent of pairs (pairs are treated in Experiment A).\n")
    for reg, g in ok.groupby("region"):
        cand = g.groupby(["fold", "buffer_km"]).agg(n_candidates=("n_candidates", "first"), pos_candidates=("pos_candidates", "first"),
                                                     n_test=("n_test", "first"), pos_test=("pos_test", "first")).reset_index()
        txt.append(f"\n### {reg} / {FS[reg]}: candidate training rows per fold and buffer\n\n"
                   + cand.pivot_table(index="fold", columns="buffer_km", values="n_candidates").to_markdown() + "\n")
        mean, cnt, delta = paired_delta(g, ["buffer_km", "n"], ["fold", "rep"])
        txt.append(f"\n### {reg} / {FS[reg]}: mean fold AUC\n\n" + fmt_n(mean.round(4)) + "\n")
        if delta is not None:
            txt.append(f"\n### {reg} / {FS[reg]}: paired ΔAUC vs {REF} over fold × rep pairs (mean ± 1.96 s.e., share > 0)\n\n" + fmt_n(delta) + "\n")
        have = [m for m in [REF, "xgboost"] if m in g.model.unique()]
        if have:
            txt.append(f"\n### {reg}: models with {', '.join(have)} available per buffer\n\n"
                       + g[g.model.isin(have)].groupby(["buffer_km", "model"]).n.nunique().unstack().to_markdown() + "\n")
        loss = []
        for (m, n), gg in g.groupby(["model", "n"]):
            a0 = gg[gg.buffer_km == 0].roc_auc.mean()
            row = {"model": m, "n": n, "auc_B0": a0}
            for B in (5, 10):
                aB = gg[gg.buffer_km == B].roc_auc.mean()
                row[f"loss_B{B}"] = a0 - aB
                row[f"loss_per_km_B{B}"] = (a0 - aB) / B
            loss.append(row)
        loss = pd.DataFrame(loss)
        for B in (5, 10):
            cell = loss.apply(lambda s: f"{s[f'loss_B{B}']:+.4f} ({s[f'loss_per_km_B{B}']:+.4f}/km)", axis=1)
            txt.append(f"\n### {reg}: AUC lost to a {B} km buffer (mean AUC at B = 0 minus mean AUC at B = {B}; per km in brackets)\n\n"
                       + fmt_n(loss.assign(cell=cell).pivot_table(index="n", columns="model", values="cell", aggfunc="first")) + "\n")
    return ok


def summarize_cells(txt):
    p = f"{OUT}/kentucky_cells/deltas.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    cells = pd.read_csv(f"{OUT}/kentucky_cells/cells_per_county.csv")
    txt.append("\n## 3. Experiment C — Kentucky loco: row bootstrap vs 0.05° cell bootstrap\n\n"
               "Paired ΔAUC on the committed out-of-fold predictions (`results/kentucky*/pred_loco.parquet`), 1000 resamples; `row` = the "
               "committed row bootstrap, `cell` = cluster bootstrap over 0.05° lat/lon cells (`seed_study.ky_groups`), `seedcell` = cells × "
               "independent seed draw from the 5-seed study (`results/seeds/kentucky`). DeLong p on the single-seed OOF predictions.\n\n"
               "Cells per held-out county:\n\n" + cells.to_markdown(index=False) + "\n")
    f = lambda lo, hi: f"[{lo:+.4f}, {hi:+.4f}]"  # noqa: E731
    d["row CI"] = [f(a, b) for a, b in zip(d.row_ci95_lo, d.row_ci95_hi)]
    d["cell CI"] = [f(a, b) for a, b in zip(d.cell_ci95_lo, d.cell_ci95_hi)]
    if "seedcell_ci95_lo" in d:
        d["seed×cell CI"] = [f(a, b) if pd.notna(a) else "" for a, b in zip(d.seedcell_ci95_lo, d.seedcell_ci95_hi)]
    cols = ["level", "ref", "model", "n_rows", "n_cells", "auc_model", "auc_ref", "dAUC", "row CI", "cell CI"] + (["seed×cell CI"] if "seed×cell CI" in d else []) + ["delong_p"]
    for ref, g in d.groupby("ref"):
        order = ["pooled"] + sorted(x for x in g.level.unique() if x != "pooled")
        g = g.set_index("level").loc[order].reset_index()
        txt.append(f"\n### ref = {ref}\n\n" + g[cols].round(4).to_markdown(index=False) + "\n")


def plot_curve(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if r.empty:
        return
    lc = sorted(glob.glob(f"{ROOT}/results/fm_scenarios/learning_curve/runs*.csv"))  # same colour map as learning_curve.png
    names = sorted(pd.concat([pd.read_csv(f, usecols=["model"]) for f in lc]).model.unique()) if lc else sorted(r.model.unique())
    names = sorted(set(names) | set(r.model.unique()))
    colors = dict(zip(names, (plt.cm.tab10 if len(names) <= 10 else plt.cm.tab20).colors))
    regions, buffers = sorted(r.region.unique()), sorted(r.buffer_km.unique())
    fig, axes = plt.subplots(len(regions), len(buffers), figsize=(4.2 * len(buffers), 3.8 * len(regions)), squeeze=False, sharey="row")
    for i, reg in enumerate(regions):
        for j, B in enumerate(buffers):
            g = r[(r.region == reg) & (r.buffer_km == B)]
            x = g.groupby("n").n_train.mean()
            mean = g.groupby(["n", "model"]).roc_auc.mean().unstack()
            ax = axes[i, j]
            for m in mean.columns:
                fm = m in FM_MODELS
                ax.plot(x, mean[m], color=colors.get(m), marker="o", lw=2.5 if fm else 1.2, ls="-" if fm else "--", label=m, ms=4)
            ax.set(xscale="log", title=f"{reg}, buffer {B:g} km", xlabel="training labels")
            if j == 0:
                ax.set_ylabel("ROC-AUC (test fold)")
            ax.grid(alpha=.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(f"{OUT}/buffered_curve.png", dpi=130)
    plt.close(fig)


def plot_county(r, k=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = r[(r.k == k) & (r.test == "in_county")]
    if g.empty:
        return
    names = sorted(g.model.unique())
    lc = sorted(glob.glob(f"{ROOT}/results/fm_scenarios/learning_curve/runs*.csv"))
    names = sorted(set(names) | (set(pd.concat([pd.read_csv(f, usecols=["model"]) for f in lc]).model.unique()) if lc else set()))
    colors = dict(zip(names, (plt.cm.tab10 if len(names) <= 10 else plt.cm.tab20).colors))
    regions, modes = sorted(g.region.unique()), sorted(g["mode"].unique())
    fig, axes = plt.subplots(len(modes), len(regions), figsize=(5.5 * len(regions), 3.8 * len(modes)), squeeze=False)
    for i, mode in enumerate(modes):
        for j, reg in enumerate(regions):
            gg = g[(g.region == reg) & (g["mode"] == mode)]
            cells = gg.groupby("buffer_km").apply(lambda x: x.groupby("model").size().max())
            keep = cells[cells >= 5].index  # buffers where at least 5 county × seed cells survive
            mean = gg[gg.buffer_km.isin(keep)].groupby(["buffer_km", "model"]).roc_auc.mean().unstack()
            ax = axes[i, j]
            for m in mean.columns:
                fm = m in FM_MODELS
                ax.plot(mean.index, mean[m], color=colors.get(m), marker="o", lw=2.5 if fm else 1.2, ls="-" if fm else "--", label=m)
            for B in mean.index:
                ax.annotate(f"n={int(cells[B])}", (B, mean.loc[B].min()), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7)
            ax.set(title=f"{reg}, k = {k}, {mode} sampling", xlabel="train–test buffer (km)", ylabel="in-county ROC-AUC")
            ax.grid(alpha=.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("n = surviving county × seed cells (≥ 20 positives and negatives); buffers with < 5 cells not drawn", fontsize=9)
    fig.tight_layout(), fig.savefig(f"{OUT}/buffered_county.png", dpi=130)
    plt.close(fig)


def write_run_info(a, extra=None):
    """One JSON per process under results/buffered/run_info/; summarize merges them into run_info.json."""
    import importlib
    os.makedirs(f"{OUT}/run_info", exist_ok=True)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "not a git repository"
    except Exception:  # noqa: BLE001
        git = "not a git repository"
    vers = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "catboost", "optuna", "torch", "tabpfn", "tabicl", "structured_data_models"]:
        try:
            vers[pkg] = importlib.import_module(pkg).__version__
        except Exception:  # noqa: BLE001
            pass
    info = {"date": time.strftime("%Y-%m-%d %H:%M"), "host": platform.node(), "python": platform.python_version(), "git": git,
            "versions": vers, "args": vars(a), "fm_precision": os.environ.get("FM_PRECISION", "auto"), "device": os.environ.get("FM_DEVICE", "auto"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **(extra or {})}
    try:
        import torch
        if torch.cuda.is_available() and (a.models is None or any(m in FM_MODELS for m in a.models)):
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            info["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    tag = f"{a.experiment}_{'_'.join(a.regions or [])}{a.suffix}_{int(time.time())}"
    json.dump(info, open(f"{OUT}/run_info/{tag}.json", "w"), indent=1, default=str)


def summarize(a):
    txt = ["# Spatially separated small-data evaluation (results/buffered, `scripts/buffered_smalldata.py`)\n",
           "Question: are the small-data advantages of the foundation models interpolation artefacts? Florida = committed table, "
           "ring negatives, `no_landcover` (49 factors); Minnesota = shared public-layer schema `sat_nolc` (34). Models: xgboost (hand-set), "
           "xgboost_tuned (nested spatial CV, inner groups = 25 km blocks of the training rows), random_forest, TabPFN-3.5, TabICLv2, "
           "Kumo-Tabular (fp32, full context). The pair of a positive = its nearest ring negative (`pairs_<region>.csv`).\n"]
    rc = summarize_county(txt)
    rb = summarize_curve(txt)
    summarize_cells(txt)
    reading = f"{OUT}/reading.md"  # optional hand-written reading of the numbers (not part of the release)
    if os.path.exists(reading):
        txt.append("\n## 4. Reading\n\n" + open(reading).read().strip() + "\n")
    open(f"{OUT}/summary.md", "w").write("\n".join(txt))
    if not rb.empty:
        plot_curve(rb)
    if not rc.empty:
        plot_county(rc)
    infos = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/run_info/*.json"))]
    merged = {"date": time.strftime("%Y-%m-%d %H:%M"), "git": infos[-1]["git"] if infos else "not a git repository",
              "versions": infos[-1]["versions"] if infos else {}, "n_processes": len(infos),
              "fm_precision": sorted({i.get("fm_precision") for i in infos if i.get("gpu")}),  # FM processes only (CPU shards ran no FM)
              "device": sorted({i.get("device") for i in infos if i.get("gpu")}),
              "gpu": sorted({i.get("gpu") for i in infos if i.get("gpu")}),
              "gpu_peak_allocated_gb_max": max([i.get("gpu_peak_allocated_gb", 0) for i in infos] + [0]),
              "gpu_peak_reserved_gb_max": max([i.get("gpu_peak_reserved_gb", 0) for i in infos] + [0]),
              "processes": infos}
    json.dump(merged, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
    print(open(f"{OUT}/summary.md").read())


EXPERIMENTS = {"county": county, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=list(EXPERIMENTS) + ["kentucky_cells", "summarize"])
    ap.add_argument("--regions", nargs="*", default=["florida", "minnesota"])
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--k", nargs="*", type=int, default=[25, 50, 100])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", nargs="*", default=["paired", "independent"])
    ap.add_argument("--buffers", nargs="*", type=float, default=None, help="km; default A: 0 1 2.5 5 10, B: 0 2.5 5 10")
    ap.add_argument("--folds", nargs="*", default=None, help="outer folds to run (curve), e.g. 0 1")
    ap.add_argument("--sizes", nargs="*", type=int, default=SIZES)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--suffix", default="", help="runs file suffix for shards (e.g. _tuned, _f01)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.experiment == "summarize":
        return summarize(a)
    t0 = time.time()
    if a.experiment == "kentucky_cells":
        kentucky_cells(a)
    else:
        a.buffers = [float(b) for b in (a.buffers or (BUFFERS_A if a.experiment == "county" else BUFFERS_B))]
        for region in a.regions:
            EXPERIMENTS[a.experiment](a, region)
    write_run_info(a, {"wall_seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
