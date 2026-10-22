# sinkhole-fm-benchmark

Code, benchmark tables and results for

> K. Banjare, *Tabular foundation models versus tuned gradient boosting for sinkhole susceptibility in data-poor karst
> regions*. Submitted to *Computers & Geosciences*, 2026.

Three tabular foundation models (TabPFN-3.5, TabICLv2, NVIDIA Kumo-Tabular) are used **without any training or tuning**:
the labelled sites are a fixed context at inference time. They are compared with default and nested-spatial-CV-tuned
gradient boosting (XGBoost, LightGBM, CatBoost), random forest, RBF-SVM and logistic regression. All models use identical
folds and pseudo-absences in three karst regions:

| Region | Inventory | Rows | Features | Validation |
|---|---|---|---|---|
| Florida | 2,736 FGS Subsidence Incident Reports + ring / global pseudo-absences | 8,208 | 68 (headline `no_landcover`: 49) | random 5-fold, **25 km spatial-block 5-fold**, **temporal** (< 2010 → ≥ 2010) |
| Kentucky Bluegrass | LiDAR depressions labelled by the KGS sinkhole inventory ([Zhu et al. 2020](https://doi.org/10.1016/j.jhydrol.2020.125049) task) | 49,926 | 10 morphometric | random 80/20 × 5, **leave-one-county-out**, 3 counties → Scott |
| SE Minnesota | 13,144 Karst Feature Inventory sinkholes + paired ring pseudo-absences | 39,328 | satellite-only schema (headline `sat_nolc`: 34) | random 5-fold, **25 km spatial-block 5-fold**, location-method split |

Settings in bold form the headline family. It was fixed before the tuned baselines and seed replications were run; the
definition is in [`results/headline_family.md`](results/headline_family.md).

## Headline results

Pooled out-of-fold ROC-AUC (seed 0). The last column is the five-seed ΔAUC of the best foundation model against tuned
XGBoost, with a 95 % interval that resamples spatial blocks and seeds (Kentucky: rows).

| Setting | tuned XGBoost | TabPFN-3.5 | TabICLv2 | Kumo-Tabular | Best FM, 5-seed ΔAUC [95 % CI] |
|---|---|---|---|---|---|
| Florida, no land cover, ring, spatial-block CV | 0.765 | **0.785** | 0.775 | 0.767 | TabPFN +0.012 [−0.004, +0.029] |
| Florida, temporal split | 0.770 | 0.790 | 0.787 | **0.799** | Kumo +0.027 [+0.013, +0.043] |
| Kentucky, leave-one-county-out | 0.965 | 0.971 | **0.972** | 0.971 | TabICL +0.006 [+0.005, +0.007] |
| Minnesota, `sat_nolc`, ring, spatial-block CV | 0.836 | **0.852** | 0.840 | 0.837 | TabPFN +0.015 [−0.004, +0.052] |

The foundation models gain most when data are scarce. They score +0.02 to +0.03 AUC higher than tuned XGBoost with at most
1,000 labels, +0.03 to +0.05 with 25 to 100 labels per county and +0.03 with five conditioning factors. Under 30 % label
noise, two of the three lose less AUC than tuned XGBoost. They show no advantage once a land-cover proxy of reporting effort
is among the features. They are not better calibrated under spatial shift, and they do not transfer zero-shot between
states. [`results/README.md`](results/README.md) has every table with confidence intervals, seed statistics and DeLong/Holm
tests.

## Repository layout

```
baselines/common.py           classical/GBDT model zoo + metrics (factory: make_x(num_cols, cat_cols, seed))
baselines/fm_models.py        foundation-model adapters, same factory signature, configured only through env / .env
baselines/tuned.py            nested-CV tuned baselines: Optuna, 100 trials, inner 4-fold spatial GroupKFold, early stopping
baselines/run_florida_baselines.py   Florida: negatives {global, ring} x protocol {random5, spatial5, temporal}
baselines/run_kentucky_baselines.py  Kentucky depressions: {paper80_20, loco, scott}
baselines/run_region_baselines.py    any region on the shared schema (--region minnesota), same layout as Florida
baselines/compare_models.py   paired (block-)bootstrap ΔAUC against a reference model on identical OOF predictions
baselines/replicate_subedi2019.py    Marion County replication of Subedi et al. (2019)
scripts/run_tuned_baselines.py       tuned grid, parallel over outer folds, resumable (search records in results/tuning_cache/)
scripts/seed_study.py         5 seeds on the headline family, block x seed bootstrap, DeLong + Holm      -> results/seeds
scripts/fm_scenarios.py       learning curve, label noise, rotation, calibration, blends              -> results/fm_scenarios
scripts/feature_sets.py       feature-source ablation, 5 to 68 factors                               -> results/fm_scenarios/feature_sets
scripts/realistic_benchmark.py county few-shot, clean-label test sets, 1:10 / 1:20 prevalence         -> results/realistic
scripts/transfer.py           fixed-context transfer Florida <-> Minnesota with k local labels         -> results/transfer
scripts/context_selection.py  state / neighbour / kNN / random contexts per Florida county             -> results/context_selection
scripts/decision_analysis.py  reliability, prioritisation, decision curves                            -> results/decision
scripts/fm_cost.py            context-size scaling, latency, GPU memory, CPU-only county model         -> results/fm_cost
scripts/fm_sensitivity.py     seed x n_estimators sensitivity of the FMs                             -> results/fm_sensitivity
scripts/interpret.py          permutation / group importance, partial dependence                       -> results/interpret
scripts/florida/03_predict_raster.py  1 km statewide Florida maps from one fixed context              -> results/maps
scripts/positional_accuracy.py AUC by report location-accuracy class                                  -> results/positional_accuracy.md
scripts/florida/, kentucky/, minnesota/, shared_features.py   data pipelines (need data/raw)
scripts/regen_comparisons.sh  every paired comparison table; scripts/collect_results.py -> results/README.md
scripts/reproduce_check.py    CPU check: refits the baselines and compares with the committed per-fold metrics
scripts/verify_headline.py    CPU check: recomputes the headline tables from the released predictions (read-only)
scripts/check_env.py          pre-flight: versions, GPU, weight sources, 1-minute smoke test per model
data/processed/               the four benchmark tables with their fold columns (see data/README.md)
results/                      metrics, summaries, comparison tables, run manifests (run_info.json), GeoTIFF maps, tuning records,
                              out-of-fold predictions of the headline family
results/README.md             every table in one page (generated by scripts/collect_results.py)
results/headline_family.md    the four settings fixed before the confirmatory runs
REPRODUCE.md                  step-by-step replication guide
MANIFEST.sha256               sha256 of every table, summary, prediction and raster
```

## Out-of-fold predictions

The out-of-fold predictions of every model in the headline family are in the repository (`results/**/*.parquet`):

| Setting | Files |
|---|---|
| Florida `no_landcover` × ring, spatial-block and temporal | `results/florida_{no_landcover,tuned_no_landcover,fm_fp32_no_landcover,tabicl_fp32_no_landcover}/oof_ring_{spatial5,temporal}.parquet` |
| Kentucky, leave-one-county-out and Scott County | `results/kentucky{,_tuned,_fm_fp32,_tabicl_fp32}/pred_{loco,scott}.parquet` |
| Minnesota `sat_nolc` × ring, spatial-block and location-method split | `results/minnesota{,_tuned,_fm_fp32}/oof_ring_{spatial5,temporal}.parquet` |
| Five seeds of the headline family | `results/seeds/{florida,kentucky,minnesota}/preds_*.parquet` (long format: `setting, model, seed, fold, id, label, p`) |

Florida and Minnesota files have one row per point (`pid`, `label`, the fold columns, one probability column per model).
Kentucky files have one row per depression (`dep_id`, `county`, `label`, `fold`, one column per model).
`python scripts/verify_headline.py` recomputes the headline comparisons, the five-seed statistics and the decision table
from these files and compares them with the committed results without overwriting anything. AUC, PR-AUC and ΔAUC come
out identical. The bootstrap intervals differ by Monte-Carlo noise (up to 0.006), because `compare_models.py` draws all
settings of a table from one random stream and the non-headline settings are not included.

The predictions of the other settings and studies are not distributed. These are the global pseudo-absences, random CV,
the all-feature and shared-schema grids, the learning curves, label noise, the realistic regime, transfer, context
selection, cost, maps and factors. Their per-fold metrics and summaries are in `results/`, and the scripts regenerate the
predictions:
- **Baselines** need no GPU and regenerate exactly. Predictions of XGBoost, random forest, logistic regression and
  tuned XGBoost rebuilt on a second machine were identical to the released ones on the headline Florida folds. Tuned models refit
  from the search records in `results/tuning_cache/` in seconds, without repeating the 100-trial searches.
- **Foundation models** need a GPU for full-size contexts; their weights download from Hugging Face on first use. Expect
  agreement within numerical precision rather than bit-identical predictions; the bf16 and fp32 runs differ by at most
  0.006 AUC.

`scripts/regen_comparisons.sh` rewrites every comparison table from whatever predictions are present. Run it only after
regenerating all settings of a table; to check the released numbers, use `verify_headline.py`. Without any predictions,
`compare_models.py`, `decision_analysis.py` and `seed_study.py summarize` stop with a message instead of overwriting the
committed summaries.

Not distributed: `data/raw/` (about 8 GB of public layers; [`data/README.md`](data/README.md) lists the download scripts
and sources), the Florida 1 km grid feature and prediction tables (`scripts/florida/03_predict_raster.py` rebuilds them
from `data/raw`; the resulting maps are in `results/maps/`) and the model weights (see below).

## Quick start: verify the release (no weights, no raw data)

These steps verify the release in a few minutes; [`REPRODUCE.md`](REPRODUCE.md) explains each check.

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt          # exact versions used for the results: requirements.lock.txt

sha256sum -c MANIFEST.sha256 --quiet && echo "checksums OK"
python scripts/reproduce_check.py           # ~1 min: XGBoost / RF / LR on the headline Florida folds = committed metrics
python scripts/verify_headline.py --checks comparisons --tasks florida   # ~5 min; drop the options for all checks (~1 h)
python scripts/collect_results.py           # rebuilds results/README.md from results/ (identical to the committed file)
python scripts/fm_scenarios.py summarize    # learning-curve / noise summaries + results/fm_scenarios/learning_curve/learning_curve.png
```

The baselines need no GPU, for example
`python baselines/run_florida_baselines.py --feature-set no_landcover --models logreg random_forest xgboost --tag rerun`.
The runners set `OMP_NUM_THREADS=4` per process. Run tuning and GBDT grids with `CUDA_VISIBLE_DEVICES=""` so their
workers do not hold idle CUDA contexts on the GPUs.

## Full reproduction

[`REPRODUCE.md`](REPRODUCE.md) gives every step, from verifying the release to rerunning the baselines and the foundation
models and rebuilding the tables from raw data. It lists the exact command behind each result directory and paper
item, the expected agreement, and the hardware used. For example, rerunning the Florida headline grid under a new tag,
so the committed results stay untouched:

```bash
FM_PRECISION=fp32 python scripts/check_env.py         # weight sources + 1-minute smoke test per model
python baselines/run_florida_baselines.py --feature-set no_landcover --tag rerun_no_landcover          # CPU
FM_PRECISION=fp32 TAG=rerun_fp32 MODELS="tabpfn kumo_tabular" TASKS=florida bash scripts/run_fm_experiments.sh
python baselines/compare_models.py florida results/florida_rerun_no_landcover results/florida_rerun_fp32_no_landcover
```

Each script prints its options with `--help`, except `decision_analysis.py` and `positional_accuracy.py`, which take no
options. Each result directory has a `run_info.json` that records the package
versions, the checkpoint, the precision and the arguments of its run. Weight directories are shown as `<WEIGHTS_DIR>`.

**Hardware used.** The fp32 foundation-model runs used one NVIDIA RTX PRO 6000 Blackwell (96 GB); the largest fit peaked
at 9.3 GiB. The Kentucky learning-curve and label-noise shards ran in bf16 on four A10 (24 GB). bf16 and fp32 agree within
0.006 AUC on every grid setting. Tuning ran on 128 CPU cores. A full foundation-model grid takes about 15 min, and the
complete study about 12 h on that machine.

**Foundation-model settings** live only in environment variables or `.env` (see [`.env.example`](.env.example)).
- `FM_PRECISION` (`auto|fp32|bf16|fp16`) selects autocast for TabPFN and Kumo.
- `FM_MAX_CONTEXT` caps the context on small GPUs. Report it if you set it; Kentucky contexts have 27k to 35k rows.
- Lower `TABICL_BATCH_SIZE` if TabICL runs out of memory.

**Weights.** You don't need to download weights by hand. When no local path is set (`TABPFN_MODEL_PATH`,
`TABICL_MODEL_PATH`, `KUMO_CHECKPOINT_DIR`), each model downloads its checkpoint from Hugging Face on first use. TabICL and Kumo go into
the standard HF cache (`HF_HOME` / `HF_HUB_CACHE`); TabPFN goes into `TABPFN_MODEL_CACHE_DIR` (default
`<HF_HOME>/tabpfn/<commit>/`). Set `HF_HUB_OFFLINE=1` afterwards to run offline. TabPFN is pinned to the
file and commit behind the reported results. `python scripts/check_env.py` prints where each model's weights will come
from and whether they are already cached.

## Where each paper item comes from

| Paper | Files |
|---|---|
| Table 1, Fig. 1 (data, folds) | `data/processed/*`, `data/README.md`, `scripts/{florida,kentucky,minnesota}/` |
| Table 2 (headline family) | `results/comparison_florida_no_landcover_tuned_fp32_ref_xgbtuned.csv`, `results/comparison_kentucky_tuned_fp32_ref_xgbtuned.csv`, `results/comparison_minnesota_sat_nolc_ref_xgbtuned.csv`, `results/seeds/summary.csv` |
| Table 3, Figs. 2–3 (data-poor regime) | `results/fm_scenarios/{learning_curve,label_noise,feature_sets}/`, `results/realistic/county/` |
| Table 4, Fig. 4 (calibration, prioritisation) | `results/decision/`, `results/fm_scenarios/calibration/`, `results/realistic/prevalence/` |
| Transfer, context selection | `results/transfer/`, `results/context_selection/` |
| Table 5 (cost) | `results/fm_cost/` |
| Fig. 5 (statewide maps) | `results/maps/*.tif`, `results/maps/summary.md` |
| Fig. 6 (conditioning factors) | `results/interpret/` |
| Supplementary tables | `results/README.md` (all sections), `results/tuning_cache/` and `baselines/tuned.py` (search spaces), `scripts/feature_sets.py` (feature sets) |

## Adding a model

To add a model, write a factory `make_x(num_cols, cat_cols, seed)` that returns a scikit-learn-style classifier. Register
it in `MODELS` (`baselines/common.py`) or `FM_MODELS` (`baselines/fm_models.py`). Every runner then accepts it through
`--models`. Write its results under a new `--tag` and compare them on the same out-of-fold rows, for example
`python baselines/compare_models.py florida results/florida_no_landcover results/florida_<tag> --ref xgboost`. Do not
change the committed folds, seeds, pseudo-absences or feature lists, because every comparison depends on them.

## Model weights and licences

Model weights are not included in this repository. The adapters download them from Hugging Face, and each keeps its
own licence:
- TabPFN-3.5: [`Prior-Labs/tabpfn_3_5`](https://huggingface.co/Prior-Labs/tabpfn_3_5), file
  `tabpfn-v3.5-20260909.safetensors` at commit `06bf2ba` (SHA-256 `ece4d67e…0be3`). TABPFN-3.5 License v1.0: research
  and evaluation only, no commercial or production use of the model or its outputs.
- TabICLv2: [`jingang/TabICL`](https://huggingface.co/jingang/TabICL). Checkpoint `tabicl-classifier-v2-20260212.ckpt`.
- Kumo-Tabular large: [`nvidia/Kumo-Tabular`](https://huggingface.co/nvidia/Kumo-Tabular). Revision `v1.0.1`, OpenMDW-1.1.

The code is MIT-licensed ([`LICENSE`](LICENSE)). The benchmark tables are derived from public data; each source keeps its
own licence, and [`data/README.md`](data/README.md) lists them (FGS/FDEP, KGS, MGS/MnDNR, USGS 3DEP, ESA WorldCover
CC BY 4.0, PRISM, POLARIS, Kim et al. 2020 Mendeley package CC BY 4.0).

## Citation

Please cite the paper ([`CITATION.cff`](CITATION.cff)). The archived release will carry a Zenodo DOI, which will be added
here after deposit.
