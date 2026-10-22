# Reproducing the results

This guide covers the results of *Tabular foundation models versus tuned gradient boosting for sinkhole susceptibility in
data-poor karst regions* (Banjare, submitted to *Computers & Geosciences*). There are four levels, each building on the
previous one:

| Level | What it shows | Needs | Time |
|---|---|---|---|
| **A. Verify** | the released tables, predictions and headline numbers are consistent, and the baselines reproduce | Python environment, no GPU | minutes |
| **B. Rerun the baselines** | every GBDT / classical result, from the committed folds | CPU cores (many for tuning) | hours |
| **C. Rerun the foundation models** | every TabPFN / TabICL / Kumo result and every study | NVIDIA GPU, ≈ 10 GiB memory | ≈ 12 h on the hardware below |
| **D. Rebuild the benchmark tables** | the tables themselves, from public raw data | ≈ 8 GB download | a day |

Levels A to C use the committed tables in `data/processed/` and their fold columns. Level D is needed only to change features.

## 1. Setup

```bash
git clone https://github.com/<org>/sinkhole-fm-benchmark.git && cd sinkhole-fm-benchmark
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env                                                     # optional; every setting has a default
python scripts/check_env.py --no-smoke                                   # versions, GPU, weight sources, tables
```

`requirements.txt` pins the versions used for the results, and `requirements.lock.txt` is the full environment that
produced them (the default PyPI torch wheel on Linux is CUDA-enabled). The foundation-model weights are downloaded from Hugging Face on first use (Section 4), so no manual step is
needed. The runners set `OMP_NUM_THREADS=4` per process. Run tuning and GBDT grids with `CUDA_VISIBLE_DEVICES=""` so
their workers do not hold idle CUDA contexts on the GPUs.

## 2. Level A: verify the release (no GPU, no weights)

```bash
sha256sum -c MANIFEST.sha256 --quiet && echo "checksums OK"   # every table, prediction, summary and raster
python scripts/reproduce_check.py                             # refits XGBoost, RF, LR on the headline Florida folds
python scripts/verify_headline.py --checks comparisons --tasks florida
python scripts/verify_headline.py                              # all checks: comparisons for 3 regions, seeds, decision
python scripts/positional_accuracy.py && git diff --exit-code results/positional_accuracy.md
python scripts/collect_results.py && git diff --exit-code results/README.md
```

What each check shows, with indicative run times (4 threads per process):

- `reproduce_check.py` (about 40 s) refits three baselines and compares every per-fold metric with
  `results/florida_no_landcover/metrics_long.csv`. Expected: identical on the CPUs used for the paper; on another CPU
  type differences of less than 0.0001 AUC (the check allows 0.001).
- `verify_headline.py --checks comparisons --tasks florida` (about 5 min) recomputes Table 2's Florida rows from the
  released predictions. Pooled AUC, PR-AUC and ΔAUC must be identical. The bootstrap intervals are printed next to the
  committed ones and differ by Monte-Carlo noise of up to 0.006, because `compare_models.py` draws every setting of a
  table from one random stream and only the headline settings are released.
- `verify_headline.py` with no options also runs Kentucky and Minnesota, the five-seed statistics (block × seed
  intervals, DeLong, Holm) and the decision table (Table 4); this takes roughly an hour, and Kentucky's row bootstrap is the
  slowest part. Seeds and decision table must be identical. The script restores every file it touches.
- `positional_accuracy.py` recomputes the supplement's AUC-by-location-accuracy table; the file must not change.
- `collect_results.py` rebuilds `results/README.md` from `results/`; it must not change.

The GitHub Actions workflow (`.github/workflows/check.yml`) runs the checksum, compile, `results/README.md`,
`reproduce_check.py` and Florida `verify_headline.py` checks on every push.

## 3. Level B: rerun the baselines (CPU)

Each command writes the directory named in its comment. The directories already exist in the repository. To compare a
rerun with the committed version, use a new `--tag` and compare its `metrics_long.csv`; as written, the commands overwrite
the committed files.

```bash
export CUDA_VISIBLE_DEVICES=""
# Florida, own features (68), and the headline set without land cover (49)
python baselines/run_florida_baselines.py                                          # results/florida
python baselines/run_florida_baselines.py --feature-set no_landcover --tag no_landcover   # results/florida_no_landcover
# Kentucky depressions (Zhu et al. 2020 protocol + leave-one-county-out + Scott County)
python baselines/run_kentucky_baselines.py                                         # results/kentucky
# shared satellite schema: Minnesota and Florida, three feature sets each
for r in minnesota florida_shared; do
  python baselines/run_region_baselines.py --region $r                             # results/$r            (sat_nolc, headline)
  python baselines/run_region_baselines.py --region $r --feature-set sat_all       # results/${r}_sat_all
  python baselines/run_region_baselines.py --region $r --feature-set terrain       # results/${r}_terrain
done
# nested spatial-CV tuned baselines (100 Optuna trials per outer fold; --jobs = parallel outer folds)
python scripts/run_tuned_baselines.py --task florida --jobs 24                       # results/florida_tuned
python scripts/run_tuned_baselines.py --task florida --feature-set no_landcover --jobs 24   # results/florida_tuned_no_landcover
python scripts/run_tuned_baselines.py --task kentucky --jobs 12                      # results/kentucky_tuned
python scripts/run_tuned_baselines.py --task minnesota --jobs 24                     # results/minnesota_tuned
# Marion County replication of Subedi et al. (2019) (needs data/raw, Section 6)
python baselines/replicate_subedi2019.py                                           # results/florida_subedi2019
```

**Tuning time.** The searches are cached in `results/tuning_cache/`, keyed by model, data and budget. With the cache, the
tuned runs only refit the selected configurations, which takes seconds to minutes; tuned XGBoost on the headline Florida
folds refits in 17 s and gives predictions identical to the released ones. To repeat the searches themselves,
point `TUNE_CACHE_DIR` at an empty directory. That took hours on 128 cores; the budget is set by `TUNE_TRIALS`,
`TUNE_INNER_FOLDS`, `TUNE_MAX_ROUNDS` and the other variables listed in `baselines/tuned.py`.

Baselines are deterministic for a given CPU type: they reproduced the committed predictions bit for bit on the two
machines used for the paper, and within 0.0001 AUC on a different CPU type, where BLAS kernels round differently.

## 4. Level C: rerun the foundation models (GPU)

### Weights

When no local path is set, the adapters (`baselines/fm_models.py`) download the weights from Hugging Face on first use:

| Model | Source | Pinned to | Licence |
|---|---|---|---|
| TabPFN-3.5 | [`Prior-Labs/tabpfn_3_5`](https://huggingface.co/Prior-Labs/tabpfn_3_5), `tabpfn-v3.5-20260909.safetensors` (876 MB) | commit `06bf2ba`, SHA-256 `ece4d67e…0be3` | TABPFN-3.5 License v1.0, non-commercial |
| TabICLv2 | [`jingang/TabICL`](https://huggingface.co/jingang/TabICL), `tabicl-classifier-v2-20260212.ckpt` | `TABICL_CHECKPOINT_VERSION` | see model card |
| Kumo-Tabular large | [`nvidia/Kumo-Tabular`](https://huggingface.co/nvidia/Kumo-Tabular) | revision `v1.0.1` (structured-data-models 0.2.1) | OpenMDW-1.1 |

TabPFN is stored in `<HF_HOME>/tabpfn/<commit>/` (or `TABPFN_MODEL_CACHE_DIR`), and TabICL and Kumo in the standard HF
cache. To use local copies instead, set `TABPFN_MODEL_PATH`, `TABICL_MODEL_PATH` and `KUMO_CHECKPOINT_DIR`. After the first
download, `HF_HUB_OFFLINE=1` runs without network. As a check of the download path, the CPU-only TabPFN rows of Table 5
(`results/fm_cost/cpu.csv`; 500-row five-factor context, `FM_DEVICE=cpu`) reproduce exactly with the downloaded weights.

### Pre-flight

```bash
FM_PRECISION=fp32 python scripts/check_env.py      # prints weight sources, then a 1-minute smoke test per model; must pass
```

### Grids

The headline runs used `FM_PRECISION=fp32` and the full training fold as context (`FM_MAX_CONTEXT=0`, the default).

```bash
export FM_PRECISION=fp32
# Florida (68 and 49 features) and Kentucky; each call also writes a quick comparison table
TAG=fm_fp32     MODELS="tabpfn kumo_tabular" bash scripts/run_fm_experiments.sh   # results/{florida,kentucky}_fm_fp32*, florida_fm_fp32_no_landcover
TAG=tabicl_fp32 MODELS="tabicl"              bash scripts/run_fm_experiments.sh   # results/*_tabicl_fp32*
# shared satellite schema
for r in minnesota florida_shared; do
  for fs in sat_nolc sat_all terrain; do
    python baselines/run_region_baselines.py --region $r --feature-set $fs --models tabpfn tabicl kumo_tabular --tag fm_fp32
  done                                                                           # results/${r}_fm_fp32[_sat_all|_terrain]
done
# precision check, bf16 vs fp32 (Supplementary Table S63)
FM_PRECISION=bf16 TAG=fm_bf16     MODELS="tabpfn kumo_tabular" bash scripts/run_fm_experiments.sh
FM_PRECISION=bf16 TAG=tabicl_bf16 MODELS="tabicl"              bash scripts/run_fm_experiments.sh
```

### Comparison tables

```bash
bash scripts/regen_comparisons.sh       # results/comparison_*_tuned_fp32[_ref_xgbtuned].{md,csv}, comparison_minnesota_*, comparison_florida_shared*
```

This rewrites each table from the predictions present, so run it only after every grid of that table has been regenerated.

### Studies

Each study writes `results/<study>/` (run files plus `summary.md`). The model lists below are what the committed results
contain. Where a study was run in shards (`--folds`, `--suffix`), the `summarize` step merges all `runs*.csv` files, so one
unsharded call gives the same files.

```bash
# M3 seed study: five seeds of the headline family (Table 2, columns 6-8)
for t in florida kentucky minnesota; do
  python scripts/seed_study.py run --task $t                                   # xgboost lightgbm catboost + the three FMs
  python scripts/seed_study.py run --task $t --models xgboost_tuned --jobs 24  # tuned reference (CPU)
done
python scripts/seed_study.py summarize                                         # results/seeds/summary.{md,csv}

# M6 learning curves and label noise (Table 3, Figs. 2-3b); mlp10 is added on Kentucky automatically
for e in learning_curve label_noise; do
  python scripts/fm_scenarios.py $e
  python scripts/fm_scenarios.py $e --models tabicl xgboost_tuned --suffix _m6
done
python scripts/fm_scenarios.py rotation --tasks kentucky                       # rotation test (supplement)
python scripts/fm_scenarios.py calibration                                     # calibration tables (supplement; needs the grids)
python scripts/fm_scenarios.py blend                                           # FM + GBDT averages (supplement; needs the grids)
python scripts/fm_scenarios.py summarize

python scripts/feature_sets.py                                                 # feature-source ablation, 5-68 factors (Table 3, Fig. 3a)

# M5 realistic regime: county few-shot, clean-label test sets, 1:10 / 1:20 prevalence (Table 3, Fig. 3c)
for e in county clean prevalence; do
  python scripts/realistic_benchmark.py $e
  python scripts/realistic_benchmark.py $e --models xgboost_tuned --suffix _tuned
done
python scripts/realistic_benchmark.py summarize

# S1 transfer between states and S2 context selection (Section 4.6)
python scripts/transfer.py --source florida_shared --target minnesota
python scripts/transfer.py --source minnesota --target florida_shared
python scripts/transfer.py --summarize
python scripts/context_selection.py

python scripts/decision_analysis.py                                            # M8, Table 4 and Fig. 4 (from the grids)
FM_PRECISION=bf16 python scripts/fm_cost.py scaling                            # M9, Table 5 (bf16, as reported)
python scripts/fm_cost.py cpu                                                  # Table 5, CPU-only rows
python scripts/fm_sensitivity.py                                               # seed x n_estimators sensitivity (supplement)
python scripts/interpret.py                                                    # M7 factors, Fig. 6
python scripts/florida/03_predict_raster.py grid                               # M7 maps, Fig. 5: 1 km grid (needs data/raw)
python scripts/florida/03_predict_raster.py predict --models tabpfn tabicl kumo_tabular xgboost xgboost_tuned
python scripts/positional_accuracy.py                                          # S6, supplement
python scripts/collect_results.py                                              # results/README.md
```

### Expected agreement

Foundation-model predictions depend on GPU kernels and precision, so a rerun agrees with the committed results within
numerical precision rather than bit for bit. For scale, the bf16 and fp32 grids differ by at most 0.006 AUC on every
setting. Baselines reproduce bit for bit on the same CPU type and within 0.0001 AUC on others. Bootstrap intervals are Monte-Carlo estimates and move slightly whenever the
set of settings in a table changes. The decision metrics treat tied scores by their expected value, so they do not depend on
the CPU.

### Hardware used for the paper

| Part | Hardware | Precision |
|---|---|---|
| Headline FM runs (grids) and the other GPU studies unless listed below | 1 × NVIDIA RTX PRO 6000 Blackwell (96 GB); the largest fit (Kentucky, 34k context) peaked at 9.3 GiB | fp32 for the headline runs (`run_info.json` records each run) |
| Kentucky learning-curve and label-noise shards | 4 × NVIDIA A10 (24 GB), one process per GPU | bf16 |
| Cost table (Table 5) | RTX PRO 6000, shared with other jobs (upper bounds) | bf16 |
| Tuned baselines | 128 CPU cores | – |

A full FM grid took about 15 min, and the complete study about 12 h on this hardware. On a GPU with less memory, set
`FM_MAX_CONTEXT` (for example 10000) and report it, or lower `TABICL_BATCH_SIZE`.

## 5. Where each result in the paper comes from

| Paper item | Produced by | Files |
|---|---|---|
| Table 1, Fig. 1 | Section 6 pipelines | `data/processed/*` |
| Table 2 (headline, seed 0) | grids + `regen_comparisons.sh` | `results/comparison_{florida_no_landcover_tuned_fp32,kentucky_tuned_fp32,minnesota_sat_nolc}*.csv` |
| Table 2 (five seeds) | `seed_study.py` | `results/seeds/summary.csv` |
| Table 3, Figs. 2–3 | `fm_scenarios.py`, `feature_sets.py`, `realistic_benchmark.py` | `results/fm_scenarios/`, `results/realistic/` |
| Table 4, Fig. 4 | `decision_analysis.py` | `results/decision/` |
| Section 4.6 | `transfer.py`, `context_selection.py` | `results/transfer/`, `results/context_selection/` |
| Table 5 | `fm_cost.py` | `results/fm_cost/` |
| Fig. 5 | `03_predict_raster.py` | `results/maps/*.tif` |
| Fig. 6 | `interpret.py` | `results/interpret/` |
| Supplement | all of the above | `results/README.md` (every table), `results/tuning_cache/`, `results/fm_sensitivity/`, `results/positional_accuracy.md` |

## 6. Level D: rebuild the benchmark tables from raw data

Only needed to change features. The committed tables are the reference, and rebuilding them changes nothing downstream
unless the inputs changed (see below).

```bash
bash scripts/fetch_raw.sh florida && bash scripts/fetch_raw.sh kentucky && bash scripts/fetch_raw.sh inventories
python scripts/fetch_kyfromabove.py                  # KyFromAbove 5 ft DEM tiles for the four Bluegrass counties
# Kim et al. (2020) Mendeley package: manual download, then scripts/convert_fava_gdb.py (see data/README.md)

python scripts/florida/01_prepare_points.py          # positives, global + ring pseudo-absences, fold columns
python scripts/florida/02_extract_features.py        # -> data/processed/florida/dataset.parquet
for c in Bourbon Woodford Jessamine Scott; do python scripts/kentucky/01_extract_depressions.py --county $c; done
python baselines/run_kentucky_baselines.py           # assembles depressions_all.parquet from the county files, then runs
python scripts/minnesota/01_prepare_points.py
python scripts/shared_features.py data/processed/minnesota/points.gpkg data/raw/usgs_3dep_1arcsec/minnesota_dem.vrt \
       data/raw/esa_worldcover/worldcover_minnesota_x4.vrt data/processed/minnesota/features_shared.parquet
python scripts/minnesota/02_build_dataset.py         # -> data/processed/minnesota/dataset.parquet
```

`data/processed/florida_shared/dataset.parquet` (the Florida points on the shared schema) comes from the same
`shared_features.py` extraction on `data/processed/florida/points.gpkg` with the Florida rasters, assembled like the
Minnesota table. There is no separate assembly script for it.

**Expected differences.** A rebuild in October 2026 reproduced the Florida table's pids, pseudo-absences and folds
exactly, and 63 of its 84 columns bit for bit. USGS has since refreshed some 3DEP tiles, so the 21 terrain columns differ for
63 points (0.8 %, at most 2.3 m in elevation). The Kentucky depression tables reproduced exactly, except for 42 of 49,926
rows at county-edge tiles. Because the folds and pseudo-absences are written into the tables, a rebuilt table is a new
benchmark. Give its results a new tag and rerun every model on it, baselines included, rather than mixing it with the
committed results.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `check_env.py` reports a missing weight path | unset the variable to use the Hugging Face download, or fix the path |
| no network on the compute node | download once elsewhere (`check_env.py` smoke test), copy `HF_HOME`, set `HF_HUB_OFFLINE=1` |
| GPU out of memory | `FM_PRECISION=bf16`, `FM_MAX_CONTEXT=10000`, `TABICL_BATCH_SIZE=2`; check `nvidia-smi` for other jobs |
| very slow baseline runs | keep `OMP_NUM_THREADS` ≤ 4 per process; oversubscribed BLAS threads were 50–90 × slower |
| idle 550 MB CUDA contexts from tuning or GBDT jobs | run them with `CUDA_VISIBLE_DEVICES=""` |
| a script stops with "no out-of-fold predictions" | run the grids it reads first (Section 4); it refuses to overwrite committed summaries |
