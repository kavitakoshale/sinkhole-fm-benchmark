#!/usr/bin/env bash
# Run the foundation-model experiments on the same folds as the committed GBDT baselines, then compare.
#
#   bash scripts/run_fm_experiments.sh                     # all FMs, all benchmarks
#   MODELS="tabicl kumo_tabular" bash scripts/run_fm_experiments.sh
#   TASKS="florida" bash scripts/run_fm_experiments.sh
#   SKIP_CHECK=1 ...                                       # skip the pre-flight smoke test
#   FM_PRECISION=bf16 TAG=fm_bf16 bash scripts/run_fm_experiments.sh   # precision variant, own result dirs
#
# Settings (device, weight paths, ...) come from .env / environment, see .env.example.
# Outputs: results/florida_$TAG/, results/florida_${TAG}_no_landcover/, results/kentucky_$TAG/, results/comparison_*.md
# (TAG defaults to fm). Each benchmark writes a log under results/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"tabpfn tabicl kumo_tabular"}
TASKS=${TASKS:-"florida kentucky"}
TAG=${TAG:-fm}
PY=${PY:-python}
mkdir -p results/logs
stamp=$(date +%Y%m%d_%H%M%S)

if [[ -z "${SKIP_CHECK:-}" ]]; then
  # shellcheck disable=SC2086
  $PY scripts/check_env.py --models $MODELS 2>&1 | tee "results/logs/check_env_${stamp}.log"
fi

run() {  # name, command...
  local name=$1; shift
  echo "=== $name: $*"
  local t0=$SECONDS
  "$@" 2>&1 | tee "results/logs/${name}_${stamp}.log"
  echo "=== $name done in $(( (SECONDS - t0) / 60 )) min"
}

if [[ " $TASKS " == *" florida "* ]]; then
  # shellcheck disable=SC2086
  run "florida_${TAG}" $PY baselines/run_florida_baselines.py --models $MODELS --tag "$TAG"
  # shellcheck disable=SC2086
  run "florida_${TAG}_no_landcover" $PY baselines/run_florida_baselines.py --models $MODELS --feature-set no_landcover --tag "${TAG}_no_landcover"
  $PY baselines/compare_models.py florida results/florida "results/florida_${TAG}"
  $PY baselines/compare_models.py florida results/florida_no_landcover "results/florida_${TAG}_no_landcover"
fi

if [[ " $TASKS " == *" kentucky "* ]]; then
  # shellcheck disable=SC2086
  run "kentucky_${TAG}" $PY baselines/run_kentucky_baselines.py --models $MODELS --tag "$TAG"
  $PY baselines/compare_models.py kentucky results/kentucky "results/kentucky_${TAG}"
fi
echo "All done. See results/comparison_*.md"
