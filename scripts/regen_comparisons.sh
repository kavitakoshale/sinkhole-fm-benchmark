#!/bin/bash
# Regenerate every paired-bootstrap comparison table (baselines + tuned baselines + all three FMs), skipping result
# directories that do not exist yet. Run after any grid / tuned run finishes:
#   bash scripts/regen_comparisons.sh            (all)      |   bash scripts/regen_comparisons.sh florida
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export CUDA_VISIBLE_DEVICES=""
only=$1

cmp() {  # cmp <task> <out-name> <ref> <dirs...>
  local task=$1 out=$2 ref=$3; shift 3
  local dirs=()
  for d in "$@"; do [ -d "$d" ] && dirs+=("$d"); done
  [ ${#dirs[@]} -ge 2 ] || { echo "skip $out (need >= 2 dirs)"; return; }
  echo "== $out  (${dirs[*]})"
  $PY -W ignore baselines/compare_models.py "$task" "${dirs[@]}" --ref "$ref" --out "results/$out.md" > /dev/null
}

if [ -z "$only" ] || [ "$only" = florida ]; then
  for fs in "" _no_landcover; do
    base=results/florida$fs
    cmp florida "comparison_florida${fs}_tuned_fp32"     xgboost       $base results/florida_tuned$fs results/florida_fm_fp32$fs results/florida_tabicl_fp32$fs
    cmp florida "comparison_florida${fs}_tuned_fp32_ref_xgbtuned" xgboost_tuned $base results/florida_tuned$fs results/florida_fm_fp32$fs results/florida_tabicl_fp32$fs
  done
  for fs in "" _sat_all _terrain; do
    cmp florida_shared "comparison_florida_shared${fs}" xgboost results/florida_shared$fs results/florida_shared_fm_fp32$fs
  done
fi
if [ -z "$only" ] || [ "$only" = kentucky ]; then
  cmp kentucky comparison_kentucky_tuned_fp32 xgboost results/kentucky results/kentucky_tuned results/kentucky_fm_fp32 results/kentucky_tabicl_fp32
  cmp kentucky comparison_kentucky_tuned_fp32_ref_xgbtuned xgboost_tuned results/kentucky results/kentucky_tuned results/kentucky_fm_fp32 results/kentucky_tabicl_fp32
fi
if [ -z "$only" ] || [ "$only" = minnesota ]; then
  cmp minnesota comparison_minnesota_sat_nolc xgboost results/minnesota results/minnesota_tuned results/minnesota_fm_fp32
  cmp minnesota comparison_minnesota_sat_nolc_ref_xgbtuned xgboost_tuned results/minnesota results/minnesota_tuned results/minnesota_fm_fp32
  cmp minnesota comparison_minnesota_sat_all xgboost results/minnesota_sat_all results/minnesota_fm_fp32_sat_all
  cmp minnesota comparison_minnesota_terrain xgboost results/minnesota_terrain results/minnesota_fm_fp32_terrain
fi
