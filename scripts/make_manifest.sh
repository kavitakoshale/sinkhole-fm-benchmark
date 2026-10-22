#!/usr/bin/env bash
# Rewrite MANIFEST.sha256 (sha256 of every benchmark table, prediction, summary and raster), e.g. after extracting the
# archived out-of-fold predictions into results/ or after rerunning an experiment.
#   bash scripts/make_manifest.sh            check: sha256sum -c MANIFEST.sha256 --quiet
set -euo pipefail
cd "$(dirname "$0")/.."
find data/processed results -type f \( -name '*.parquet' -o -name '*.tif' -o -name '*.csv' -o -name '*.md' \) \
  -exec sha256sum {} \; | LC_ALL=C sort -k2 > MANIFEST.sha256
wc -l MANIFEST.sha256
