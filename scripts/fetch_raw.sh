#!/usr/bin/env bash
# Rebuild data/raw from public sources (everything except the Mendeley/Youn 2020 package, which is downloaded by hand
# into data/raw/florida_mendeley_youn2020/, see data/README.md).
#
#   bash scripts/fetch_raw.sh florida      # FGS layers, SIR, favorability raster, 3DEP DEM, WorldCover, PRISM
#   bash scripts/fetch_raw.sh kentucky     # KGS LiDAR sinkholes, TIGER counties (DEM tiles: fetch_kyfromabove.py)
#   bash scripts/fetch_raw.sh inventories  # IN/KY 2011, Pennsylvania, Minnesota, Virginia inventories
#   bash scripts/fetch_raw.sh all
# Idempotent: existing outputs are skipped. Every download gets a .meta.json next to it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
RAW=data/raw
# some hosts sit behind a TLS-inspecting proxy here: trust the system CA store from python too
[[ -f /etc/ssl/certs/ca-certificates.crt ]] && export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt} SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
FDEP=https://ca.dep.state.fl.us/arcgis/rest/services/OpenData
mkdir -p results/logs

layer() {  # url out.gpkg [where]
  if [[ -s "$2" ]]; then echo "have $2"; else $PY scripts/fetch_arcgis_layer.py "$1" "$2" ${3:+--where "$3"}; fi
}

florida() {
  mkdir -p $RAW/florida_fgs/layers $RAW/usgs_3dep_1arcsec $RAW/esa_worldcover $RAW/prism/ppt_annual
  layer $FDEP/FGS_SUBSIDENCE/MapServer/0  $RAW/florida_fgs/subsidence_incident_reports.gpkg
  layer $FDEP/FGS_PUBLIC/MapServer/6      $RAW/florida_fgs/layers/stratigraphic_geology.gpkg
  layer $FDEP/FGS_PUBLIC/MapServer/0      $RAW/florida_fgs/layers/environmental_geology.gpkg
  layer $FDEP/FGS_PUBLIC/MapServer/5      $RAW/florida_fgs/layers/sinkhole_types.gpkg
  layer $FDEP/FGS_PUBLIC/MapServer/10     $RAW/florida_fgs/layers/geomorphology_districts.gpkg
  layer $FDEP/FGS_PUBLIC/MapServer/2      $RAW/florida_fgs/layers/swallets.gpkg
  layer $FDEP/AQUIFER/MapServer/5         $RAW/florida_fgs/layers/drastic_floridan.gpkg
  layer $FDEP/SPRINGS/MapServer/1         $RAW/florida_fgs/layers/springs.gpkg
  layer $FDEP/MMP_MINEDUNITS/MapServer/3  $RAW/florida_fgs/layers/phosphate_mined_units_2022.gpkg
  if [[ ! -s $RAW/florida_fgs/layers/fgs_sinkhole_favorability_100m.tif ]]; then
    $PY scripts/fetch_arcgis_image.py $FDEP/FGS_FAV_SINK_FORMATION/ImageServer $RAW/florida_fgs/layers/fgs_sinkhole_favorability_100m.tif
    curl -s "$FDEP/FGS_FAV_SINK_FORMATION/ImageServer/rasterAttributeTable?f=json" > $RAW/florida_fgs/layers/fgs_sinkhole_favorability_100m_RAT.json || true
  fi
  [[ -s $RAW/usgs_3dep_1arcsec/florida_dem.vrt ]] || $PY scripts/fetch_3dep_tiles.py $RAW/usgs_3dep_1arcsec --bbox -87.7 24.4 -79.9 31.1 --vrt florida_dem.vrt
  $PY scripts/fetch_worldcover_overview.py $RAW/esa_worldcover --bbox -88 24 -79 31.5 --factor 4
  [[ -s $RAW/esa_worldcover/worldcover_florida_x4.vrt ]] || $PY scripts/build_vrt.py $RAW/esa_worldcover/worldcover_florida_x4.vrt --glob "$RAW/esa_worldcover/worldcover2021_*_x4.tif"
  if [[ ! -s $RAW/prism/ppt_annual/prism_ppt_us_30s_2020_avg_30y.tif ]]; then
    curl -sL -o $RAW/prism/ppt_annual/prism_ppt_us_30s_2020_avg_30y.zip https://ftp.prism.oregonstate.edu/normals/us/800m/ppt/monthly/prism_ppt_us_30s_2020_avg_30y.zip
    (cd $RAW/prism/ppt_annual && unzip -o -q prism_ppt_us_30s_2020_avg_30y.zip && rm prism_ppt_us_30s_2020_avg_30y.zip)
    echo '{"source":"https://ftp.prism.oregonstate.edu/normals/us/800m/ppt/monthly/prism_ppt_us_30s_2020_avg_30y.zip","note":"PRISM 1991-2020 annual precipitation normal, 30 arc-sec","downloaded":"'"$(date +%F)"'"}' > $RAW/prism/ppt_annual/meta.json
  fi
  echo "florida raw done (Mendeley package still needed: $RAW/florida_mendeley_youn2020/)"
}

kentucky() {
  mkdir -p $RAW/kentucky_kgs $RAW/kentucky_kyfromabove
  layer https://kgs.uky.edu/arcgis/rest/services/KYWater/LiDAR_Sinkholes/MapServer/0 $RAW/kentucky_kgs/lidar_sinkholes_statewide.gpkg
  if [[ ! -s $RAW/kentucky_kyfromabove/bluegrass4_counties.gpkg ]]; then
    curl -sL -o /tmp/tl_2023_us_county.zip https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip
    $PY - <<'EOF'
import geopandas as gpd
c = gpd.read_file("/tmp/tl_2023_us_county.zip")
c.to_file("data/raw/tiger2023_us_county.gpkg", driver="GPKG")
ky = c[(c.STATEFP == "21") & c.NAME.isin(["Bourbon", "Woodford", "Jessamine", "Scott"])]
ky.to_file("data/raw/kentucky_kyfromabove/bluegrass4_counties.gpkg", driver="GPKG")
print("counties:", ky.NAME.tolist())
EOF
    rm -f /tmp/tl_2023_us_county.zip
  fi
  $PY - <<'EOF'
import geopandas as gpd, os
src, out = "data/raw/kentucky_kgs/lidar_sinkholes_statewide.gpkg", "data/raw/kentucky_kgs/lidar_sinkholes_bluegrass4.gpkg"
if not os.path.exists(out):
    g = gpd.read_file(src)
    col = next(c for c in g.columns if c.lower() == "county")
    g[g[col].str.title().isin(["Bourbon", "Woodford", "Jessamine", "Scott"])].to_file(out, driver="GPKG")
    print(out, g[col].str.title().isin(["Bourbon", "Woodford", "Jessamine", "Scott"]).sum())
EOF
  echo "kentucky vectors done; DEM tiles: python scripts/fetch_kyfromabove.py"
}

inventories() {
  mkdir -p $RAW/indiana_kentucky_igs $RAW/pennsylvania_dcnr $RAW/minnesota_mgs $RAW/virginia_dgmr
  layer https://portal.igs.indiana.edu/arcgis/rest/services/Karst_Sinkhole_Inventory_IN_KY_2011/FeatureServer/118 $RAW/indiana_kentucky_igs/karst_sinkhole_inventory_2011.gpkg
  layer https://maps.dcnr.pa.gov/agsprod/rest/services/topo/Sinkholes/MapServer/0 $RAW/pennsylvania_dcnr/karst_features.gpkg
  layer https://energy.virginia.gov/gis/rest/services/DGMR/KarstFeatures/MapServer/0 $RAW/virginia_dgmr/sinkholes.gpkg
  if [[ ! -s $RAW/minnesota_mgs/karst_feature_inventory_pts.gpkg ]]; then
    curl -sL -o $RAW/minnesota_mgs/karst_feature_inventory_pts.gpkg https://operations.gis.data.mn.gov/api/publicdownload/download/787/geos_karst_feature_inventory_pts.gpkg
    echo '{"source":"https://gis.data.mn.gov/datasets/minnesota::karst-feature-inventory-points-2","file":"https://operations.gis.data.mn.gov/api/publicdownload/download/787/geos_karst_feature_inventory_pts.gpkg","publisher":"Minnesota Geological Survey / MnDNR","downloaded":"'"$(date +%F)"'"}' > $RAW/minnesota_mgs/karst_feature_inventory_pts.gpkg.meta.json
  fi
  echo "inventories done"
}

case "${1:-all}" in
  florida) florida ;;
  kentucky) kentucky ;;
  inventories) inventories ;;
  all) florida; kentucky; inventories ;;
  *) echo "usage: $0 {florida|kentucky|inventories|all}"; exit 1 ;;
esac
