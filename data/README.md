# Data catalog

All raw data were downloaded 2026-10-20 from public sources. Each ArcGIS download has a `*.meta.json` next to it (URL, query, count, date, service description).
Re-download scripts: `scripts/fetch_arcgis_layer.py`, `scripts/fetch_arcgis_image.py`, `scripts/fetch_worldcover_overview.py`.

## Rebuilding `data/raw` (~8 GB; only needed to change features)
The benchmark tables in `data/processed/` are committed, so no experiment needs `data/raw`. To rebuild it:
```bash
bash scripts/fetch_raw.sh florida        # FGS SIR + 8 FGS/FDEP layers + favorability raster, 3DEP 1" tiles -> florida_dem.vrt,
                                         # WorldCover x4 overview -> worldcover_florida_x4.vrt, PRISM 2020 normals
bash scripts/fetch_raw.sh kentucky       # KGS LiDAR sinkholes (statewide + bluegrass4), TIGER 2023 counties
python scripts/fetch_kyfromabove.py      # Phase-1 5 ft DEM tiles for the 4 Bluegrass counties + 2,500 ft pad (1,342 tiles, 6.5 GB)
bash scripts/fetch_raw.sh inventories    # IN/KY 2011, Pennsylvania, Virginia, Minnesota inventories
```
Notes: no GDAL CLI is needed (`scripts/build_vrt.py` writes VRTs with rasterio). Behind a TLS-inspecting proxy, export
`REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`. The Mendeley package of Kim et al. (2020) (doi:10.17632/wybccj4wd7.1,
CC BY 4.0) is not fetched by the script (some networks reset the TLS connection): download its per-file zips in a browser into
`raw/florida_mendeley_youn2020/`, unzip the shapefiles into same-named folders and the three FAVA FileGDBs into `gdb/`, then
`.venv-gdal/bin/python scripts/convert_fava_gdb.py` writes `tif/{SAS_DTW,IAS_THK,HEAD_DIFF}.tif` (GDAL binary wheels from
`https://girder.github.io/large_image_wheels` in a separate uv venv; the rasterio wheel lacks the OpenFileGDB raster driver).
Layer ids used: FGS_PUBLIC 0/2/5/6/10, AQUIFER/5 (DRASTIC Floridan), SPRINGS/1, MMP_MINEDUNITS/3 (2022),
KGS `kgs.uky.edu/arcgis/.../KYWater/LiDAR_Sinkholes/MapServer/0`, IGS FeatureServer/118, PA DCNR `topo/Sinkholes/MapServer/0`.

**Reproduction check (2026-10-21).** Running `scripts/florida/01_prepare_points.py` + `02_extract_features.py` on a freshly
downloaded `data/raw` (into a separate directory, not over `data/processed/`) reproduces the committed
`dataset.parquet` exactly: same 8,208 pids, negatives and folds, and 63 of 84 columns bit-identical (all mapped layers, FAVA,
POLARIS soils, land cover, depressions, distances, expert map). The 21 terrain columns differ for 63 points (0.8 %, ≤ 2.3 m in
elevation) because USGS refreshed some 3DEP "current" tiles after the original download. Kentucky: all four county depression
tables reproduce exactly (49,884 / 49,926 rows bit-identical, the rest at county-edge tiles).

## Additional inventories for the multi-region benchmark (downloaded 2026-10-20)
| Path | Source | Records | Notes |
|---|---|---|---|
| `raw/minnesota_mgs/karst_feature_inventory_pts.gpkg` | Minnesota Geological Survey / MnDNR *Karst Feature Inventory Points* (MN Geospatial Commons) | 16,930 (15,642 sinkholes `feature=D`), **2,143 field-checked**, 21 counties | location method (`gcm`: LiDAR / digitized 1:24k / GPS) and date fields; glaciated carbonate karst, SE Minnesota |
| `raw/virginia_dgmr/sinkholes.gpkg` | Virginia DGMR *Karst Features* (`DGMR/KarstFeatures/MapServer/0`) | 51,455 | mostly digitised from topo maps, Shenandoah Valley |
| `raw/pennsylvania_dcnr/karst_features.gpkg` | PA DCNR *Karst Features* (`topo/Sinkholes/MapServer/0`) | 142,138 | 3,619 `Sinkhole`, 138,519 `Surface Depression`; 14 counties inventoried |
| `raw/indiana_kentucky_igs/karst_sinkhole_inventory_2011.gpkg` | IGS (`Karst_Sinkhole_Inventory_IN_KY_2011/FeatureServer/118`) | 154,889 | mapped and modelled sinkholes, southern IN + northern KY |
| Missouri DNR sinkholes (`gis.dnr.mo.gov/.../geology/sinkholes`) | "known and probable" sinkholes | — | **not downloaded**: the server blocks scripted access (Incapsula) |

## Sinkhole inventories (labels)

| Path | Source | Records | Geometry | Notes / license |
|---|---|---|---|---|
| `raw/florida_fgs/subsidence_incident_reports.gpkg` | FDEP / Florida Geological Survey, *Subsidence Incident Reports* (ArcGIS `OpenData/FGS_SUBSIDENCE/MapServer/0`) | 4,457 (1909–2026) | points | Citizen/agency reports, mostly **not field-verified** (`TRUE_SINK`: U 4238, N 110, Y 103). `ACCURACY` ranges from PARCEL/GPS to township (~10 km). Event dates allow temporal splits. Public; credit FDEP/FGS. |
| `raw/florida_mendeley_youn2020/Florida_Subsidence_Incident_Reports/` | Mendeley doi:10.17632/wybccj4wd7.1 | 4,004 | points | 2020 snapshot of the same SIR database (CC BY 4.0). |
| `raw/kentucky_kgs/lidar_sinkholes_statewide.gpkg` | Kentucky Geological Survey, *LiDAR Sinkholes* (`KYWater/LiDAR_Sinkholes/MapServer/0`) | 155,808 | polygons | LiDAR-derived (KGS, Zhu et al. method) with a `Field_Veri` flag. Provisional. |
| `raw/kentucky_kgs/lidar_sinkholes_bluegrass4.gpkg` | same, filtered | 6,941 | polygons | Bourbon 1,291 / Woodford 2,305 / Jessamine 2,463 / Scott 882. Compare Zhu 2020 Table 1: 1,294 / 2,075 / 2,257, and Scott 919. |
| `raw/indiana_kentucky_igs/karst_sinkhole_inventory_2011.gpkg` | Indiana Geological & Water Survey, *Karst Sinkhole Inventory IN/KY 2011* | 154,889 | points | Mapped and modelled sinkholes, southern IN + KY. |
| `raw/pennsylvania_dcnr/karst_features.gpkg` | PA DCNR / PA Geological Survey, *Karst Features* | 142,138 | points | 3,619 `Sinkhole`, 138,519 `Surface Depression`; incomplete coverage, verified sites not flagged. |

## Florida conditioning factors

| Path | Source | Type / res | Used as |
|---|---|---|---|
| `raw/florida_mendeley_youn2020/tif/SAS_DTW.tif` | FGS FAVA (via Mendeley, CC BY 4.0), converted from FileGDB by `scripts/convert_fava_gdb.py` | 30 m, ft | `sas_depth_to_water_ft` (NaN where no surficial aquifer) |
| `…/tif/IAS_THK.tif` | FGS FAVA | 30 m, ft | `ias_thickness_ft` |
| `…/tif/HEAD_DIFF.tif` | FGS FAVA | 90 m, ft | `head_diff_ft` (water table minus Floridan head) |
| `…/rcharg_dec03/` | FGS/WMD Floridan recharge-discharge | polygons | `floridan_recharge_class` |
| `…/ctds_2004/` | FGS closed topographic depressions (163,019 polygons) | polygons | `ctd_*` (inside, distance, count and area fraction in 0.5/1/2 km) |
| `raw/florida_fgs/layers/stratigraphic_geology.gpkg` | FGS stratigraphic geology (`FGS_PUBLIC/6`) | 37 units | `geo_unit`, `geo_lithology`, `geo_age` |
| `…/environmental_geology.gpkg` | FGS environmental geology (`FGS_PUBLIC/0`) | 2,822 polygons | `envgeo_category` |
| `…/sinkhole_types.gpkg` | FGS sinkhole-type areas (Sinclair & Stewart 1985, Areas I–IV) | 8,935 polygons | `sinkhole_type_area` |
| `…/geomorphology_districts.gpkg` | FGS geomorphology districts | 10 polygons | `geomorph_district` |
| `…/drastic_floridan.gpkg` | FDEP DRASTIC for the Floridan aquifer | 11,351 polygons | `drastic_floridan_index`, `drastic_setting` |
| `…/swallets.gpkg`, `…/springs.gpkg`, `…/phosphate_mined_units_2022.gpkg` | FGS / FDEP | points / polygons | `dist_*_m` |
| `…/fgs_sinkhole_favorability_100m.tif` (+ `_RAT.txt`) | FGS *Sinkhole Favorability* (WofE) ImageServer | 100 m, values 1–14 in 4 classes | **expert baseline only, never a feature** |
| `raw/usgs_3dep_1arcsec/` (27 tiles, `florida_dem.vrt`) | USGS 3DEP 1 arc-second DEM | ~30 m | terrain features (elevation, slope, Laplacian, TPI / relief / std / local min and max at 150/300/600 m) |
| `raw/esa_worldcover/` (`worldcover_florida_x4.vrt`) | ESA WorldCover 2021 v200, COG overview ×4 | ~40 m | `lc_class`, class fractions at 250 m / 1 km. **Encodes the urban reporting bias.** |
| `raw/prism/ppt_annual/` | PRISM 1991–2020 normals, annual precipitation | 800 m | `ppt_annual_mm` |
| POLARIS (remote for the 8,208 points; `data/raw/polaris/<var>_<depth>/lat..lon..tif` for the statewide grid: 26 tiles × 4 layers, 1.8 GB, fetched by `scripts/florida/03_predict_raster.py`; both paths give identical values) | Duke POLARIS v1.0 soil properties | 30 m | `soil_ksat_0_5`, `soil_ksat_30_60` (log10 cm/h), `soil_clay_0_5`, `soil_sand_0_5` |

## Kentucky inputs
| Path | Source | Notes |
|---|---|---|
| `raw/kentucky_kyfromabove/dem_phase1/` | KyFromAbove Phase 1 DEM (AWS open data `s3://kyfromabove/elevation/DEM/Phase1/`), 5 ft, EPSG:3089, Z in US ft | tiles for Bourbon, Woodford, Jessamine, Scott: 1,214 (5.6 GB) in the download behind the committed tables; `fetch_kyfromabove.py` with its default 2,500 ft pad fetches 1,342 (6.5 GB), a superset |
| `raw/kentucky_kyfromabove/kyfromabove_phase1_dem_5k_grid.gpkg` | tile index | |
| `raw/kentucky_kyfromabove/bluegrass4_counties.gpkg` | US Census TIGER 2023 counties | |

## Processed benchmark tables
Committed in this repository: `processed/florida/dataset.parquet`, `processed/florida_shared/dataset.parquet`,
`processed/kentucky/depressions_all.parquet`, `processed/minnesota/dataset.parquet` (checksums in `MANIFEST.sha256`).
The other files below (point GeoPackages, cached feature groups, the 1 km Florida grid tables) are intermediates that are
not distributed; the pipelines rebuild them from `data/raw`.

| Path | Rows | Description |
|---|---|---|
| `processed/florida/points.gpkg` | 8,208 | 2,736 SIR positives (accuracy ∈ {PARCEL, GPSSA, QTR3, QTR2}, `TRUE_SINK`≠N, deduplicated at 50 m) + 2,736 `global` negatives + 2,736 `ring` (500–5000 m) negatives. Columns `fold_random`, `fold_spatial` (25 km blocks), `split_temporal` (events ≥ 2010 → test). |
| `processed/florida/features/*.parquet` | 8,208 | cached feature groups |
| `processed/florida/dataset.parquet` | 8,208 × 85 | **Florida benchmark table** (59 numeric + 9 categorical features + metadata) |
| `processed/kentucky/depressions_<County>.gpkg` | | LiDAR depressions with Zhu 2020's 10 morphometric variables and KGS-overlap label |
| `processed/kentucky/depressions_all.parquet` | | **Kentucky benchmark table** |
| `processed/minnesota/points.gpkg` | 39,328 | 13,144 KFI sinkholes (`feature=D`, location method LiDAR / digitized 1:24k / GPS, deduplicated at 50 m, 11 counties with ≥ 10 sinkholes) + 13,144 `global` negatives (uniform over those counties, ≥ 500 m from any karst feature) + 13,040 `ring` negatives (**paired**: one per positive in its 500–5000 m annulus; the Florida ring-union sampling would starve dense Fillmore County). `field_checked` (827 positives) marks the clean-label subset. Same fold columns as Florida; `split_temporal` cut at `geoc_date` ≥ 2014 (the LiDAR additions). EPSG:26915. |
| `processed/minnesota/features_shared.parquet`, `dataset.parquet` | 39,328 × 70 | **Minnesota benchmark table**, shared-schema features only (below) |

## Statewide Florida grid (maps)

`scripts/florida/03_predict_raster.py grid --res 1000` builds a 1 km grid over Florida land (147,084 cells inside
`cntbnd_sep15`) and extracts the B1 feature groups with the functions of `02_extract_features.py`:
`data/processed/florida/grid_1000m.parquet` (features, 53 columns, 38 MB), `grid_1000m_points.parquet` (geometry),
`grid_features_1000m/<group>.parquet` (not committed, ~60 MB; rebuilt in ~3 minutes once the rasters are local).
Cells have no land-cover features (the headline models don't use them). `predict` writes
`results/maps/florida_<model>_1000m.tif` (EPSG:3086) and `results/maps/florida_grid_predictions_1000m.parquet`.

## Shared-schema features (`scripts/shared_features.py`, for every region)
Only layers that exist everywhere, so tables from different states have identical columns (multi-state transfer):
21 terrain features from the 30 m DEM (same definitions as Florida, verified identical on Florida points), 12 `dep_*` features
from a priority-flood fill of a 5 km window around each point (fill depth at the point, inside flag, distance to the nearest
depression cell, count / area fraction / max depth within 0.5, 1, 2 km; threshold 0.5 m; replaces Florida's mapped `ctds`
polygons, with which `dep_inside` correlates r ≈ 0.56), WorldCover class + fractions, and PRISM annual precipitation.
Feature sets in `baselines/run_region_baselines.py`: `terrain` (21), `sat_nolc` (34, headline), `sat_all` (52 + `lc_class`).
Minnesota rasters: `raw/usgs_3dep_1arcsec/minnesota_dem.vrt` (12 tiles), `raw/esa_worldcover/worldcover_minnesota_x4.vrt`.
Window reads over a network file system are slow; `shared_features.localize()` first materializes each VRT as a tiled GeoTIFF
in `$SHARED_FEATURES_CACHE` (default `/tmp/sinkhole_fm_cache`; 0.6 + 0.3 GB for Minnesota).

## Known data caveats
- The SIR inventory is a **reporting** inventory. Density follows population, which is why land-cover/built-up features dominate. Use `--feature-set no_landcover` for geology-only models.
- Positional uncertainty: QTR2 ≈ 400 m, so 30 m terrain features at those points are noisy. PARCEL/GPSSA (n = 1,933) are the precise subset.
- Kentucky labels come from the KGS inventory, which was itself produced by the Zhu et al. (2014) workflow plus visual review. Our depressions come from our own fill (0.5 ft depth, 200 ft² area thresholds), so counts differ somewhat from the paper's 22,884.
