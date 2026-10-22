# Factor analysis (Florida no_landcover × ring × spatial5)

Permutation importance = AUC drop on the test fold when the feature is shuffled (2 shuffles × 5 folds, mean). Group importance shuffles all features of a source together (same row permutation). Partial dependence = mean prediction on a test-fold sample with the feature fixed at 11 quantiles.

## Feature-group importance

| group       |   n_features |   kumo_tabular |   tabicl |   tabpfn |   xgboost |   xgboost_tuned |
|:------------|-------------:|---------------:|---------:|---------:|----------:|----------------:|
| terrain     |           21 |         0.1313 |   0.1164 |   0.1193 |    0.1013 |          0.0968 |
| proximity   |            3 |         0.0474 |   0.0444 |   0.0453 |    0.0221 |          0.0186 |
| hydro       |            6 |         0.0289 |   0.0319 |   0.0447 |    0.0359 |          0.0326 |
| soil        |            4 |         0.0252 |   0.0265 |   0.0193 |    0.0462 |          0.0577 |
| climate     |            1 |         0.0224 |   0.0138 |   0.0192 |    0.0052 |          0.0037 |
| depressions |            8 |         0.0163 |   0.0149 |   0.0181 |    0.0087 |          0.0073 |
| geology     |            6 |        -0.0005 |   0.0084 |   0.0107 |    0.0067 |          0.0045 |

## Top 25 features

| feature                |   kumo_tabular |   tabicl |   tabpfn |   xgboost |   xgboost_tuned |
|:-----------------------|---------------:|---------:|---------:|----------:|----------------:|
| elev_m                 |         0.0285 |   0.0466 |   0.0455 |    0.0215 |          0.0147 |
| dist_phosphate_mine_m  |         0.0317 |   0.0289 |   0.0308 |    0.0086 |          0.0074 |
| ppt_annual_mm          |         0.0203 |   0.0146 |   0.0211 |    0.005  |          0.0037 |
| dist_swallet_m         |         0.0236 |   0.0126 |   0.0142 |    0.0062 |          0.0056 |
| slope_mean_600m        |         0.011  |   0.0122 |   0.0119 |    0.0086 |          0.0075 |
| head_diff_ft           |         0.0047 |   0.013  |   0.0121 |    0.0117 |          0.0074 |
| height_above_min_600m  |         0.0098 |   0.0082 |   0.008  |    0.0076 |          0.0056 |
| drastic_floridan_index |         0.0053 |   0.0063 |   0.0069 |    0.0092 |          0.0099 |
| soil_ksat_0_5          |         0.0055 |   0.0029 |   0.0047 |    0.0108 |          0.0131 |
| soil_sand_0_5          |         0.005  |   0.0075 |   0.003  |    0.0092 |          0.0122 |
| dist_spring_m          |         0.0141 |   0.0038 |   0.0095 |    0.0042 |          0.0041 |
| elev_std_300m          |         0.0099 |   0.0157 |   0.0075 |    0.0013 |          0.0007 |
| height_above_min_300m  |         0.0025 |   0.0081 |   0.0039 |    0.007  |          0.011  |
| slope_mean_300m        |         0.0061 |   0.0123 |   0.005  |    0.0031 |          0.001  |
| drastic_setting        |         0.0013 |   0.0089 |   0.0048 |    0.0037 |          0.0026 |
| relief_150m            |         0.007  |   0.0084 |   0.0041 |    0.0017 |         -0.0004 |
| elev_std_600m          |         0.007  |   0.0076 |   0.0033 |    0.0003 |          0.0006 |
| ias_thickness_ft       |        -0.0002 |   0.0026 |   0.0038 |    0.0069 |          0.0057 |
| soil_clay_0_5          |         0.0045 |   0.0047 |   0.0031 |    0.0037 |          0.0011 |
| tpi_300m               |         0.0073 |   0.0041 |   0.0034 |    0.0013 |          0.0005 |
| laplacian_curv         |         0.0074 |   0.0007 |   0.0012 |    0.0044 |          0.0018 |
| ctd_area_frac_2000m    |         0.003  |   0.0055 |   0.0026 |    0.003  |          0.0013 |
| slope_mean_150m        |         0.0054 |   0.0057 |   0.0035 |   -0.0004 |          0.0009 |
| relief_300m            |         0.0028 |   0.0113 |   0.0023 |   -0.002  |         -0.0004 |
| elev_std_150m          |         0.0048 |   0.0044 |   0.0042 |    0.0005 |         -0.0003 |

## Rank correlation of feature importance between models (Spearman)

| model         |   kumo_tabular |   tabicl |   tabpfn |   xgboost |   xgboost_tuned |
|:--------------|---------------:|---------:|---------:|----------:|----------------:|
| kumo_tabular  |          1     |    0.699 |    0.776 |     0.526 |           0.501 |
| tabicl        |          0.699 |    1     |    0.801 |     0.512 |           0.518 |
| tabpfn        |          0.776 |    0.801 |    1     |     0.566 |           0.548 |
| xgboost       |          0.526 |    0.512 |    0.566 |     1     |           0.914 |
| xgboost_tuned |          0.501 |    0.518 |    0.548 |     0.914 |           1     |

Figures: `results/interpret/importance.png`, `results/interpret/pdp.png` (top features: elev_m, dist_phosphate_mine_m, ppt_annual_mm, dist_swallet_m, slope_mean_600m, head_diff_ft).
