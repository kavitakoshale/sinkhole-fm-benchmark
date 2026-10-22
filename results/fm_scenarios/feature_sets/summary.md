# Feature-source ablation (Florida, ring negatives)

terrain = DEM only; sat_nolc = DEM + depressions + rainfall; sat_all = + land cover; maps = geology / hydrogeology / soils / distance layers only; subedi5 = Subedi 2019 style 5 factors. Mean fold AUC; spatial5 over 5 folds, temporal is one split.

## Mean ROC-AUC

| protocol   | feature_set   |   n_features |   kumo_tabular |   logreg |   random_forest |   tabpfn |   xgboost |
|:-----------|:--------------|-------------:|---------------:|---------:|----------------:|---------:|----------:|
| spatial5   | all           |           68 |         0.8722 |   0.8112 |          0.8671 |   0.8756 |    0.8706 |
| spatial5   | maps          |           19 |         0.7584 |   0.7155 |          0.7436 |   0.7582 |    0.7333 |
| spatial5   | no_landcover  |           49 |         0.7727 |   0.7256 |          0.7724 |   0.7946 |    0.7719 |
| spatial5   | sat_all       |           49 |         0.8522 |   0.7528 |          0.8462 |   0.8537 |    0.8456 |
| spatial5   | sat_nolc      |           30 |         0.7174 |   0.6521 |          0.7033 |   0.707  |    0.6969 |
| spatial5   | subedi5       |            5 |         0.7087 |   0.6614 |          0.6789 |   0.7043 |    0.6769 |
| spatial5   | terrain       |           21 |         0.6893 |   0.6272 |          0.6633 |   0.6863 |    0.666  |
| temporal   | all           |           68 |         0.8602 |   0.7832 |          0.8408 |   0.8566 |    0.8496 |
| temporal   | maps          |           19 |         0.7642 |   0.6802 |          0.7342 |   0.7562 |    0.7389 |
| temporal   | no_landcover  |           49 |         0.797  |   0.703  |          0.7665 |   0.7896 |    0.7754 |
| temporal   | sat_all       |           49 |         0.8543 |   0.7466 |          0.84   |   0.8522 |    0.8457 |
| temporal   | sat_nolc      |           30 |         0.7757 |   0.6255 |          0.716  |   0.7648 |    0.7248 |
| temporal   | subedi5       |            5 |         0.7406 |   0.6654 |          0.6994 |   0.7362 |    0.6998 |
| temporal   | terrain       |           21 |         0.722  |   0.6142 |          0.6962 |   0.7191 |    0.7062 |

## ΔAUC vs xgboost (paired over fold × rep: mean ± 1.96·SE, share of pairs won)

| protocol   | feature_set   |   n_features | kumo_tabular            | logreg                 | random_forest          | tabpfn                  |
|:-----------|:--------------|-------------:|:------------------------|:-----------------------|:-----------------------|:------------------------|
| spatial5   | all           |           68 | +0.0015 ± 0.0110 (60%)  | -0.0594 ± 0.0216 (0%)  | -0.0035 ± 0.0061 (40%) | +0.0050 ± 0.0053 (80%)  |
| spatial5   | maps          |           19 | +0.0251 ± 0.0213 (100%) | -0.0179 ± 0.0489 (20%) | +0.0102 ± 0.0165 (60%) | +0.0248 ± 0.0074 (100%) |
| spatial5   | no_landcover  |           49 | +0.0008 ± 0.0239 (60%)  | -0.0463 ± 0.0256 (0%)  | +0.0005 ± 0.0090 (60%) | +0.0226 ± 0.0129 (80%)  |
| spatial5   | sat_all       |           49 | +0.0067 ± 0.0119 (60%)  | -0.0928 ± 0.0413 (0%)  | +0.0007 ± 0.0077 (80%) | +0.0082 ± 0.0102 (80%)  |
| spatial5   | sat_nolc      |           30 | +0.0205 ± 0.0204 (80%)  | -0.0448 ± 0.0491 (20%) | +0.0064 ± 0.0116 (80%) | +0.0101 ± 0.0262 (80%)  |
| spatial5   | subedi5       |            5 | +0.0318 ± 0.0077 (100%) | -0.0155 ± 0.0332 (20%) | +0.0020 ± 0.0150 (40%) | +0.0273 ± 0.0117 (100%) |
| spatial5   | terrain       |           21 | +0.0233 ± 0.0045 (100%) | -0.0388 ± 0.0546 (40%) | -0.0027 ± 0.0049 (40%) | +0.0203 ± 0.0058 (100%) |
| temporal   | all           |           68 | +0.0106 ± nan (100%)    | -0.0664 ± nan (0%)     | -0.0088 ± nan (0%)     | +0.0070 ± nan (100%)    |
| temporal   | maps          |           19 | +0.0253 ± nan (100%)    | -0.0587 ± nan (0%)     | -0.0047 ± nan (0%)     | +0.0172 ± nan (100%)    |
| temporal   | no_landcover  |           49 | +0.0216 ± nan (100%)    | -0.0724 ± nan (0%)     | -0.0090 ± nan (0%)     | +0.0142 ± nan (100%)    |
| temporal   | sat_all       |           49 | +0.0085 ± nan (100%)    | -0.0991 ± nan (0%)     | -0.0057 ± nan (0%)     | +0.0065 ± nan (100%)    |
| temporal   | sat_nolc      |           30 | +0.0510 ± nan (100%)    | -0.0993 ± nan (0%)     | -0.0088 ± nan (0%)     | +0.0400 ± nan (100%)    |
| temporal   | subedi5       |            5 | +0.0408 ± nan (100%)    | -0.0344 ± nan (0%)     | -0.0004 ± nan (0%)     | +0.0364 ± nan (100%)    |
| temporal   | terrain       |           21 | +0.0157 ± nan (100%)    | -0.0921 ± nan (0%)     | -0.0100 ± nan (0%)     | +0.0129 ± nan (100%)    |
