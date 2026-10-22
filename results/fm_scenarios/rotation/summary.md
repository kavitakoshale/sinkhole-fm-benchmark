# Rotation test (Kentucky loco)

raw = original features; quantile = per-feature quantile→normal (monotone, trees are invariant); rotated = quantile + random orthogonal rotation of all 10 features (3 rotations). A large AUC loss under rotation means the model relies on axis-aligned structure.

## Mean ROC-AUC

| task     | condition   |   catboost |   kumo_tabular |   lightgbm |   logreg |   mlp10 |   random_forest |   tabpfn |   xgboost |
|:---------|:------------|-----------:|---------------:|-----------:|---------:|--------:|----------------:|---------:|----------:|
| kentucky | quantile    |     0.9647 |         0.9699 |     0.964  |   0.9539 |  0.9588 |          0.9582 |   0.9706 |    0.9639 |
| kentucky | raw         |     0.9648 |         0.9702 |     0.9638 |   0.931  |  0.9565 |          0.9581 |   0.9707 |    0.9638 |
| kentucky | rotated     |     0.9637 |         0.9707 |     0.9637 |   0.954  |  0.9606 |          0.96   |   0.9709 |    0.9636 |

## ΔAUC vs xgboost (paired over fold × rep: mean ± 1.96·SE, share of pairs won)

| task     | condition   | catboost               | kumo_tabular            | lightgbm               | logreg                | mlp10                  | random_forest         | tabpfn                  |
|:---------|:------------|:-----------------------|:------------------------|:-----------------------|:----------------------|:-----------------------|:----------------------|:------------------------|
| kentucky | quantile    | +0.0008 ± 0.0016 (67%) | +0.0060 ± 0.0022 (100%) | +0.0000 ± 0.0004 (33%) | -0.0100 ± 0.0045 (0%) | -0.0051 ± 0.0027 (0%)  | -0.0058 ± 0.0009 (0%) | +0.0067 ± 0.0020 (100%) |
| kentucky | raw         | +0.0010 ± 0.0013 (67%) | +0.0064 ± 0.0020 (100%) | +0.0000 ± 0.0004 (33%) | -0.0328 ± 0.0028 (0%) | -0.0073 ± 0.0106 (33%) | -0.0057 ± 0.0009 (0%) | +0.0068 ± 0.0020 (100%) |
| kentucky | rotated     | +0.0001 ± 0.0005 (67%) | +0.0070 ± 0.0010 (100%) | +0.0000 ± 0.0002 (56%) | -0.0096 ± 0.0022 (0%) | -0.0031 ± 0.0018 (11%) | -0.0037 ± 0.0005 (0%) | +0.0072 ± 0.0012 (100%) |

## AUC lost to rotation (quantile − rotated)

| model         |   auc_lost_to_rotation |
|:--------------|-----------------------:|
| random_forest |                -0.0018 |
| mlp10         |                -0.0018 |
| kumo_tabular  |                -0.0007 |
| tabpfn        |                -0.0003 |
| logreg        |                -0.0001 |
| lightgbm      |                 0.0003 |
| xgboost       |                 0.0003 |
| catboost      |                 0.001  |
