# Seed / n_estimators sensitivity: Florida no_landcover x ring

seeds=[0, 1, 2], FM_PRECISION=fp32

| protocol   | model        | n_est   |   pooled_auc |   mean_fold_auc |   seconds |   pooled_auc_sd_over_seeds |   mean_fold_auc_sd_over_seeds |
|:-----------|:-------------|:--------|-------------:|----------------:|----------:|---------------------------:|------------------------------:|
| spatial5   | xgboost      | -       |       0.7674 |          0.7719 |    7.0667 |                     0.0026 |                        0.0029 |
| spatial5   | lightgbm     | -       |       0.7677 |          0.7714 |    5.9667 |                     0.0012 |                        0.0008 |
| spatial5   | tabpfn       | 8       |       0.7817 |          0.7903 |   12.5333 |                     0.0032 |                        0.0038 |
| spatial5   | tabpfn       | 32      |       0.7812 |          0.7901 |   30.9333 |                     0.0007 |                        0.0007 |
| spatial5   | kumo_tabular | 8       |       0.7667 |          0.7738 |   52.6333 |                     0.0011 |                        0.001  |
| spatial5   | kumo_tabular | 32      |       0.7665 |          0.7736 |   88.3    |                     0.0007 |                        0.0003 |
| temporal   | xgboost      | -       |       0.7773 |          0.7773 |    1.4    |                     0.0023 |                        0.0023 |
| temporal   | lightgbm     | -       |       0.7746 |          0.7746 |    1.1    |                     0.0033 |                        0.0033 |
| temporal   | tabpfn       | 8       |       0.7885 |          0.7885 |    1.8    |                     0.0015 |                        0.0015 |
| temporal   | tabpfn       | 32      |       0.7882 |          0.7882 |    5.7    |                     0.0007 |                        0.0007 |
| temporal   | kumo_tabular | 8       |       0.7975 |          0.7975 |   10.4    |                     0.001  |                        0.001  |
| temporal   | kumo_tabular | 32      |       0.7975 |          0.7975 |   17.3    |                     0.0003 |                        0.0003 |
