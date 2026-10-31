# Results (simulated data, seed 0)

## 1. Multimodal fusion (AUC, 95% bootstrap interval)

| Model | AUC | 95% CI |
|---|---|---|
| imaging only | 0.874 | 0.852 to 0.896 |
| wearable only | 0.886 | 0.863 to 0.908 |
| tabular only | 0.836 | 0.810 to 0.862 |
| late fusion (average of 3 models) | 0.902 | 0.882 to 0.923 |
| early fusion (one model, all features) | 0.903 | 0.883 to 0.923 |
| reference: true severity (ceiling) | 0.927 | 0.910 to 0.943 |

## 2. Sparse longitudinal data (mean of 10 seeds)

| Visits per patient | True rate | Estimated rate | Offset correlation |
|---|---|---|---|
| 2 | 0.59 | 0.57 | 0.96 |
| 3 | 0.59 | 0.60 | 0.97 |
| 4 | 0.59 | 0.60 | 0.98 |
| 6 | 0.59 | 0.60 | 0.99 |

## 3. Causal effect (true effect -0.7; 200 simulated studies)

| Estimator | Bias | RMSE |
|---|---|---|
| naive difference | +0.656 | 0.658 |
| regression adjusted (noisy baseline) | +0.150 | 0.153 |
| inverse-probability weighting | +0.152 | 0.157 |
| regression adjusted (true z, oracle) | -0.001 | 0.026 |

## 4. Missing modalities (AUC as the share of missing modalities grows)

| Model | 0% | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| logistic regression, mean-imputed | 0.901 | 0.885 | 0.865 | 0.807 | 0.720 |
| boosted trees, trained on complete data | 0.887 | 0.874 | 0.846 | 0.799 | 0.718 |
| boosted trees, trained with modality dropout | 0.891 | 0.878 | 0.866 | 0.809 | 0.733 |

## 5. Fairness (minority group has a 4x noisier wearable sensor; mean and SD over 10 seeds)

| Training | AUC majority | AUC minority | AUC gap | TPR gap |
|---|---|---|---|---|
| pooled model | 0.892 (0.006) | 0.876 (0.008) | +0.016 (0.010) | +0.013 (0.020) |
| group-reweighted training | 0.889 (0.006) | 0.881 (0.007) | +0.009 (0.009) | +0.002 (0.027) |
| separate model per group | 0.900 (0.006) | 0.881 (0.007) | +0.019 (0.008) | +0.018 (0.025) |
