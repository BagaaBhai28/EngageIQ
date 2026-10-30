# Model report

Generated measurements are recorded in `models/model_metrics.json`; rerun training to regenerate them with the locked environment.

Training data: UCI Facebook Metrics, 500 posts from 2014. Seven predictors are available before publication: page total likes, type, category, month, weekday, hour and paid promotion. No post-publication outcome is used as an input.

345 posts from months 1–9 form development data. 155 posts from months 10–12 form an untouched test set. Whole months stay together; there is no day-level timestamp, so within-month chronology cannot be recovered. Expanding validation folds train on earlier months and validate on each of months 6–9. Month is one-hot encoded; unseen test months are ignored by the encoder, which limits temporal extrapolation.

Preprocessing uses train-only numerical imputation and scaling plus categorical imputation and one-hot encoding inside a pipeline. Ridge, Decision Tree, Random Forest and Gradient Boosting are compared with raw and log1p targets, plus a median dummy baseline. The lowest mean validation MAE selects the final pipeline before the test set is evaluated. Metrics are on the original interaction scale.

Selected: Ridge (log target). Held-out MAE 135.863, RMSE 252.030, R² -0.011; median baseline MAE 136.710. This small improvement does not establish useful generalization. There is no claim of high accuracy, calibrated uncertainty or causal effects. Viral posts, missing creative information, a single brand, small sample and dataset age limit the results. Feature importance is omitted because the selected transformed linear model does not have tree-style importance.

K-Means uses log1p reactions/comments/shares, then standardization, fixed seed 42 and 10 restarts. Candidate k=2,3,4,5 silhouette scores are 0.554, 0.468, 0.373 and 0.370 on a reproducible 2,000-row sample. k=2 is selected. Cluster IDs have no ordinal meaning; the interface orders profiles by mean interactions. Uploads do not inherit those clusters.
