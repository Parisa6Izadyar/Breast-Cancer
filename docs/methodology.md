# Evaluation protocol

## Outcome and features

The task is retrospective classification of the recorded `Alive`/`Dead` label in a cohort of people already diagnosed with breast cancer. `Dead` is encoded as 1 and is the positive class in average precision, recall, F1 and ROC AUC. `Alive` is 0. These are not cancer/no-cancer labels.

Four numeric predictors and ten categorical predictors are retained. `Survival Months` is excluded. Follow-up differs between records, so the classifier is not a time-to-event model or a fixed-year mortality-risk estimator.

## Partitioning

Exact duplicate records are removed before partitioning. A deterministic hash of the 14 pre-outcome predictor values defines groups, independent of Status or follow-up duration. The first fold of five shuffled `StratifiedGroupKFold` folds with seed 42 forms the held-out test partition. The remaining rows form the development partition. Five further stratified group folds within development estimate candidate performance.

Identical predictor profiles remain in one partition, including profiles with conflicting recorded labels. The data lacks patient IDs, dates and collection-site metadata; grouped profiles are a conservative overlap control rather than evidence of independent patients or external validity.

## Preprocessing and models

Every candidate is a pipeline containing schema normalization, median imputation and standardization for numeric columns, constant missing-category imputation, one-hot encoding and a classifier. All learned statistics and category vocabularies fit the training portion of each CV fold. Unseen category values are handled by the fitted encoder. At inference the saved pipeline applies the same schema and transformations.

The predefined candidates are a majority/prior baseline, Logistic Regression with and without balanced class weights, and Random Forest with and without balanced-subsample weights. Logistic Regression uses C=1, L2 regularization and at most 2,000 iterations. Forests use 200 trees, minimum leaf size 3, default square-root feature sampling and seed 42. There is no resampling and no hyperparameter search.

## Selection and reporting

Highest mean development-CV average precision selects the model; alphabetical name resolves exact ties. CV standard deviation is variation across folds, not a confidence interval. Candidate settings and selection do not use test scores. Each candidate is then fitted on development data and scored on the same held-out test partition. The saved selected model is not refitted on the test set.

Average precision summarizes precision-recall ranking for the minority `Dead` label. The prior baseline's test AP equals test prevalence. ROC AUC uses positive-class probabilities, rather than hard predictions. Classification metrics use a fixed threshold of 0.5: accuracy, balanced accuracy, macro F1, Dead precision/recall/F1 and explicitly labelled confusion-matrix counts. The threshold is not tuned on the test set. Brier score is also reported; no probability-calibration claim is made.

Class weighting changes the decision tradeoff and does not guarantee improved ranking or calibrated risk scores. A model can have high overall accuracy and low Dead recall, which is why accuracy is not the selection criterion.

## References

- [scikit-learn on data leakage and pipelines](https://scikit-learn.org/1.8/common_pitfalls.html)
- [StratifiedGroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Average precision](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html)
- [imbalanced-learn on resampling before splitting](https://imbalanced-learn.org/stable/common_pitfalls.html)
- [SEER survival-time definitions](https://seer.cancer.gov/survivaltime/SurvivalTimeCalculation.pdf)
