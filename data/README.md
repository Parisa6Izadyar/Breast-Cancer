# Input data

`Breast_Cancer.csv` is the CSV supplied with the original coursework repository. It contains 4,024 rows and 16 columns, including the recorded vital-status label and follow-up duration. One exact duplicate is removed, leaving 4,023 rows: 3,407 `Alive` and 616 `Dead`.

The public dataset is commonly listed as [Breast Cancer by Reihaneh Namdari](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer). The original download receipt, extraction query and licensing record were not archived with this copy; the CSV is preserved unchanged rather than replacing it with another breast-cancer dataset.

Column names and category strings are stripped of surrounding whitespace. `Reginol Node Positive` and `differentiate` are normalized to `Regional Node Positive` and `Differentiation`. Numeric and Roman-numeral grade values are mapped explicitly; the source's `anaplastic; Grade IV` category is retained as grade 4.

`Survival Months` is excluded from predictors because follow-up duration is unavailable at an initial prediction point. SEER's [survival-time definition](https://seer.cancer.gov/survivaltime/SurvivalTimeCalculation.pdf) describes survival months using diagnosis and last-contact dates. It can describe outcomes, but cannot be treated as a baseline covariate.

No patient identifiers or consistent prediction horizon are available. Identical values across the 14 predictors define split groups: 4,010 profiles, including 13 repeated profiles and one profile associated with both labels. This grouping prevents identical predictor rows crossing folds; it does not establish patient-level, temporal or external validation.
