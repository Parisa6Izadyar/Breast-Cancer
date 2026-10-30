# Technical review

| Earlier behavior | Current implementation |
| --- | --- |
| Google Drive mounts and an unrelated Excel load were required | The bundled CSV is loaded through a shared local path |
| Random oversampling occurred before the train/test split | Grouped partitions are created first; predefined class-weight variants handle imbalance |
| Follow-up duration was used to classify recorded vital status | `Survival Months` is excluded from the feature schema |
| Imputation and scaling could fit the full dataset | Each CV training fold fits its own preprocessing pipeline |
| Grade mappings expected Roman numerals while the CSV used numeric strings | Grade values are normalized explicitly and preserved |
| Whitespace, a misspelled marital category and omitted T4 values lost category information | Shared normalization and fitted one-hot encoding preserve those categories |
| Manual concatenation after duplicate removal misaligned row indices | Column transformations preserve row alignment |
| One age summary read the tumor-size column; some outcome legends were reversed | Descriptive tables and explicit Alive/Dead labels use the development data |
| Distribution normality was confused with feature scaling | Standardization is a fitted transformation; no normal-distribution requirement is claimed |
| Saved outputs depended on notebook execution order | CLI and notebook call the same implementation and save versioned run metadata |

The earlier 96.8% Random Forest accuracy used a resampled test distribution and follow-up information. It is not an independent performance estimate and is not used in the current benchmark or CV. New scores use the natural class distribution and a different feature/partition protocol, so they are not presented as a like-for-like accuracy improvement.
