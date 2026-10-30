# Validation record

On 30 October 2026, all 25 automated tests passed locally with Python 3.12 and the pinned requirements. The full CLI benchmark fitted all five candidates, using five development CV folds each, and evaluated the same 805 held-out records with their natural class distribution. No fit warnings were recorded.

The source CSV hash is `667fa828fb39d85f6a622dd2067ed3b5221f5460aca6ceec2be5fcfb318d69bd`. After one duplicate is removed, 4,023 rows define 4,010 distinct predictor profiles. Development contains 3,218 rows (2,725 Alive, 493 Dead); test contains 805 rows (682 Alive, 123 Dead). Predictor-profile overlap is zero across the outer split and each inner CV fold.

Tests cover the source profile, label semantics, whitespace and grade handling, predictor exclusion, duplicate column names, invalid numeric values, missing-value handling, grouping of conflicting labels, deterministic folds, training-only preprocessing, unseen categories, saved-pipeline reloads, selection criteria, positive-class metrics, output generation and CLI errors.

Logistic Regression was selected by mean development-CV AP 0.4113501 with fold standard deviation 0.0230548. Test AP was 0.3763789 and ROC AUC 0.7300622. Threshold-dependent Dead recall was 0.1788618. The balanced variant's higher recall is reported separately and does not replace the validation-based selection rule.

CI executes the tests and all 14 notebook cells on the bundled data. The local execution environment restricts Jupyter sockets; actual kernel execution is verified through hosted CI. Full-precision aggregate results and plots are committed under `results/benchmark/`; fitted artifacts and row-level run outputs remain under ignored `results/reproduced/`.
