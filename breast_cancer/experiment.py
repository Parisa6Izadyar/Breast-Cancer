import importlib.metadata
import json
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import __version__
from .data import FEATURES, TARGET, grouped_splits, predictor_groups
from .evaluate import metrics, plot_results
from .models import candidates


def choose_model(cv):
    return min(cv, key=lambda name: (-cv[name]["mean_average_precision"], name))


def run_experiment(frame, quality, output: Path, seed=42, folds=5):
    development_ids, test_ids = grouped_splits(frame, folds, seed)[0]
    development = frame.iloc[development_ids].copy()
    test = frame.iloc[test_ids].copy()
    cv_splits = grouped_splits(development, folds, seed)
    estimators = candidates(seed)
    cv = {}
    for name, template in estimators.items():
        print(f"Training CV: {name}", flush=True)
        scores, fit_warnings = [], set()
        for train_ids, validation_ids in cv_splits:
            train, validation = development.iloc[train_ids], development.iloc[validation_ids]
            model = clone(template)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train[FEATURES], train[TARGET])
            fit_warnings.update(str(w.message) for w in caught)
            probability = model.predict_proba(validation[FEATURES])[:, list(model.classes_).index(1)]
            scores.append(metrics(validation[TARGET], probability))
        ap = [s["average_precision"] for s in scores]
        cv[name] = {"fold_metrics": scores, "mean_average_precision": float(np.mean(ap)),
                    "std_average_precision": float(np.std(ap)), "warnings": sorted(fit_warnings)}
    selected = choose_model(cv)
    results, fitted, predictions = {}, {}, pd.DataFrame({"source_row": test.index, "recorded_dead": test[TARGET].to_numpy()})
    for name, template in estimators.items():
        model = clone(template)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(development[FEATURES], development[TARGET])
        probability = model.predict_proba(test[FEATURES])[:, list(model.classes_).index(1)]
        results[name] = {**metrics(test[TARGET], probability), "warnings": sorted({str(w.message) for w in caught})}
        fitted[name] = model
        predictions[name] = probability
        print(f"Test {name}: AP {results[name]['average_precision']:.4f}, ROC AUC {results[name]['roc_auc']:.4f}, Dead recall {results[name]['dead_recall']:.4f}", flush=True)
    partitions = {}
    groups = predictor_groups(frame)
    for name, part in [("development", development), ("test", test)]:
        partitions[name] = {"rows": len(part), "alive": int(part[TARGET].eq(0).sum()), "dead": int(part[TARGET].eq(1).sum()), "unique_profiles": int(groups.loc[part.index].nunique())}
    manifest = {
        "pipeline_version": __version__, "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]},
        "data_quality": quality, "positive_class": {"label": "Dead", "encoded_value": 1},
        "protocol": {"outer_split": f"First of {folds} stratified group folds is held out", "inner_folds": folds,
                     "seed": seed, "group_key": "Identical pre-outcome predictor values", "decision_threshold": 0.5,
                     "selection": "Highest mean development-CV average precision; alphabetical name breaks exact ties",
                     "imbalance_handling": "Predefined class-weight variants; no resampling"},
        "partitions": partitions, "profile_overlap": len(set(groups.loc[development.index]) & set(groups.loc[test.index])),
        "selected_model": selected, "cv_results": cv, "test_metrics": results,
        "estimator_settings": {name: model.named_steps["model"].get_params() for name, model in fitted.items()},
        "limitations": ["Recorded vital status with varying follow-up; not a fixed-horizon survival estimate.",
                        "No external, temporal or patient-ID validation; identical predictor profiles are grouped.",
                        "Retrospective educational benchmark; no clinical validation or calibrated individual risk claim."],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame({"source_row": frame.index, "profile_group": groups, "partition": np.where(frame.index.isin(test.index), "test", "development")}).to_csv(output / "split_assignments.csv", index=False)
    joblib.dump({"pipeline": fitted[selected], "metadata": {"selected_model": selected, "positive_label": "Dead", "decision_threshold": 0.5, "pipeline_version": __version__}}, output / "model.joblib")
    plot_results(results, selected, output)
    print(f"Selected by development CV average precision: {selected}")
    print(f"Finished. Results: {output.resolve()}")
    return manifest
