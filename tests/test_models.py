import joblib
import numpy as np
import pytest
from sklearn.base import clone

from breast_cancer.data import CATEGORICAL, FEATURES
from breast_cancer.evaluate import metrics
from breast_cancer.experiment import choose_model, run_experiment
from breast_cancer.models import candidates


def test_scaler_and_vocabulary_fit_training_only(small):
    train=small.iloc[:80].copy()
    model=candidates()["logistic_regression"].fit(train[FEATURES],train.Status)
    numeric=model.named_steps["preprocess"].named_transformers_["numeric"]
    np.testing.assert_allclose(numeric.named_steps["scale"].mean_,train[["Age","Tumor Size","Regional Node Examined","Regional Node Positive"]].mean())
    encoder=model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"]
    future=small.iloc[80:].copy();future["Race"]="Unseen validation category"
    before=[v.copy() for v in encoder.categories_]
    scores=model.predict_proba(future)
    assert np.isfinite(scores).all()
    for a,b in zip(before,encoder.categories_): np.testing.assert_array_equal(a,b)
    assert "Unseen validation category" not in encoder.categories_[CATEGORICAL.index("Race")]


def test_missing_predictors_are_imputed_inside_pipeline(small):
    train=small.copy()
    train.loc[0,"Age"]=np.nan
    train.loc[1,"Race"]=np.nan
    model=candidates()["logistic_regression"].fit(train,train.Status)
    assert np.isfinite(model.predict_proba(train)).all()


def test_predictions_ignore_followup_and_target(small):
    model=candidates()["random_forest"].fit(small,small.Status)
    altered=small.copy();altered["Survival Months"]=1e6;altered["Status"]=1-altered.Status
    np.testing.assert_allclose(model.predict_proba(small),model.predict_proba(altered),rtol=0,atol=0)


def test_saved_pipeline_round_trip_and_raw_column_aliases(source, small, tmp_path):
    model=candidates()["logistic_regression"].fit(small,small.Status)
    path=tmp_path / "model.joblib";joblib.dump(model,path)
    restored=joblib.load(path)
    np.testing.assert_allclose(model.predict_proba(source.head(8)),restored.predict_proba(source.head(8)))


def test_selection_uses_cv_average_precision_only():
    cv={"more_accurate":{"mean_average_precision":0.2},"better_minority_ranking":{"mean_average_precision":0.5}}
    assert choose_model(cv)=="better_minority_ranking"
    assert choose_model({"z":{"mean_average_precision":0.5},"a":{"mean_average_precision":0.5}})=="a"


def test_metrics_name_dead_as_positive_and_show_baseline():
    result=metrics([0,0,0,1],[0.1,0.1,0.1,0.1])
    assert result["accuracy"]==0.75
    assert result["balanced_accuracy"]==0.5
    assert result["dead_recall"]==0
    assert result["average_precision"]==0.25
    assert result["confusion_matrix"]=={"true_alive":3,"false_dead":0,"false_alive":1,"true_dead":0}


@pytest.mark.parametrize("probability", [[0.1,np.nan],[0.1,1.1],[0.1]])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError): metrics([0,1],probability)


def test_end_to_end_benchmark_keeps_holdout_and_saves_model(small,tmp_path):
    manifest=run_experiment(small,{"fixture":"small subset of bundled data"},tmp_path,folds=3)
    assert manifest["profile_overlap"]==0
    assert manifest["selected_model"]==choose_model(manifest["cv_results"])
    assert len(manifest["test_metrics"])==5
    assert set(manifest["cv_results"])==set(manifest["test_metrics"])
    saved=joblib.load(tmp_path / "model.joblib")
    assert saved["metadata"]["positive_label"]=="Dead"
    assert np.isfinite(saved["pipeline"].predict_proba(small)).all()
    assert (tmp_path / "model_comparison.svg").stat().st_size>1000
    assert (tmp_path / "selected_confusion.png").stat().st_size>1000
