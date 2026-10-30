import numpy as np
import pandas as pd
import pytest

from breast_cancer.data import FEATURES, grouped_splits, load_data, predictor_groups, prepare_features


def test_bundled_data_profile_and_positive_class(dataset):
    frame, quality = dataset
    assert quality["source_sha256"] == "667fa828fb39d85f6a622dd2067ed3b5221f5460aca6ceec2be5fcfb318d69bd"
    assert quality["source_rows"] == 4024
    assert quality["retained_rows"] == 4023
    assert quality["exact_duplicates_removed"] == 1
    assert quality["class_counts"] == {"Alive": 3407, "Dead": 616}
    assert quality["unique_predictor_profiles"] == 4010
    assert quality["profiles_with_both_labels"] == 1
    assert set(frame.Status) == {0, 1}


def test_whitespace_aliases_and_grade_levels_preserve_information(source):
    features = prepare_features(source)
    assert "T Stage" in features and "Regional Node Positive" in features
    assert "Single" in set(features["Marital Status"])
    assert "Separated" in set(features["Marital Status"])
    assert "T4" in set(features["T Stage"])
    assert set(features.Grade) == {"1", "2", "3", "4"}
    assert features.index.equals(source.index)


def test_post_outcome_information_is_excluded(source):
    expected = prepare_features(source)
    altered = source.copy()
    altered["Survival Months"] = 1e9
    altered["Status"] = "different outcome"
    pd.testing.assert_frame_equal(expected, prepare_features(altered))
    assert expected.columns.tolist() == FEATURES


@pytest.mark.parametrize("change", ["column_missing", "column_collision", "bad_grade", "infinite", "negative"])
def test_invalid_predictors_are_rejected(source, change):
    frame = source.head(5).copy()
    if change == "column_missing": frame = frame.drop(columns="Age")
    elif change == "column_collision": frame[" Age "] = frame.Age
    elif change == "bad_grade": frame.loc[0,"Grade"] = "not a grade"
    elif change == "infinite": frame["Age"] = np.inf
    else: frame["Age"] = -1
    with pytest.raises(ValueError): prepare_features(frame)


@pytest.mark.parametrize("label", [None, "Unknown", "No cancer"])
def test_invalid_target_is_rejected(source, tmp_path, label):
    frame = source.head(30).copy()
    frame.loc[0,"Status"] = label
    path=tmp_path / "data.csv";frame.to_csv(path,index=False)
    with pytest.raises(ValueError, match="Status"): load_data(path)


def test_outer_and_inner_splits_keep_profiles_disjoint(dataset):
    frame=dataset[0];groups=predictor_groups(frame)
    development_ids,test_ids=grouped_splits(frame)[0]
    assert not set(groups.iloc[development_ids]) & set(groups.iloc[test_ids])
    development=frame.iloc[development_ids]
    inner_groups=predictor_groups(development)
    for train,validation in grouped_splits(development):
        assert not set(inner_groups.iloc[train]) & set(inner_groups.iloc[validation])


def test_conflicting_labels_for_identical_profiles_stay_together(small):
    changed=small.copy()
    row=changed.iloc[[0]].copy();row.Status=1-int(row.Status.iloc[0])
    changed=pd.concat([changed,row],ignore_index=True)
    for train,test in grouped_splits(changed):
        assert (0 in train)==(len(changed)-1 in train)


def test_split_is_deterministic(small):
    first,second=grouped_splits(small),grouped_splits(small)
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1],b[1])
