import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, prepare_features


class FeatureSchema(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        prepare_features(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return self

    def transform(self, X):
        return prepare_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def candidates(seed=42):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    estimators = {
        "majority_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "logistic_balanced": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=1),
        "random_forest_balanced": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=1),
    }
    from sklearn.base import clone
    return {name: Pipeline([("schema", FeatureSchema()), ("preprocess", clone(preprocess)), ("model", model)])
            for name, model in estimators.items()}
