"""Custom transformer, preprocessing ColumnTransformer and candidate models."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Add engineered ratio features. Follows the full scikit-learn estimator API,
    so it can be cloned, cross-validated, grid-searched and pickled."""

    def __init__(self, add_ratios=True):
        self.add_ratios = add_ratios

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        if self.add_ratios:
            X["avg_monthly_paid"] = X["total_charges"] / (X["tenure"] + 1)
            X["calls_per_month"] = X["support_calls"] / (X["tenure"] + 1)
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        return np.array(names + (["avg_monthly_paid", "calls_per_month"] if self.add_ratios else []),
                        dtype=object)


def build_preprocessor() -> Pipeline:
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                OneHotEncoder(handle_unknown="ignore"))
    columns = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([("ratios", RatioFeatures()), ("columns", columns)])


def build_models(seed=0) -> dict:
    """Full pipelines (preprocessing + model) ready for cross-validation."""
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("model", est)])
            for name, est in candidates.items()}
