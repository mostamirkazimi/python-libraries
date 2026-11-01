import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from churnml import RatioFeatures, build_models, build_preprocessor, make_churn_dataset


def test_dataset_shape_and_missing_values():
    X, y = make_churn_dataset(n=500, seed=1)
    assert X.shape == (500, 8) and len(y) == 500
    assert X.isna().any().any()
    assert set(y.unique()) == {0, 1}


def test_ratio_features_adds_columns_and_names():
    X, _ = make_churn_dataset(n=100)
    t = RatioFeatures().fit(X)
    out = t.transform(X)
    assert {"avg_monthly_paid", "calls_per_month"} <= set(out.columns)
    assert list(t.get_feature_names_out()) == list(out.columns)
    assert "avg_monthly_paid" not in clone(RatioFeatures(add_ratios=False)).fit(X).transform(X)


def test_preprocessor_outputs_no_nan():
    X, _ = make_churn_dataset(n=300)
    Z = build_preprocessor().fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    assert not np.isnan(Z).any()


def test_unseen_category_does_not_crash():
    X, y = make_churn_dataset(n=600)
    model = build_models()["logistic_regression"].fit(X, y)
    X_new = X.head(5).copy()
    X_new["payment_method"] = "crypto"          # never seen during training
    assert model.predict_proba(X_new).shape == (5, 2)


def test_models_beat_chance():
    X, y = make_churn_dataset(n=2000, seed=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
    for name in ["logistic_regression", "hist_gradient_boosting"]:
        model = build_models()[name].fit(Xtr, ytr)
        assert roc_auc_score(yte, model.predict_proba(Xte)[:, 1]) > 0.7, name
