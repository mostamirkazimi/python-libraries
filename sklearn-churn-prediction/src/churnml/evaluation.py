"""Model comparison, tuning, test-set evaluation and plots."""
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (RocCurveDisplay, classification_report, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate


def compare_models(models, X, y, seed=0):
    """5-fold stratified cross-validation of several pipelines."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = {}
    for name, pipe in models.items():
        r = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision", "f1"])
        rows[name] = {m: r[f"test_{m}"].mean() for m in ["roc_auc", "average_precision", "f1"]}
        rows[name]["roc_auc_std"] = r["test_roc_auc"].std()
    return pd.DataFrame(rows).T.sort_values("roc_auc", ascending=False).round(4)


def tune_gradient_boosting(pipe, X, y, n_iter=8, seed=0):
    """Randomised hyper-parameter search over the model step of the pipeline."""
    space = {
        "model__learning_rate": loguniform(0.02, 0.3),
        "model__max_leaf_nodes": randint(8, 48),
        "model__min_samples_leaf": randint(10, 60),
        "model__l2_regularization": loguniform(1e-3, 10),
    }
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc",
                                cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                random_state=seed, n_jobs=-1)
    return search.fit(X, y)


def test_metrics(estimator, X_test, y_test):
    proba = estimator.predict_proba(X_test)[:, 1]
    return {"roc_auc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
            "report": classification_report(y_test, proba > 0.5, output_dict=True)}


def feature_importance(estimator, X_test, y_test, seed=0):
    """Permutation importance on the *raw* columns (works through the whole pipeline)."""
    r = permutation_importance(estimator, X_test, y_test, scoring="roc_auc",
                               n_repeats=5, random_state=seed, n_jobs=-1)
    return (pd.Series(r.importances_mean, index=X_test.columns)
            .sort_values(ascending=False))


def plot_evaluation(fitted_models, X_test, y_test, importance, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, est in fitted_models.items():
        RocCurveDisplay.from_estimator(est, X_test, y_test, name=name, ax=ax[0])
        CalibrationDisplay.from_estimator(est, X_test, y_test, n_bins=8, name=name, ax=ax[1])
    ax[0].set_title("ROC curves")
    ax[1].set_title("Calibration")
    importance.iloc[::-1].plot.barh(ax=ax[2], color="#2563eb")
    ax[2].set_title("Permutation importance (ROC AUC drop)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
