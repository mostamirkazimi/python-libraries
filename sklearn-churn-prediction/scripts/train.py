"""Full workflow: compare -> tune -> evaluate -> explain -> save."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split

from churnml import build_models, make_churn_dataset
from churnml.evaluation import (compare_models, feature_importance, plot_evaluation,
                                test_metrics, tune_gradient_boosting)

OUT = Path("outputs")


def main():
    OUT.mkdir(exist_ok=True)
    X, y = make_churn_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)

    models = build_models()
    table = compare_models(models, X_train, y_train)
    print("Cross-validation:\n", table, "\n")
    table.to_csv(OUT / "cv_results.csv")

    search = tune_gradient_boosting(models["hist_gradient_boosting"], X_train, y_train)
    print("Best CV AUC:", round(search.best_score_, 4))
    print("Best params:", search.best_params_)

    fitted = {name: m.fit(X_train, y_train) for name, m in models.items()
              if name != "hist_gradient_boosting"}
    fitted["tuned_gradient_boosting"] = search.best_estimator_

    metrics = {name: test_metrics(m, X_test, y_test) for name, m in fitted.items()}
    for name, m in metrics.items():
        print(f"{name:26s} test AUC {m['roc_auc']:.4f}  AP {m['average_precision']:.4f}")
    (OUT / "metrics.json").write_text(json.dumps(
        {"cv": table.to_dict("index"), "test": metrics,
         "best_params": {k: float(v) for k, v in search.best_params_.items()}}, indent=2))

    importance = feature_importance(search.best_estimator_, X_test, y_test)
    print("\nPermutation importance:\n", importance.round(4))
    plot_evaluation(fitted, X_test, y_test, importance, OUT / "evaluation.png")

    joblib.dump(search.best_estimator_, OUT / "model.joblib")


if __name__ == "__main__":
    main()
