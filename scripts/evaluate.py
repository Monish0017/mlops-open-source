import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_validate

def evaluate(model, X, y):
    preds = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, average="weighted"),
        "recall": recall_score(y, preds, average="weighted"),
        "precision": precision_score(y, preds, average="weighted"),
    }
    if proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y, proba, multi_class="ovr")
        except:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    # Cross-validation scores
    cv_results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'f1_weighted', 'recall_weighted'])
    metrics["cv_accuracy"] = cv_results['test_accuracy'].mean()
    metrics["cv_f1"] = cv_results['test_f1_weighted'].mean()
    metrics["cv_recall"] = cv_results['test_recall_weighted'].mean()

    return metrics
