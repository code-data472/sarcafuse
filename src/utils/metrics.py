"""Evaluation metrics and statistical testing."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
    }

def bootstrap_test(y_true, pred_a, pred_b, n_resamples=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1_a = f1_score(y_true, pred_a, average="macro")
    f1_b = f1_score(y_true, pred_b, average="macro")
    observed_diff = f1_a - f1_b
    count = 0
    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        d = f1_score(y_true[idx], pred_a[idx], average="macro") - \
            f1_score(y_true[idx], pred_b[idx], average="macro")
        if d <= 0:
            count += 1
    return {"observed_diff": observed_diff, "p_value": count / n_resamples}
