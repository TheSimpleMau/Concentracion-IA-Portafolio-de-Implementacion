import math
import pandas as pd


def regression_metrics(y_real, y_pred, tolerances=(5, 10, 15, 30)):
    n = len(y_real)
    if n == 0:
        raise ValueError("No hay observaciones para evaluar.")

    errors = []
    squared_errors = []
    absolute_errors = []

    for real, pred in zip(y_real, y_pred):

        current_error = (pred - real)
        errors.append(current_error)
        squared_errors.append(current_error ** 2)
        absolute_errors.append(abs(current_error))

    mse = (sum(squared_errors)/ n)
    rmse = math.sqrt(mse)
    mae = (sum(absolute_errors)/ n)
    mean_real = (sum(y_real)/ n)
    ss_res = 0.0
    ss_tot = 0.0

    for real, pred in zip(y_real,y_pred):
        ss_res += (real - pred) ** 2
        ss_tot += (real - mean_real) ** 2

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = (1 - ss_res / ss_tot)

    mean_error = (sum(errors)/ n)
    sorted_abs_errors = sorted(absolute_errors)

    middle = n // 2
    if n % 2 == 0:
        median_absolute_error = (sorted_abs_errors[middle - 1] + sorted_abs_errors[middle]) / 2
    else:
        median_absolute_error = (sorted_abs_errors[middle])

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_error": mean_error,
        "median_absolute_error": median_absolute_error
        }

    for tolerance in tolerances:
        count = sum(1 for absolute_error in absolute_errors if absolute_error <= tolerance)
        metrics[f"within_{tolerance}"] = (count / n) * 100

    return metrics


def evaluate_all_splits(y_train, train_pred, y_val, val_pred, y_test, test_pred):
    splits = {
        "train": (
            y_train,
            train_pred
        ),
        "val": (
            y_val,
            val_pred
        ),
        "test": (
            y_test,
            test_pred
        )
    }

    metrics = {}

    for split_name, (y_real, y_pred) in splits.items():
        split_metrics = regression_metrics(y_real, y_pred)
        for metric_name, value in split_metrics.items():
            metrics[f"{split_name}_{metric_name}"] = value

    return metrics


def create_comparison_dataframe(results: dict):
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "modelo": model_name,
            "MAE": metrics.get(
                "test_mae"
            ),
            "RMSE": metrics.get(
                "test_rmse"
            ),
            "R2": metrics.get(
                "test_r2"
            ),
            "MedAE": metrics.get(
                "test_median_absolute_error"
            ),
            "±5 min": metrics.get(
                "test_within_5"
            ),
            "±10 min": metrics.get(
                "test_within_10"
            ),
            "±15 min": metrics.get(
                "test_within_15"
            ),
            "±30 min": metrics.get(
                "test_within_30"
            ),
            "Bias": metrics.get(
                "test_mean_error"
            )
        })
    return pd.DataFrame(rows)