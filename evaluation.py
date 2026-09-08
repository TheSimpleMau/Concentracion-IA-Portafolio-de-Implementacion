import math
import numpy as np
import pandas as pd


def regression_metrics(y_real, y_pred, tolerances=(5, 10, 15, 30)):
    real = np.asarray(y_real, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(real) != len(pred):
        raise ValueError(
            "y_real y y_pred deben tener la misma cantidad de observaciones."
        )

    n = len(real)
    if n == 0:
        raise ValueError("No hay observaciones para evaluar.")

    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(pred)):
        raise ValueError("Las métricas no aceptan valores NaN o infinitos.")

    errors = pred - real
    absolute_errors = np.abs(errors)
    mse = float(np.mean(np.square(errors)))
    rmse = math.sqrt(mse)
    mae = float(np.mean(absolute_errors))
    mean_real = float(np.mean(real))
    ss_res = float(np.sum(np.square(real - pred)))
    ss_tot = float(np.sum(np.square(real - mean_real)))

    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = (1 - ss_res / ss_tot)

    mean_error = float(np.mean(errors))
    median_absolute_error = float(np.median(absolute_errors))

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_error": mean_error,
        "median_absolute_error": median_absolute_error
        }

    for tolerance in tolerances:
        metrics[f"within_{tolerance}"] = float(
            np.mean(absolute_errors <= tolerance) * 100
        )

    return metrics


def evaluate_all_splits(y_train, train_pred, y_val, val_pred, y_test=None, test_pred=None):
    splits = {
        "train": (
            y_train,
            train_pred
        ),
        "val": (
            y_val,
            val_pred
        )
    }

    if (y_test is None) != (test_pred is None):
        raise ValueError("y_test y test_pred deben proporcionarse juntos.")
    if y_test is not None:
        splits["test"] = (y_test, test_pred)

    metrics = {}

    for split_name, (y_real, y_pred) in splits.items():
        split_metrics = regression_metrics(y_real, y_pred)
        for metric_name, value in split_metrics.items():
            metrics[f"{split_name}_{metric_name}"] = value

    return metrics


def create_comparison_dataframe(results: dict, split="test"):
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "modelo": model_name,
            "MAE": metrics.get(
                f"{split}_mae"
            ),
            "RMSE": metrics.get(
                f"{split}_rmse"
            ),
            "R2": metrics.get(
                f"{split}_r2"
            ),
            "MedAE": metrics.get(
                f"{split}_median_absolute_error"
            ),
            "±5 min": metrics.get(
                f"{split}_within_5"
            ),
            "±10 min": metrics.get(
                f"{split}_within_10"
            ),
            "±15 min": metrics.get(
                f"{split}_within_15"
            ),
            "±30 min": metrics.get(
                f"{split}_within_30"
            ),
            "Bias": metrics.get(
                f"{split}_mean_error"
            )
        })
    return pd.DataFrame(rows)



# Diagnóstico validation / test

def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _percentage_change(reference, value):
    if reference is None or value is None or reference == 0:
        return None
    return ((value - reference) / abs(reference)) * 100


def create_diagnostic_dataframe(results: dict):
    rows = []

    for model_name, metrics in results.items():
        train_rmse = metrics.get("train_rmse")
        val_rmse = metrics.get("val_rmse")
        test_rmse = metrics.get("test_rmse")

        train_mae = metrics.get("train_mae")
        val_mae = metrics.get("val_mae")
        test_mae = metrics.get("test_mae")

        train_r2 = metrics.get("train_r2")
        val_r2 = metrics.get("val_r2")
        test_r2 = metrics.get("test_r2")

        train_bias = metrics.get("train_mean_error")
        val_bias = metrics.get("val_mean_error")
        test_bias = metrics.get("test_mean_error")

        rows.append({
            "Modelo": model_name,
            # RMSE
            "Train RMSE": train_rmse,
            "Validation RMSE": val_rmse,
            "Test RMSE": test_rmse,
            "Val-Train RMSE %": _percentage_change(train_rmse,val_rmse),
            "Test-Val RMSE %": _percentage_change(val_rmse,test_rmse),
            "Test/Val RMSE": _safe_ratio(test_rmse,val_rmse),
            # MAE
            "Train MAE": train_mae,
            "Validation MAE": val_mae,
            "Test MAE": test_mae,
            "Val-Train MAE %": _percentage_change(train_mae,val_mae),
            "Test-Val MAE %": _percentage_change(val_mae,test_mae),
            # R²
            "Train R²": train_r2,
            "Validation R²": val_r2,
            "Test R²": test_r2,
            "Test-Val R²": (test_r2 - val_r2 if test_r2 is not None and val_r2 is not None else None),
            # Bias
            "Train Bias": train_bias,
            "Validation Bias": val_bias,
            "Test Bias": test_bias,
            "Test-Val Bias": (test_bias - val_bias if test_bias is not None and val_bias is not None else None),
            # Tolerancias
            "Validation ±5": metrics.get("val_within_5"),
            "Test ±5": metrics.get("test_within_5"),
            "Validation ±10": metrics.get("val_within_10"),
            "Test ±10": metrics.get("test_within_10"),
            "Validation ±15": metrics.get("val_within_15"),
            "Test ±15": metrics.get("test_within_15"),
            "Validation ±30": metrics.get("val_within_30"),
            "Test ±30": metrics.get("test_within_30")
        })

    return pd.DataFrame(rows)


def print_diagnostic_summary(results: dict):
    print("\n")
    print("=" * 70)
    print("DIAGNÓSTICO DE VALIDATION Y TEST")
    print("=" * 70)

    for model_name, metrics in results.items():
        print(f"\n{'-' * 70}")
        print(f"MODELO: {model_name}")
        print(f"{'-' * 70}")
        print("\n[ Métricas principales ]")
        print(
            f"  Train      -> "
            f"MAE: {metrics['train_mae']:.4f} | "
            f"RMSE: {metrics['train_rmse']:.4f} | "
            f"R²: {metrics['train_r2']:.4f}"
        )

        print(
            f"  Validation -> "
            f"MAE: {metrics['val_mae']:.4f} | "
            f"RMSE: {metrics['val_rmse']:.4f} | "
            f"R²: {metrics['val_r2']:.4f}"
        )

        has_test = "test_rmse" in metrics
        if has_test:
            print(
                f"  Test       -> "
                f"MAE: {metrics['test_mae']:.4f} | "
                f"RMSE: {metrics['test_rmse']:.4f} | "
                f"R²: {metrics['test_r2']:.4f}"
            )

        val_train_rmse = _percentage_change(
            metrics["train_rmse"],
            metrics["val_rmse"]
        )

        val_train_mae = _percentage_change(
            metrics["train_mae"],
            metrics["val_mae"]
        )

        print("\n[ Train → Validation ]")

        print(
            f"  Cambio RMSE: "
            f"{val_train_rmse:+.2f}%"
        )

        print(
            f"  Cambio MAE:  "
            f"{val_train_mae:+.2f}%"
        )

        print(
            f"  Cambio R²:   "
            f"{metrics['val_r2'] - metrics['train_r2']:+.4f}"
        )

        if has_test:
            test_val_rmse = _percentage_change(
                metrics["val_rmse"],
                metrics["test_rmse"]
            )

            test_val_mae = _percentage_change(
                metrics["val_mae"],
                metrics["test_mae"]
            )

            test_val_r2 = (
                metrics["test_r2"]
                - metrics["val_r2"]
            )

            print("\n[ Validation → Test ]")

            print(
                f"  Cambio RMSE: "
                f"{test_val_rmse:+.2f}%"
            )

            print(
                f"  Cambio MAE:  "
                f"{test_val_mae:+.2f}%"
            )

            print(
                f"  Cambio R²:   "
                f"{test_val_r2:+.4f}"
            )

        print("\n[ Bias ]")

        print(
            f"  Train      : "
            f"{metrics['train_mean_error']:+.4f}"
        )

        print(
            f"  Validation : "
            f"{metrics['val_mean_error']:+.4f}"
        )

        if has_test:
            print(
                f"  Test       : "
                f"{metrics['test_mean_error']:+.4f}"
            )

        print("\n[ Predicciones dentro de tolerancia ]")

        for tolerance in [5, 10, 15, 30]:
            val_percentage = metrics[
                f"val_within_{tolerance}"
            ]
            if has_test:
                test_percentage = metrics[
                    f"test_within_{tolerance}"
                ]
                difference = (
                    test_percentage
                    - val_percentage
                )
                print(
                    f"  ±{tolerance:2d} minutos -> "
                    f"Validation: {val_percentage:6.2f}% | "
                    f"Test: {test_percentage:6.2f}% | "
                    f"Δ: {difference:+.2f}%"
                )
            else:
                print(
                    f"  ±{tolerance:2d} minutos -> "
                    f"Validation: {val_percentage:6.2f}%"
                )

        print("\n[ Error absoluto mediano ]")
        print(
            f"  Validation : "
            f"{metrics['val_median_absolute_error']:.4f}"
        )
        if has_test:
            print(
                f"  Test       : "
                f"{metrics['test_median_absolute_error']:.4f}"
            )

    print("\n" + "=" * 70)