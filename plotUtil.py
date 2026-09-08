import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 300

def ensure_graphics_directory():
    os.makedirs("./graficas", exist_ok=True)

def correlation_matrix(df: pd.DataFrame):
    candidate_columns = [
        "DepDelay",
        "CRSDepTime",
        "CRSArrTime",
        "Distance",
        "Month_sin",
        "Month_cos",
        "DayofMonth_sin",
        "DayofMonth_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
    ]
    columns = [column for column in candidate_columns if column in df.columns]
    ensure_graphics_directory()
    corr = df[columns].corr()
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=45, ha="right")
    plt.yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig("./graficas/correlation_matrix.png", dpi=DPI)
    plt.close()


def delays_hist(df:pd.DataFrame):
    ensure_graphics_directory()
    plt.figure(figsize=(10, 6))
    bins = int(np.sqrt(len(df)))
    plt.hist(df["DepDelay"], bins=100, range=(-25, 150)) 
    plt.xlabel("Retraso en la salida (minutos)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de los retrasos en la salida")
    plt.savefig("./graficas/depdelay_histogram.png", dpi=DPI)
    plt.close()


def mean_delay_by_month(df: pd.DataFrame):
    ensure_graphics_directory()
    month = _temporal_values(df, "Month", 12)
    monthly_mean = df.assign(Month=month).groupby("Month")["DepDelay"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_mean.index, monthly_mean.values, marker="o")
    plt.xlabel("Mes")
    plt.ylabel("Retraso promedio (minutos)")
    plt.title("Retraso promedio en la salida por mes")
    plt.xticks(range(1, 13))
    plt.grid()
    plt.savefig("./graficas/mean_depdelay_by_month.png", dpi=DPI)
    plt.close()


def mean_delay_by_week(df: pd.DataFrame):
    ensure_graphics_directory()
    day_of_week = _temporal_values(df, "DayOfWeek", 7)
    weekly_mean = df.assign(DayOfWeek=day_of_week).groupby("DayOfWeek")["DepDelay"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_mean.index, weekly_mean.values, marker="o")
    plt.xlabel("Día de la semana")
    plt.ylabel("Retraso promedio (minutos)")
    plt.title("Retraso promedio en la salida por día de la semana")
    plt.xticks(range(1, 8))
    plt.grid()
    plt.savefig("./graficas/mean_depdelay_by_weekly.png", dpi=DPI)
    plt.close()


def mean_delay_by_hour(df: pd.DataFrame):
    ensure_graphics_directory()
    temp = df.copy()
    temp["Hour"] = (temp["CRSDepTime"] // 60).astype(int)
    hourly_mean = temp.groupby("Hour")["DepDelay"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_mean.index, hourly_mean.values, marker="o")
    plt.xlabel("Hora de salida programada")
    plt.ylabel("Retraso promedio (minutos)")
    plt.title("Retraso promedio según hora de salida programada")
    plt.xticks(range(25))
    plt.grid()
    plt.savefig("./graficas/mean_depdelay_by_hour.png", dpi=DPI)
    plt.close()


def _temporal_values(df: pd.DataFrame, column: str, period: int):
    """Recupera la categoría temporal desde su par seno/coseno si hace falta."""
    if column in df.columns:
        return df[column]

    sin_column = f"{column}_sin"
    cos_column = f"{column}_cos"
    if sin_column not in df.columns or cos_column not in df.columns:
        raise ValueError(
            f"No se encontró {column} ni el par {sin_column}/{cos_column}."
        )

    angles = np.mod(np.arctan2(df[sin_column], df[cos_column]), 2 * np.pi)
    values = np.rint(angles * period / (2 * np.pi)).astype(int)
    return values.where(values != 0, period)

def pair_plot(df:pd.DataFrame):
    print(df)
    df = df.iloc[::, 0:7]
    sns.pairplot(df, hue="DayOfWeek").savefig("./graficas/pair_plot.png", dpi=DPI)

def predicted_vs_actual(y_real, y_pred, r2, dataset_name="Test", max_points=20000):
    """
    Predicción vs realidad.

    Si hay demasiadas observaciones se utiliza sampling
    para evitar intentar dibujar millones de puntos.
    """

    ensure_graphics_directory()

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    if len(y_real) > max_points:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(y_real),
            size=max_points,
            replace=False
        )

        y_real_plot = y_real[indices]
        y_pred_plot = y_pred[indices]

    else:

        y_real_plot = y_real
        y_pred_plot = y_pred

    min_value = min(
        y_real_plot.min(),
        y_pred_plot.min()
    )
    max_value = max(
        y_real_plot.max(),
        y_pred_plot.max()
    )

    plt.figure(figsize=(8, 8))
    plt.hexbin(
        y_real_plot,
        y_pred_plot,
        gridsize=60,
        mincnt=1
    )
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "--",
        linewidth=2,
        label="Predicción perfecta"
    )
    plt.xlabel("Retraso real (minutos)")
    plt.ylabel("Retraso predicho (minutos)")
    plt.title(
        f"Predicción vs. Real ({dataset_name})\n"
        f"R² = {r2:.4f}"
    )
    plt.colorbar(label="Número de observaciones")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        f"./graficas/predicted_vs_actual_{dataset_name.lower()}.png",
        dpi=DPI
    )
    plt.close()


def cost_evolution(train_costs, validation_costs, train_r2, validation_r2, test_r2):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_costs) + 1), train_costs, label="Training")
    plt.plot(range(1, len(validation_costs) + 1), validation_costs, label="Validation")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Evolución del error durante el entrenamiento")
    r2_text = (f"$R^2$ train: {train_r2:.4f}\n" f"$R^2$ validation: {validation_r2:.4f}\n" f"$R^2$ test: {test_r2:.4f}")
    plt.gca().text(
        0.98, 0.95, r2_text,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./cost_evolution.png", dpi=DPI)

def rf_cost_evolution(train_mse: list,validation_mse: list,estimators: list):
    plt.figure(figsize=(10, 6))
    plt.plot(estimators,train_mse,marker="o",label="Training MSE")
    plt.plot(estimators,validation_mse,marker="o",label="Validation MSE")
    plt.xlabel("Número de árboles")
    plt.ylabel("MSE")
    plt.title("Evolución del MSE según el número de árboles")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./graficas/rf_cost_evolution.png",dpi=DPI)
    plt.close()


def xgb_cost_evolution(train_rmse: list, val_rmse: list, best_iteration: int = None):
    """
    Grafica la evolución del RMSE de train y validation por cada árbol
    (boosting round) que XGBoost va agregando al modelo.

    Si se indica best_iteration (la iteración en la que actuó el early
    stopping), se marca con una línea vertical para visualizar en qué
    punto se detuvo el entrenamiento antes de empezar a sobreajustar.
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(train_rmse) + 1)
    plt.plot(iterations, train_rmse, label="Training RMSE")
    plt.plot(iterations, val_rmse, label="Validation RMSE")

    if best_iteration is not None:
        plt.axvline(
            x=best_iteration + 1,
            color="red",
            linestyle="--",
            alpha=0.6,
            label=f"Early stopping (árbol {best_iteration + 1})"
        )

    plt.xlabel("Número de árboles (boosting rounds)")
    plt.ylabel("RMSE")
    plt.title("Evolución del RMSE durante el entrenamiento (XGBoost)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./graficas/xgb_cost_evolution.png", dpi=DPI)
    plt.close()


def residual_plot(y_real, y_pred, dataset_name="Test", max_points=20000):
    """
    Residuales = predicción - realidad.
    """
    ensure_graphics_directory()

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    residuals = y_pred - y_real

    if len(y_pred) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(y_pred),
            size=max_points,
            replace=False
        )
        y_plot = y_pred[indices]
        residuals_plot = residuals[indices]
    else:
        y_plot = y_pred
        residuals_plot = residuals

    plt.figure(figsize=(10, 6))
    plt.hexbin(
        y_plot,
        residuals_plot,
        gridsize=70,
        mincnt=1
    )
    plt.axhline(
        0,
        linestyle="--",
        linewidth=2,
        label="Error = 0"
    )
    plt.xlabel("Retraso predicho (minutos)")
    plt.ylabel("Residual (predicho - real)")
    plt.title(
        f"Residuales vs. predicción ({dataset_name})"
    )
    plt.colorbar(label="Número de observaciones")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        f"./graficas/residuals_{dataset_name.lower()}.png",
        dpi=DPI
    )
    plt.close()


def error_distribution(y_real, y_pred, dataset_name="Test"):
    ensure_graphics_directory()

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    errors = y_pred - y_real

    plt.figure(figsize=(10, 6))
    plt.hist(
        errors,
        bins=100
    )
    plt.axvline(
        0,
        linestyle="--",
        linewidth=2
    )
    plt.xlabel("Error (predicción - realidad)")
    plt.ylabel("Frecuencia")
    plt.title(
        f"Distribución de errores ({dataset_name})"
    )
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        f"./graficas/error_distribution_{dataset_name.lower()}.png",
        dpi=DPI
    )
    plt.close()


def binned_prediction_plot(
    models_predictions,
    y_real,
    n_bins=20,
    dataset_name="Test"
):
    """
    Agrupa las predicciones en cuantiles y compara:

        promedio predicho
        vs
        promedio real

    Permite comparar varios modelos en una misma gráfica.
    """

    ensure_graphics_directory()

    y_real = np.asarray(y_real)

    plt.figure(figsize=(9, 8))

    min_value = float("inf")
    max_value = float("-inf")

    for model_name, predictions in models_predictions.items():

        y_pred = np.asarray(predictions)

        order = np.argsort(y_pred)

        sorted_pred = y_pred[order]
        sorted_real = y_real[order]

        bins = np.array_split(
            np.arange(len(sorted_pred)),
            n_bins
        )

        pred_means = []
        real_means = []

        for indices in bins:

            pred_means.append(
                sorted_pred[indices].mean()
            )

            real_means.append(
                sorted_real[indices].mean()
            )

        min_value = min(
            min_value,
            min(pred_means),
            min(real_means)
        )

        max_value = max(
            max_value,
            max(pred_means),
            max(real_means)
        )

        plt.plot(
            pred_means,
            real_means,
            marker="o",
            label=model_name
        )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "--",
        linewidth=2,
        label="Predicción perfecta"
    )

    plt.xlabel(
        "Predicción promedio por grupo (minutos)"
    )

    plt.ylabel(
        "Retraso real promedio por grupo (minutos)"
    )

    plt.title(
        f"Predicho vs. real agrupado ({dataset_name})"
    )

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        f"./graficas/binned_prediction_{dataset_name.lower()}.png",
        dpi=DPI
    )

    plt.close()


def compare_models_bar(
    results,
    metric,
    title=None
):
    """
    Compara modelos mediante una gráfica de barras.

    results:
        {
            "Random Forest": metrics,
            "XGBoost": metrics
        }

    metric:
        "test_mae"
        "test_rmse"
        "test_r2"
        etc.
    """

    ensure_graphics_directory()

    model_names = list(results.keys())

    values = [
        results[name][metric]
        for name in model_names
    ]

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        model_names,
        values
    )

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    plt.ylabel(metric)

    if title is None:
        title = f"Comparación — {metric}"

    plt.title(title)

    plt.xticks(
        rotation=15
    )

    plt.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        f"./graficas/comparison_{metric}.png",
        dpi=DPI
    )

    plt.close()


def plot_training_history(
    x,
    train_values,
    validation_values,
    xlabel,
    ylabel,
    title,
    filename
):
    """
    Gráfica genérica de aprendizaje.
    """

    ensure_graphics_directory()

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        train_values,
        marker="o",
        label="Training"
    )

    plt.plot(
        x,
        validation_values,
        marker="o",
        label="Validation"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        f"./graficas/{filename}.png",
        dpi=DPI
    )

    plt.close()