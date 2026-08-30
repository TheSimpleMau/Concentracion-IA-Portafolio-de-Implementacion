# Archivo auxiliar para generar imagenes.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 900 # Resolución final de los gráficos

def correlation_matrix(df: pd.DataFrame):
    columns = [
        "DepDelay",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "Distance"
    ]
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
    plt.figure(figsize=(10, 6))
    bins = int(np.sqrt(len(df)))
    plt.hist(df["DepDelay"], bins=100, range=(-25, 150)) 
    plt.xlabel("Retraso en la salida (minutos)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de los retrasos en la salida")
    plt.savefig("./graficas/depdelay_histogram.png", dpi=DPI)
    plt.close()


def mean_delay_by_month(df: pd.DataFrame):
    monthly_mean = df.groupby("Month")["DepDelay"].mean()
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
    weekly_mean = df.groupby("DayOfWeek")["DepDelay"].mean()
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

def pair_plot(df:pd.DataFrame):
    print(df)
    df = df.iloc[::, 0:7]
    sns.pairplot(df, hue="DayOfWeek").savefig("./graficas/pair_plot.png", dpi=DPI)

def predicted_vs_actual(y_real: list, y_pred: list, r2: float, dataset_name="Test"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, y_pred, s=1, alpha=0.15)
    plt.legend()
    lims = [min(min(y_real), min(y_pred)), max(max(y_real), max(y_pred))]
    plt.plot(lims, lims, 'r--', label="Predicción perfecta (y = x)")
    plt.xlabel("Retraso real (minutos)")
    plt.ylabel("Retraso predicho (minutos)")
    plt.title(f"Predicción vs. Real ({dataset_name}) — $R^2$ = {r2:.4f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./graficas/predicted_vs_actual_{dataset_name.lower()}.png", dpi=DPI)
    plt.close()