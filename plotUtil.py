# Archivo auxiliar para generar imagenes.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 600 # Resolución final de los gráficos

# def scatter_matrix_plot(df:pd.DataFrame):
#     print("Generando Scatter Matrix (solo con columnas CRSDepTime, CRSArrTime y Distance)...")
#     axes = pd.plotting.scatter_matrix(df, figsize=(9, 9))
#     for ax in axes.flatten():
#         ax.xaxis.label.set_rotation(45)
#         ax.yaxis.label.set_rotation(0)
#         ax.yaxis.label.set_ha('right')
#     plt.tight_layout()
#     plt.title("")
#     plt.savefig("scatter_matrix.png", dpi=DPI)
#     print("¡Gráfico generado! guardado como scatter_matrix.png")
#     plt.close()

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
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=45, ha="right")
    plt.yticks(range(len(columns)), columns)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=DPI)
    plt.close()


def delays_hist(df:pd.DataFrame):
    print("Generando histograma de DepDelay...")
    plt.figure(figsize=(10, 6))
    bins = int(np.sqrt(len(df)))
    plt.hist(df["DepDelay"], bins=100, range=(-25, 150)) 
    plt.xlabel("Retraso en la salida (minutos)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de los retrasos en la salida")
    plt.savefig("depdelay_histogram.png", dpi=DPI)
    print("¡Grafico generado! guardado como depdelay_histogram.png")
    plt.close()


def delays_vs_distance(df:pd.DataFrame):
    print("Generando scatter plot de Distance vs DepDelay...")
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Distance"], df["DepDelay"], s=1, alpha=0.2)
    plt.xlabel("Distancia (millas)")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Relación entre distancia y retraso")
    plt.savefig("distance_vs_depdelay.png", dpi=DPI)
    print("¡Grafico generado! guardado como distance_vs_depdelay.png")
    plt.close()


def delays_vs_deptime(df:pd.DataFrame):
    print("Generando scatter plot de DepTime vs DepDelay...")
    plt.figure(figsize=(10, 6))
    plt.scatter(df["CRSDepTime"],df["DepDelay"],s=1,alpha=0.2)
    plt.xlabel("Hora de salida programada (minutos desde medianoche)")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Relación entre hora programada y retraso")
    plt.savefig("departure_time_vs_depdelay.png", dpi=DPI)
    print("¡Grafico generado! guardado como departure_time_vs_depdelay.png")
    plt.close()


def delays_by_month(df:pd.DataFrame):
    plt.figure(figsize=(10, 6))
    df.boxplot(column="DepDelay", by="Month")
    plt.xlabel("Mes")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Retraso en la salida por mes")
    plt.suptitle("")
    plt.savefig("depdelay_by_month.png", dpi=DPI)
    plt.close()


def delays_by_week(df:pd.DataFrame):
    plt.clf()
    plt.figure(figsize=(10, 6))
    df.boxplot(column="DepDelay", by="DayOfWeek")
    plt.xlabel("Día de la semana")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Retraso en la salida por día de la semana")
    plt.suptitle("")
    plt.savefig("depdelay_by_dayofweek.png", dpi=DPI)
    plt.close()


def mean_delay_by_month(df: pd.DataFrame):
    monthly_mean = df.groupby("Month")["DepDelay"].mean()
    plt.figure(figsize=(10, 6))
    # plt.bar(monthly_mean.index, monthly_mean.values)
    plt.plot(monthly_mean.index, monthly_mean.values, marker="o")
    plt.xlabel("Mes")
    plt.ylabel("Retraso promedio (minutos)")
    plt.title("Retraso promedio en la salida por mes")
    plt.xticks(range(1, 13))
    plt.grid()
    plt.savefig("mean_depdelay_by_month.png", dpi=DPI)
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
    plt.savefig("mean_depdelay_by_weekly.png", dpi=DPI)
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
    plt.savefig("mean_depdelay_by_hour.png", dpi=DPI)
    plt.close()

# Implementación tomada de https://www.geeksforgeeks.org/python/pairplot-in-matplotlib/ y ajustada a mi caso
def pair_plot(df: pd.DataFrame):
    # Number of features
    num_features = len(df.columns)
    # Create Subplots Grid
    fig, axes = plt.subplots(num_features, num_features, figsize=(10, 10))
    # Loop through each pair of features
    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: Histogram of the feature
                ax.hist(df.iloc[:, i], bins=15, color='skyblue', edgecolor='black')
            else:
                # Scatter plot for feature pairs
                ax.scatter(df.iloc[:, j], df.iloc[:, i], alpha=0.7, s=10, color="blue")

            # Set labels on the left and bottom axes
            if j == 0:
                ax.set_ylabel(df.columns[i], fontsize=10)
            if i == num_features - 1:
                ax.set_xlabel(df.columns[j], fontsize=10)

            # Remove ticks for a cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

    # Adjust layout
    plt.tight_layout()
    plt.title("Pair plot")
    plt.savefig("pair_plot.png", dpi=DPI)