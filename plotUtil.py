# Archivo auxiliar para generar imagenes.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 600 # Resolución final de los gráficos

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


def delays_vs_distance(df:pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Distance"], df["DepDelay"], s=1, alpha=0.2)
    plt.xlabel("Distancia (millas)")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Relación entre distancia y retraso")
    plt.savefig("./graficas/distance_vs_depdelay.png", dpi=DPI)
    plt.close()


def delays_vs_deptime(df:pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["CRSDepTime"],df["DepDelay"],s=1,alpha=0.2)
    plt.xlabel("Hora de salida programada (minutos desde medianoche)")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Relación entre hora programada y retraso")
    plt.savefig("./graficas/departure_time_vs_depdelay.png", dpi=DPI)
    plt.close()


def delays_by_month(df:pd.DataFrame):
    plt.figure(figsize=(10, 6))
    df.boxplot(column="DepDelay", by="Month")
    plt.xlabel("Mes")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Retraso en la salida por mes")
    plt.suptitle("")
    plt.savefig("./graficas/depdelay_by_month.png", dpi=DPI)
    plt.close()


def delays_by_week(df:pd.DataFrame):
    plt.clf()
    plt.figure(figsize=(10, 6))
    df.boxplot(column="DepDelay", by="DayOfWeek")
    plt.xlabel("Día de la semana")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Retraso en la salida por día de la semana")
    plt.suptitle("")
    plt.savefig("./graficas/depdelay_by_dayofweek.png", dpi=DPI)
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
    df = df.iloc[::, 0:7]
    sns.pairplot(df, hue="DayOfWeek").savefig("./graficas/pair_plot.png", dpi=DPI)