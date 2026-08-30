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

def depArr_plot(df: pd.DataFrame, DPI: int = 300):
    df_plot = df.copy()

    # --- VARIABLE 3: DISTANCIA (Clasificación por Color) ---
    rangos_distancia = [-float("inf"), 500, 1500, 2500, float("inf")]
    etiquetas_distancia = [
        "Corta (<500 mi)",
        "Media (500-1500 mi)",
        "Larga (1500-2500 mi)",
        "Muy Larga (>2500 mi)",
    ]
    df_plot["RangoDistancia"] = pd.cut(
        df_plot["Distance"],
        bins=rangos_distancia,
        labels=etiquetas_distancia,
    )

    colores_mapa = {
        "Corta (<500 mi)": "#3498db",      # Azul
        "Media (500-1500 mi)": "#2ecc71",  # Verde
        "Larga (1500-2500 mi)": "#f39c12", # Naranja
        "Muy Larga (>2500 mi)": "#9b59b6", # Púrpura
    }

    # --- VARIABLE 4: RETRASOS (6 Categorías para Tamaño de Burbuja) ---
    # Definimos los cortes en minutos (puedes ajustar estos umbrales según tu criterio)
    rangos_retraso = [-float('inf'), 0, 15, 45, 90, 180, float('inf')]
    etiquetas_retraso = [
        'A tiempo / Adelantado', 
        'Muy poco (1-15 min)', 
        'Poco (16-45 min)', 
        'Moderado (46-90 min)', 
        'Alto (91-180 min)', 
        'Muy Alto (>180 min)'
    ]
    df_plot['RangoRetraso'] = pd.cut(
        df_plot['DepDelay'], 
        bins=rangos_retraso, 
        labels=etiquetas_retraso
    )
    
    # Asignamos un tamaño en píxeles que vaya creciendo para cada categoría
    tamanos_mapa = {
        'A tiempo / Adelantado': 15,
        'Muy poco (1-15 min)': 40,
        'Poco (16-45 min)': 80,
        'Moderado (46-90 min)': 130,
        'Alto (91-180 min)': 190,
        'Muy Alto (>180 min)': 260
    }
    
    # Mapeamos la categoría a su tamaño correspondiente
    df_plot['TamanoBurbuja'] = df_plot['RangoRetraso'].map(tamanos_mapa)

    # --- CONSTRUCCIÓN DEL GRÁFICO 2D ---
    fig, ax = plt.subplots(figsize=(11, 7))

    # Graficamos iterando por grupo de distancia (colores) para armar la leyenda principal
    for etiqueta in etiquetas_distancia:
        subset = df_plot[df_plot["RangoDistancia"] == etiqueta]
        
        # Evitamos plotear si la categoría está vacía en este dataset
        if subset.empty:
            continue

        ax.scatter(
            subset["CRSDepTime"],
            subset["CRSArrTime"],
            s=subset["TamanoBurbuja"],       # El tamaño viene de los 6 rangos discretos
            color=colores_mapa[etiqueta],    
            label=etiqueta,
            alpha=0.5,
            edgecolors="white",
            linewidths=0.5,
        )

    # --- FORMATEO Y CONFIGURACIÓN ACADÉMICA ---
    ax.set_title(
        "Análisis Multivariable de Vuelos en EE.UU.\n(Horarios, Distancia por Color y Retraso por Tamaño)",
        fontsize=12,
        pad=15,
    )
    ax.set_xlabel("Hora Programada de Salida (CRSDepTime)", fontsize=10)
    ax.set_ylabel("Hora Programada de Llegada (CRSArrTime)", fontsize=10)

    # 1. Leyenda Principal: Clasificación de Distancias (Colores)
    leyenda_colores = ax.legend(
        title="Distancia del Vuelo", loc="upper left", bbox_to_anchor=(1, 1)
    )
    ax.add_artist(leyenda_colores)

    # 2. Leyenda Secundaria: Guía de los 6 Niveles de Retraso (Tamaños)
    lineas_tamano = [
        ax.scatter([], [], s=tamanos_mapa[etiq], color="gray", alpha=0.5, edgecolors="white", linewidths=0.5)
        for etiq in etiquetas_retraso
    ]

    ax.legend(
        lineas_tamano,
        etiquetas_retraso,
        title="Nivel de Retraso (Salida)",
        loc="lower left",
        bbox_to_anchor=(1, 0),
        labelspacing=1.2, # Un poco más compacto para acomodar las 6 etiquetas
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig("test_2d_gemini_03.png", dpi=DPI, bbox_inches="tight")
    plt.close()

def heatmap_day_vs_hour(df: pd.DataFrame):
    temp = df.copy()
    temp["Hour"] = (temp["CRSDepTime"] // 60).astype(int)
    
    # Agrupar por hora y día de la semana
    heatmap_data = temp.groupby(["Hour", "DayOfWeek"])["DepDelay"].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False)
    plt.xlabel("Día de la Semana")
    plt.ylabel("Hora de salida programada")
    plt.title("Mapa de Calor: Retraso Promedio por Hora y Día de la Semana")
    plt.savefig("./graficas/heatmap_day_hour.png", dpi=DPI)
    plt.close()

def mean_median_delay_by_hour(df):
    temp = df.copy()
    temp["Hour"] = (temp["CRSDepTime"] // 60).astype(int)

    stats = temp.groupby("Hour")["DepDelay"].agg(
        Mean="mean",
        Median="median"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        stats.index,
        stats["Mean"],
        marker="o",
        label="Media"
    )

    plt.plot(
        stats.index,
        stats["Median"],
        marker="o",
        label="Mediana"
    )

    plt.xlabel("Hora de salida programada")
    plt.ylabel("Retraso (minutos)")
    plt.title("Media y mediana del retraso según hora programada")
    plt.xticks(range(25))
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        "./graficas/mean_median_delay_by_hour.png",
        dpi=DPI
    )
    plt.close()

def heatmap_hour_day(df):

    temp = df.copy()
    temp["Hour"] = (temp["CRSDepTime"] // 60).astype(int)

    pivot = temp.pivot_table(
        values="DepDelay",
        index="DayOfWeek",
        columns="Hour",
        aggfunc="mean"
    )

    plt.figure(figsize=(14, 6))

    sns.heatmap(
        pivot,
        cmap="viridis"
    )

    plt.xlabel("Hora de salida programada")
    plt.ylabel("Día de la semana")
    plt.title("Retraso promedio según día de la semana y hora")

    plt.tight_layout()
    plt.savefig(
        "./graficas/heatmap_hour_day.png",
        dpi=DPI
    )
    plt.close()

def delay_vs_distance_hexbin(df):

    plt.figure(figsize=(10, 6))

    plt.hexbin(
        df["Distance"],
        df["DepDelay"],
        gridsize=60,
        mincnt=1
    )

    plt.colorbar(label="Cantidad de vuelos")

    plt.xlabel("Distancia (millas)")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Distribución de retrasos según distancia del vuelo")

    plt.tight_layout()
    plt.savefig(
        "./graficas/delay_vs_distance_hexbin.png",
        dpi=DPI
    )
    plt.close()


def flight_distribution_by_hour(df: pd.DataFrame):
    temp = df.copy()

    # Convertimos la hora programada a hora del día
    temp["Hour"] = (temp["CRSDepTime"] // 60).astype(int)

    hourly_count = temp["Hour"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))

    plt.bar(
        hourly_count.index,
        hourly_count.values
    )

    plt.xlabel("Hora de salida programada")
    plt.ylabel("Cantidad de vuelos")
    plt.title("Distribución de vuelos según hora de salida")
    plt.xticks(range(24))
    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(
        "./graficas/flight_distribution_by_hour.png",
        dpi=DPI
    )
    plt.close()

def flight_distribution_by_weekday(df: pd.DataFrame):

    weekly_count = (
        df["DayOfWeek"]
        .value_counts()
        .sort_index()
    )

    days = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo"
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        weekly_count.index,
        weekly_count.values
    )

    plt.xlabel("Día de la semana")
    plt.ylabel("Cantidad de vuelos")
    plt.title("Distribución de vuelos según día de la semana")

    plt.xticks(
        range(1, 8),
        days
    )

    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(
        "./graficas/flight_distribution_by_weekday.png",
        dpi=DPI
    )
    plt.close()


def flight_distribution_by_month(df: pd.DataFrame):

    monthly_count = (
        df["Month"]
        .value_counts()
        .sort_index()
    )

    months = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre"
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        monthly_count.index,
        monthly_count.values
    )

    plt.xlabel("Mes")
    plt.ylabel("Cantidad de vuelos")
    plt.title("Distribución de vuelos según mes")

    plt.xticks(
        range(1, 13),
        months,
        rotation=45
    )

    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(
        "./graficas/flight_distribution_by_month.png",
        dpi=DPI
    )
    plt.close()

def delay_distribution_by_hour(df: pd.DataFrame):
    temp = df.copy()

    # Convertir minutos desde medianoche a hora
    temp["Hour"] = (temp["CRSDepTime"] // 60).astype(int)

    fig, axes = plt.subplots(
        4,
        6,
        figsize=(16, 10),
        sharex=True,
        sharey=True
    )

    for hour, ax in enumerate(axes.flat):

        delays = temp.loc[
            temp["Hour"] == hour,
            "DepDelay"
        ]

        ax.hist(
            delays,
            bins=50,
            range=(-30, 150)
        )

        ax.set_title(f"{hour:02d}:00")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Distribución de retrasos según hora de salida",
        fontsize=16
    )

    fig.supxlabel("Retraso en la salida (minutos)")
    fig.supylabel("Frecuencia")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(
        "./graficas/delay_distribution_by_hour.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

def delay_distribution_by_weekday(df: pd.DataFrame):

    days = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo"
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
        sharex=True,
        sharey=True
    )

    for day in range(1, 8):

        ax = axes.flat[day - 1]

        delays = df.loc[
            df["DayOfWeek"] == day,
            "DepDelay"
        ]

        ax.hist(
            delays,
            bins=50,
            range=(-30, 150)
        )

        ax.set_title(days[day - 1])
        ax.grid(axis="y", alpha=0.3)

    # Ocultar el octavo subplot
    axes.flat[7].axis("off")

    fig.suptitle(
        "Distribución de retrasos según día de la semana",
        fontsize=16
    )

    fig.supxlabel("Retraso en la salida (minutos)")
    fig.supylabel("Frecuencia")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        "./graficas/delay_distribution_by_weekday.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

def delay_distribution_by_month(df: pd.DataFrame):

    months = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre"
    ]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14, 9),
        sharex=True,
        sharey=True
    )

    for month in range(1, 13):

        ax = axes.flat[month - 1]

        delays = df.loc[
            df["Month"] == month,
            "DepDelay"
        ]

        ax.hist(
            delays,
            bins=50,
            range=(-30, 150)
        )

        ax.set_title(months[month - 1])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Distribución de retrasos según mes",
        fontsize=16
    )

    fig.supxlabel("Retraso en la salida (minutos)")
    fig.supylabel("Frecuencia")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        "./graficas/delay_distribution_by_month.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

def delay_boxplot_by_month(df: pd.DataFrame):

    months = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre"
    ]

    plt.figure(figsize=(12, 7))

    df.boxplot(
        column="DepDelay",
        by="Month"
    )

    plt.xlabel("Mes")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Distribución de retrasos por mes")
    plt.suptitle("")

    plt.xticks(
        range(1, 13),
        months,
        rotation=45
    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "./graficas/depdelay_boxplot_by_month.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

def delay_boxplot_by_weekday(df: pd.DataFrame):

    days = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo"
    ]

    plt.figure(figsize=(10, 7))

    df.boxplot(
        column="DepDelay",
        by="DayOfWeek"
    )

    plt.xlabel("Día de la semana")
    plt.ylabel("Retraso en la salida (minutos)")
    plt.title("Distribución de retrasos por día de la semana")
    plt.suptitle("")

    plt.xticks(
        range(1, 8),
        days
    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "./graficas/depdelay_boxplot_by_weekday.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

def predicted_vs_actual(y_real: list, y_pred: list, r2: float, dataset_name="Test"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, y_pred, s=1, alpha=0.15)
    
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