import subprocess # ELIMINAR AL FINAL, SÓLO ES PARA USO EN DESARROLLO
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff # Para leer el tipo de archivo en el que viene el dataset
from plotUtil import *
from OneHotEnoding import OneHotEncoding

# FUNCION PARA SABER EL ESTADO DEL PROGRAMA, NO ES PARA USO FINAL EN DESARROLLO.
def program_status(message):
    subprocess.run(["say", "-r", "250", message])

def plot_graphs(df:pd.DataFrame):
    # scatter_matrix_plot(df[["CRSDepTime", "CRSArrTime", "Distance"]])
    # program_status("Comenzando con histograma de delays.")
    # delays_hist(df)
    # program_status("Comenzando con scatter plot de delays vs distance.")
    # delays_vs_distance(df)
    # program_status("Comenzando con scatter plot de delays vs deptime.")
    # delays_vs_deptime(df)
    # program_status("Comenzando con box plot de delays by month.")
    # delays_by_month(df)
    # program_status("Comenzando con box plot de delays by week.")
    # delays_by_week(df)
    # program_status("Comenzando con plot de promedio de retraso por hora.")
    # mean_delay_by_hour(df)
    # program_status("Comenzando con plot de promedio de retraso por mes.")
    # mean_delay_by_month(df)
    # program_status("Comenzando con matriz de correlación.")
    # correlation_matrix(df)
    # program_status("Comenzando con plot de promedio de retraso por día de la semana.")
    # mean_delay_by_week(df)
    program_status("Comenzando con pair plot.")
    pair_plot(df)



def delays_summary(df: pd.DataFrame):
    print("\nResumen de DepDelay:")
    percentiles = df["DepDelay"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    for percentile, value in percentiles.items():
        print(f"{percentile * 100:.0f}%: {value:.2f} minutos")

    print(f"Media: {df['DepDelay'].mean():.2f} minutos")
    print(f"Desviación estándar: {df['DepDelay'].std():.2f} minutos")
    print(f"Máximo retraso: {df['DepDelay'].max():.2f} minutos")
    print(f"Mínimo retraso: {df['DepDelay'].min():.2f} minutos")

def etl_process():
    # ETL
    ## Extracting
    program_status("Cargando datos...")
    print("Cargando los datos...")
    start = time.time()
    arff_file = arff.loadarff('./airlines_train_regression_10000000.arff')
    df = pd.DataFrame(arff_file[0])
    end = time.time()
    print(f"Datos cargados en {round(end-start, 2)} segundos.")
    program_status("Datos cargados.")
    print("\nDatos generales:")
    print(df.info())

    ## Transform
    print("\nExcluyendo columnas UniqueCarrier, Origin y Dest")

    df = df.drop(columns=["UniqueCarrier", "Origin", "Dest"])

    print("\nConvirtiendo CRSDepTime y CRSArrTime a minutos desde la medianoche...")

    def hhmm_to_minutes(time):
        time = int(time)
        hours = time // 100
        minutes = time % 100
        return hours * 60 + minutes

    df["CRSDepTime"] = df["CRSDepTime"].apply(hhmm_to_minutes)
    df["CRSArrTime"] = df["CRSArrTime"].apply(hhmm_to_minutes)

    print("Conversión realizada.")

    print("Datos generales:")

    print(df.info())
    print(df.describe())
    print("Cantidad de valores nulos:")
    print(df.isna().sum())
    print("Nota: No existen valores nulos :D")
    delays_summary(df)

    df = OneHotEncoding(df)
    

    # Loading
    df = df.sample(frac=1)
    # df_x = df.iloc[::, 1::]
    # df_y = df.iloc[::, 0]
    # Xvalues = df[["Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime", "Distance"]]
    # Yvalues = df[["DepDelay"]]
    Xvalues = df.iloc[::, 1::] # Sabemos que, después de la primera columna, todas las demas las utilizaresmos como features.
    Yvalues = df.iloc[::, 0] # Sabesmos que la primera columna es DepDelay.
    df_len = len(df)
    # Al ser un gran tamaño de datos, dividiré los datos 98/1/1
    Xtrain = Xvalues[0:(df_len*98)//100]
    Xvalidation = Xvalues[(df_len*98)//100:(df_len*99)//100]
    Xtest = Xvalues[(df_len*99)//100::]
    ytrain = Yvalues[0:(df_len*98)//100]
    yvalidation = Yvalues[(df_len*98)//100:(df_len*99)//100]
    ytest = Yvalues[(df_len*99)//100::]

    return df, Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest


def main():
    _plot_graphs = input("¿Desea generar gráficos a lo largo del programa? (y/n) ")
    _plot_graphs = True if plot_graphs == "y" else False
    program_status("Iniciando ETL...")
    df, Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest = etl_process()
    program_status("ETL Finalizado.")
    if _plot_graphs:
        program_status("Iniciando gráficos...")
        plot_graphs(df)
        program_status("Gráficos terminados.")
    print(df)
    program_status("Programa finalizado.")



if __name__ == '__main__':
    main()