import subprocess # ELIMINAR AL FINAL, SÓLO ES PARA USO EN DESARROLLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotUtil import *
from etlProcess import *

def plot_graphs(df:pd.DataFrame):
    subprocess.run(["say","Iniciando graficas..."])
    delays_hist(df)
    delays_vs_distance(df)
    delays_vs_deptime(df)
    delays_by_month(df)
    delays_by_week(df)
    mean_delay_by_hour(df)
    mean_delay_by_month(df)
    correlation_matrix(df)
    mean_delay_by_week(df)
    # pair_plot(df)
    subprocess.run(["say", "graficas terminadas"])

def etl_process():
    # ETL
    ## Extracting
    
    df, _skip = extract_data()
    if not _skip:
        ## Transform
        df = transform_data(df)
    
    # Loading
    # Como tal, no se subirán los datos a algun repositorio u otro lugar, sin embargo
    # sirve realizar en esta sección la separación de los datos.
    df, Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest = split_data(df)

    print("Datos separados en train (98%), validation (1%) y test (1%)")

    return df, Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest


def eda_process(df:pd.DataFrame):
    print("Descripción general de los datos:")
    print(df.describe())




def main():
    df, Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest = etl_process()
    _plot_graphs = str(input("¿Desea generar gráficos? (y/n) "))
    if _plot_graphs == "y":
        plot_graphs(df)
    # _plot_graphs = True if plot_graphs == "y" else False # dejo de funcionar por algun motivo :/
    # if _plot_graphs:
        # plot_graphs(df)
    print(df)

    eda_process(df)



if __name__ == '__main__':
    main()
    subprocess.run(["say", "programa finalizado"])