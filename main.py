import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotUtil import *
from etlProcess import *
from regresionModel import train, predict, cost

def plot_graphs(df:pd.DataFrame):
    # delays_hist(df)
    # delays_vs_distance(df)
    # delays_vs_deptime(df)
    # delays_by_month(df)
    # delays_by_week(df)
    # mean_delay_by_hour(df)
    # mean_delay_by_month(df)
    correlation_matrix(df)
    # mean_delay_by_week(df)
    # pair_plot(df)
    # depArr_plot(df)
    # heatmap_day_vs_hour(df)
    # mean_median_delay_by_hour(df)
    # heatmap_hour_day(df)
    # delay_vs_distance_hexbin(df)
    # flight_distribution_by_hour(df)
    # flight_distribution_by_weekday(df)
    # flight_distribution_by_month(df)
    # delay_distribution_by_hour(df)
    # delay_distribution_by_weekday(df)
    # delay_distribution_by_month(df)
    # delay_boxplot_by_month(df)
    # delay_boxplot_by_weekday(df)

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
    print("Normalizando las columnas CRSDepTime, CRSArrTime y Distance...")
    Xtrain_norm, Xvalidation_norm, Xtest_norm = normalize_data(Xtrain, Xvalidation, Xtest)

    return df, Xtrain_norm, Xvalidation_norm, Xtest_norm, ytrain, yvalidation, ytest


def main():
    # ETL
    df, Xtrain_norm, Xvalidation_norm, Xtest_norm, ytrain, yvalidation, ytest = etl_process()


    # Datos a utilizar.
    # Nota: Esto sólo es para realizar pruebas rápidas por el tamaño del dataset.
    # Al final, data_percentage debe de valer 1.0 = 100% de los datos.
    data_percentage = 1.0

    train_samples = int(len(Xtrain_norm) * data_percentage)
    validation_samples = int(len(Xvalidation_norm) * data_percentage)
    test_samples = int(len(Xtest_norm) * data_percentage)

    print("\nPreparando datos para el modelo...")

    params = Xtrain_norm.iloc[:train_samples].values.tolist()
    y = ytrain.iloc[:train_samples].values.tolist()

    validation_params = Xvalidation_norm.iloc[:validation_samples].values.tolist()
    validation_y = yvalidation.iloc[:validation_samples].values.tolist()

    test_params = Xtest_norm.iloc[:test_samples].values.tolist()
    test_y = ytest.iloc[:test_samples].values.tolist()

    print("Porcentaje de datos utilizado:", data_percentage * 100, "%")
    print("Número de observaciones de entrenamiento:", len(params))
    print("Número de variables:", len(params[0]))
    print("Número de observaciones de validation:", len(validation_params))
    print("Número de observaciones de test:", len(test_params))

    # Parámetros del modelo
    lr = 0.001
    epochs = 200
    batch_size = 1024

    # Entrenamiento
    weights, bias, train_costs, validation_costs = train(params, y, validation_params, validation_y, lr, epochs, batch_size)

    # Resultados
    print("\nParámetros finales:")
    print("Weights:", weights)
    print("Bias:", bias)

    # Costos de train y validation

    print("\nCosto de entrenamiento:")
    print("Inicial:", train_costs[0])
    print("Final:", train_costs[-1])

    print("\nCosto de validation:")
    print("Inicial:", validation_costs[0])
    print("Final:", validation_costs[-1])

    # Evaluación final en test
    test_predictions = predict(test_params, weights, bias)
    test_cost = cost(test_predictions, test_y)

    print("\nCosto de test:")
    print(test_cost)


    # Gráfica de training y validation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_costs) + 1), train_costs, label="Training")
    plt.plot(range(1, len(validation_costs) + 1), validation_costs, label="Validation")
    plt.axhline(y=test_cost, linestyle="--", label="Test final")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Evolución del error durante el entrenamiento")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("cost_evolution.png", dpi=DPI)


if __name__ == '__main__':
    main()