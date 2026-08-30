import os
import pandas as pd
import numpy as np
from scipy.io import arff # Para leer el tipo de archivo en el que viene el dataset

def hhmm_to_minutes(time):
    time = int(time)
    hours = time // 100
    minutes = time % 100
    return hours * 60 + minutes

def extract_data():
    print("Cargando los datos...")
    df = None
    if "airlines_10M.parquet" not in os.listdir():
        arff_file = arff.loadarff('./airlines_train_regression_10000000.arff')
        df = pd.DataFrame(arff_file[0])
        print("\nEstado final del dataset:")
        print(df.info())
        return df, False
    else:
        df = pd.read_parquet("airlines_10M.parquet", engine="pyarrow")
        print("\nDatos generales:")
        print(df.info())
        print("Saltando transformación de los datos. Ya se han sido limpiados.")
        return df, True

def transform_data(df:pd.DataFrame):
    print("\nExcluyendo columnas UniqueCarrier, Origin y Dest")
    
    df = df.drop(columns=["UniqueCarrier", "Origin", "Dest"])

    print("\nConvirtiendo CRSDepTime y CRSArrTime a minutos desde la medianoche...")

    df["CRSDepTime"] = df["CRSDepTime"].apply(hhmm_to_minutes)
    df["CRSArrTime"] = df["CRSArrTime"].apply(hhmm_to_minutes)

    print("Conversión realizada.")

    print("Datos generales:")

    print(df.info())
    print(df.describe())
    print("Cantidad de valores nulos:")
    print(df.isna().sum())
    print("Nota: No existen valores nulos :D")

    print("Aplicando One Hot Encoding para columnas DayOfWeek, DayofMonth y Month...")
    dummies = pd.get_dummies(df[["DayOfWeek", "DayofMonth", "Month"]],columns=["DayOfWeek", "DayofMonth", "Month"],dtype=int)
    df = pd.concat([df, dummies], axis=1)
    print("One Hot Encoding aplicado.")


    print("Guardando datos en formato parquet para tener una carga más rápida en futuras pruebas...")
    df.to_parquet("airlines_10M.parquet", engine="pyarrow")
    print("Datos en parquet guardados.")

    return df

def split_data(df:pd.DataFrame):
    df = df.sample(frac=1).reset_index(drop=True)
    Yvalues = df["DepDelay"]
    Xvalues = df.drop(columns=["DepDelay", "DayOfWeek", "DayofMonth", "Month"])
    df_len = len(df)
    # Al ser un gran tamaño de datos, dividiré los datos 98/1/1
    Xtrain = Xvalues[0:(df_len*98)//100]
    Xvalidation = Xvalues[(df_len*98)//100:(df_len*99)//100]
    Xtest = Xvalues[(df_len*99)//100::]
    ytrain = Yvalues[0:(df_len*98)//100]
    yvalidation = Yvalues[(df_len*98)//100:(df_len*99)//100]
    ytest = Yvalues[(df_len*99)//100::]
    # Regreso también el dataset completo para realizar gráficas en caso de ser necesario
    return df, Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest


def normalize_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    columns_to_normalize = ["CRSDepTime", "CRSArrTime", "Distance"]
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    X_test_norm = X_test.copy()

    # Min y max calculados con train
    min_vals = X_train[columns_to_normalize].min()
    max_vals = X_train[columns_to_normalize].max()

    # Normalización
    X_train_norm[columns_to_normalize] = (X_train[columns_to_normalize] - min_vals) / (max_vals - min_vals)
    X_val_norm[columns_to_normalize]   = (X_val[columns_to_normalize]   - min_vals) / (max_vals - min_vals)
    X_test_norm[columns_to_normalize]  = (X_test[columns_to_normalize]  - min_vals) / (max_vals - min_vals)

    return X_train_norm, X_val_norm, X_test_norm