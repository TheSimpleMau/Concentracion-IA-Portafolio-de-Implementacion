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

def transform_data(df: pd.DataFrame):

    print("\nConvirtiendo CRSDepTime y CRSArrTime a minutos desde la medianoche...")

    df["CRSDepTime"] = df["CRSDepTime"].apply(hhmm_to_minutes)
    df["CRSArrTime"] = df["CRSArrTime"].apply(hhmm_to_minutes)

    print("Conversión realizada.")

    print("\nDatos generales:")
    print(df.info())
    print(df.describe())

    print("\nCantidad de valores nulos:")
    print(df.isna().sum())

    # Algunas versiones del lector ARFF pueden devolver strings
    # como bytes. Los convertimos a strings normales.
    categorical_columns = [
        "UniqueCarrier",
        "Origin",
        "Dest"
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].astype(str)

    print("\nVariables categóricas conservadas:")
    print(categorical_columns)

    print(
        "\nNOTA: One-Hot Encoding y normalización "
        "serán realizados dentro de frameworkModel.py"
    )

    print("\nGuardando datos en formato parquet...")

    df.to_parquet(
        "airlines_10M.parquet",
        engine="pyarrow"
    )

    print("Datos en parquet guardados.")

    return df

def split_data(df: pd.DataFrame):

    # Usamos una semilla para que las pruebas sean reproducibles.
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    Yvalues = df["DepDelay"]

    # Solamente eliminamos el target.
    Xvalues = df.drop(
        columns=["DepDelay"]
    )

    df_len = len(df)

    # 98% train
    # 1% validation
    # 1% test

    train_end = (df_len * 98) // 100
    validation_end = (df_len * 99) // 100

    Xtrain = Xvalues[:train_end]
    Xvalidation = Xvalues[train_end:validation_end]
    Xtest = Xvalues[validation_end:]

    ytrain = Yvalues[:train_end]
    yvalidation = Yvalues[train_end:validation_end]
    ytest = Yvalues[validation_end:]

    return (
        df,
        Xtrain,
        Xvalidation,
        Xtest,
        ytrain,
        yvalidation,
        ytest
    )


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