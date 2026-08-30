import os
import pandas as pd
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
        print("\nDatos generales:")
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
    df = pd.get_dummies(df, columns=["DayOfWeek", "DayofMonth", "Month"], dtype=int)
    print("One Hot Encoding aplicado.")

    print("Guardando datos en formato parquet para tener una carga más rápida en futuras pruebas...")
    df.to_parquet("airlines_10M.parquet", engine="pyarrow")
    print("Datos en parquet guardados.")

    return df

def split_data(df:pd.DataFrame):
    df = df.sample(frac=1)
    Xvalues = df.iloc[::, 1::] # Sabemos que después de la primera columna, todas las demás son las variables predictorias.
    Yvalues = df.iloc[::, 0] # Sabesmos que la primera columna es DepDelay.
    df_len = len(df)
    # Al ser un gran tamaño de datos, dividiré los datos 98/1/1
    Xtrain = Xvalues[0:(df_len*98)//100]
    Xvalidation = Xvalues[(df_len*98)//100:(df_len*99)//100]
    Xtest = Xvalues[(df_len*99)//100::]
    Xtrain = Xtrain.drop(columns=["Month", "DayofMonth", "DayOfWeek"])
    Xvalidation = Xvalidation.drop(columns=["Month", "DayofMonth", "DayOfWeek"])
    Xtest = Xtest.drop(columns=["Month", "DayofMonth", "DayOfWeek"])
    ytrain = Yvalues[0:(df_len*98)//100]
    yvalidation = Yvalues[(df_len*98)//100:(df_len*99)//100]
    ytest = Yvalues[(df_len*99)//100::]
    # Aunque el dataset al final lo separo, también regreso todo el dataset compelto para poder hacer un EDA
    # más profundo en otra sección.
    return df, Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest


def normalize_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    # Encontramos los mínimos y máximos en los datos de entrenamiento
    min_vals = X_train.min()
    max_vals = X_train.max()
    
    # Aplicamos la fórmula a los tres conjuntos usando los parámetros de train
    X_train_norm = (X_train - min_vals) / (max_vals - min_vals)
    X_val_norm = (X_val - min_vals) / (max_vals - min_vals)
    X_test_norm = (X_test - min_vals) / (max_vals - min_vals)
    
    return X_train_norm, X_val_norm, X_test_norm