import os
import pandas as pd
import numpy as np
from scipy.io import arff


RAW_DATA_PATH = "./airlines_train_regression_10000000.arff"
PARQUET_PATH = "airlines_10M.parquet"

TRAIN_FRACTION = 0.98
VAL_FRACTION = 0.01
TEST_FRACTION = 0.01


def hhmm_to_minutes(time):
    time = int(time)
    hours = time // 100
    minutes = time % 100
    return hours * 60 + minutes


def _decode_categorical(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

def extract_data():
    print("Cargando los datos...")

    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH,engine="pyarrow")
        print("\nDatos generales:")
        print(df.info())
        print("\nDatos ya transformados. Saltando transformación.")

        return df, True

    arff_data = arff.loadarff(RAW_DATA_PATH)
    df = pd.DataFrame(arff_data[0])
    print("\nEstado inicial del dataset:")
    print(df.info())

    return df, False


def transform_data(df: pd.DataFrame):
    df = df.copy()
    print("\nConvirtiendo CRSDepTime y CRSArrTime a minutos desde la medianoche...")

    df["CRSDepTime"] = (df["CRSDepTime"].apply(hhmm_to_minutes))
    df["CRSArrTime"] = (df["CRSArrTime"].apply(hhmm_to_minutes))

    categorical_columns = [
        "UniqueCarrier",
        "Origin",
        "Dest"
    ]
    
    cyclical_features = {
        "Month": 12.0,
        "DayofMonth": 31.0,
        "DayOfWeek": 7.0
    }

    for col, max_val in cyclical_features.items():
        if col in df.columns:
            df[col] = df[col].astype(float)
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
            df.drop(columns=[col], inplace=True)

    for column in categorical_columns:
        if column in df.columns:
            df[column] = (df[column].apply(_decode_categorical))

    print("Conversión realizada.")
    print("\nDatos generales:")
    print(df.info())
    print("\nEstadísticas:")
    print(df.describe())
    print("\nCantidad de valores nulos:")
    print(df.isna().sum())
    print("\nVariables categóricas conservadas:")
    print(categorical_columns)

    print("\nGuardando datos limpios en parquet...")
    df.to_parquet(PARQUET_PATH,engine="pyarrow")
    print("Datos en parquet guardados.")

    return df


def split_data(df: pd.DataFrame, train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION):
    total_fraction = (train_fraction + val_fraction + test_fraction)
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("Las proporciones de split deben sumar 1.")

    n = len(df)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return df, train_df, val_df, test_df