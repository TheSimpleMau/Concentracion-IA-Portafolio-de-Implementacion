import pandas as pd
import numpy as np

def OneHotEncoding(df:pd.DataFrame)->pd.DataFrame:
    df_encoded = df.copy()

    # Month	DayOfMonth	DayOfWeek
    # Para DayOfWeek
    for i in range(1,8):
        df_encoded[f"Week_{i}"] = df_encoded["DayOfWeek"] == i
        df_encoded[f"Week_{i}"] = df_encoded[f"Week_{i}"].replace(True, 1)
        df_encoded[f"Week_{i}"] = df_encoded[f"Week_{i}"].replace(False, 0)
        # df_encoded[f"Week_{i}"] = df_encoded[f"Week_{i}"].astype("int32")
    # Para DayOfMonth
    for i in range(1,32):
        df_encoded[f"DayofMonth_{i}"] = df_encoded["DayofMonth"] == i
        df_encoded[f"DayofMonth_{i}"] = df_encoded[f"DayofMonth_{i}"].replace(True, 1)
        df_encoded[f"DayofMonth_{i}"] = df_encoded[f"DayofMonth_{i}"].replace(False, 0)
        # df_encoded[f"Week_{i}"] = df_encoded[f"Week_{i}"].astype("int32")
    # Para Month
    for i in range(1,13):
        df_encoded[f"Month_{i}"] = df_encoded["Month"] == i
        df_encoded[f"Month_{i}"] = df_encoded[f"Month_{i}"].replace(True, 1)
        df_encoded[f"Month_{i}"] = df_encoded[f"Month_{i}"].replace(False, 0)
        # df_encoded[f"Week_{i}"] = df_encoded[f"Week_{i}"].astype("int32")

    return df_encoded