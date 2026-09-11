import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "DepDelay"
CATEGORICAL_COLUMNS = [
    "UniqueCarrier",
    "Origin",
    "Dest",
]
TIME_CATEGORICAL_COLUMNS = [
    "Month_cat",
    "DayofMonth_cat",
    "DayOfWeek_cat",
    "Departure_hour_bin",
    "Arrival_hour_bin",
]


def _validate_sample_fraction(sample_fraction):
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction debe estar entre 0 y 1.")


def _sample_dataframe(df: pd.DataFrame, sample_fraction: float, random_state: int):
    samples = int(len(df) * sample_fraction)
    if samples <= 0:
        raise ValueError("sample_fraction produce 0 observaciones.")
    if sample_fraction == 1:
        return df
    return df.sample(n=samples, random_state=random_state)


def _fit_one_hot_encoding(train_df: pd.DataFrame, categorical_columns):
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                categorical_columns,
            )
        ],
        remainder="passthrough",
        sparse_threshold=1.0,
    )
    encoded_train = encoder.fit_transform(train_df).astype(np.float32, copy=False)
    return encoded_train, encoder


def _apply_one_hot_encoding(df: pd.DataFrame, encoder):
    return encoder.transform(df).astype(np.float32, copy=False)


def _recover_cyclical_value(df: pd.DataFrame, name: str, period: int):
    sin_column = f"{name}_sin"
    cos_column = f"{name}_cos"
    if sin_column not in df.columns or cos_column not in df.columns:
        raise ValueError(f"Faltan {sin_column} y {cos_column}.")

    angle = np.mod(
        np.arctan2(
            df[sin_column].to_numpy(dtype=np.float64),
            df[cos_column].to_numpy(dtype=np.float64),
        ),
        2 * np.pi,
    )
    values = np.rint(angle * period / (2 * np.pi)).astype(np.int16)
    values[values == 0] = period
    if not np.isin(values, np.arange(1, period + 1)).all():
        raise ValueError(f"No se pudo reconstruir {name} dentro de 1..{period}.")
    return pd.Series(values, index=df.index, name=f"{name}_cat")


def _add_time_categorical_features(df: pd.DataFrame):
    result = df.copy()
    for name, period in {
        "Month": 12,
        "DayofMonth": 31,
        "DayOfWeek": 7,
    }.items():
        result[f"{name}_cat"] = _recover_cyclical_value(result, name, period)

    for source, column in (
        ("CRSDepTime", "Departure_hour_bin"),
        ("CRSArrTime", "Arrival_hour_bin"),
    ):
        if source not in result.columns:
            raise ValueError(f"Falta {source} para crear categorías temporales.")
        values = pd.to_numeric(result[source], errors="raise")
        result[column] = (values // 60).astype(np.int8)
    return result


def _normalize(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    numeric_columns = train_df.select_dtypes(include="number").columns

    for column in numeric_columns:
        train_min = train_df[column].min()
        train_max = train_df[column].max()
        denominator = (train_max - train_min)

        if denominator == 0:
            train_df[column] = 0.0
            val_df[column] = 0.0
            test_df[column] = 0.0
            continue

        train_df[column] = (train_df[column] - train_min) / denominator
        val_df[column] = (val_df[column] - train_min) / denominator
        test_df[column] = (test_df[column] - train_min) / denominator

    return train_df, val_df, test_df


def prepare_model_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    sample_fraction: float = 1.0, 
                    drop_columns=None, 
                    encode_categorical: bool = False, 
                    normalize: bool = False,
                    random_state: int = 42,
                    return_preprocessor: bool = False,
                    add_time_categories: bool = False):

    _validate_sample_fraction(sample_fraction)

    if encode_categorical and normalize:
        raise ValueError(
            "La normalización actual sólo acepta DataFrames densos; "
            "no debe combinarse con one-hot disperso."
        )
    if add_time_categories and not encode_categorical:
        raise ValueError(
            "Las categorías temporales requieren encode_categorical=True."
        )

    for split_name, split_df in {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }.items():
        if TARGET_COLUMN not in split_df.columns:
            raise ValueError(f"Falta {TARGET_COLUMN} en el conjunto {split_name}.")

    train_df = _sample_dataframe(train_df, sample_fraction, random_state)

    drop_columns = set(drop_columns or [])
    drop_columns.add(TARGET_COLUMN)
    preprocessor = None

    y_train = train_df[TARGET_COLUMN].copy()
    y_val = val_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    X_train = train_df.drop(columns=list(drop_columns), errors="ignore")
    X_val = val_df.drop(columns=list(drop_columns), errors="ignore")
    X_test = test_df.drop(columns=list(drop_columns), errors="ignore")

    if add_time_categories:
        X_train = _add_time_categorical_features(X_train)
        X_val = _add_time_categorical_features(X_val)
        X_test = _add_time_categorical_features(X_test)

    if encode_categorical:
        categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in X_train.columns]
        if add_time_categories:
            categorical_columns.extend(
                column for column in TIME_CATEGORICAL_COLUMNS if column in X_train.columns
            )

        if categorical_columns:
            X_train, preprocessor = _fit_one_hot_encoding(X_train, categorical_columns)
            X_val = _apply_one_hot_encoding(X_val, preprocessor)
            X_test = _apply_one_hot_encoding(X_test, preprocessor)

    if normalize:
        X_train, X_val, X_test = _normalize(X_train, X_val, X_test)

    result = (X_train, y_train, X_val, y_val, X_test, y_test)
    if return_preprocessor:
        return (*result, preprocessor)
    return result


def prepare_final_model_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             sample_fraction: float = 1.0,
                             drop_columns=None,
                             encode_categorical: bool = False,
                             normalize: bool = False,
                             random_state: int = 42,
                             return_preprocessor: bool = False,
                             add_time_categories: bool = False):
    _validate_sample_fraction(sample_fraction)
    if encode_categorical and normalize:
        raise ValueError(
            "La normalización actual sólo acepta DataFrames densos; "
            "no debe combinarse con one-hot disperso."
        )
    if add_time_categories and not encode_categorical:
        raise ValueError(
            "Las categorías temporales requieren encode_categorical=True."
        )
    for split_name, split_df in {"train": train_df, "test": test_df}.items():
        if TARGET_COLUMN not in split_df.columns:
            raise ValueError(f"Falta {TARGET_COLUMN} en el conjunto {split_name}.")

    train_df = _sample_dataframe(train_df, sample_fraction, random_state)
    drop_columns = set(drop_columns or [])
    drop_columns.add(TARGET_COLUMN)
    y_train = train_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()
    X_train = train_df.drop(columns=list(drop_columns), errors="ignore")
    X_test = test_df.drop(columns=list(drop_columns), errors="ignore")

    if add_time_categories:
        X_train = _add_time_categorical_features(X_train)
        X_test = _add_time_categorical_features(X_test)

    preprocessor = None
    if encode_categorical:
        categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in X_train.columns]
        if add_time_categories:
            categorical_columns.extend(
                column for column in TIME_CATEGORICAL_COLUMNS if column in X_train.columns
            )
        if categorical_columns:
            X_train, preprocessor = _fit_one_hot_encoding(X_train, categorical_columns)
            X_test = _apply_one_hot_encoding(X_test, preprocessor)

    if normalize:
        X_train, X_test, _ = _normalize(X_train, X_test, X_test)

    if return_preprocessor:
        return X_train, y_train, X_test, y_test, preprocessor
    return X_train, y_train, X_test, y_test
