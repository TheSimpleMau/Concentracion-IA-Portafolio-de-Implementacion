import pandas as pd

TARGET_COLUMN = "DepDelay"
CATEGORICAL_COLUMNS = [
    "UniqueCarrier",
    "Origin",
    "Dest",
]


def _validate_sample_fraction(sample_fraction):
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction debe estar entre 0 y 1.")


def _sample_dataframe(df: pd.DataFrame, sample_fraction: float):
    samples = int(len(df) * sample_fraction)
    if samples <= 0:
        raise ValueError("sample_fraction produce 0 observaciones.")
    # return df.sample(frac=sample_fraction, random_state=42)
    return df


def _fit_one_hot_encoding(train_df: pd.DataFrame, categorical_columns):
    encoded_train = pd.get_dummies(train_df,columns=categorical_columns,dtype="float32")
    return encoded_train


def _apply_one_hot_encoding(df: pd.DataFrame, train_columns,categorical_columns):
    encoded = pd.get_dummies(df, columns=categorical_columns, dtype="float32")
    encoded = encoded.reindex(columns=train_columns)
    return encoded


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
                    normalize: bool = False):

    _validate_sample_fraction(sample_fraction)

    drop_columns = drop_columns or []
    train_df = _sample_dataframe(train_df,sample_fraction)
    val_df = _sample_dataframe(val_df,sample_fraction)
    test_df = _sample_dataframe(test_df,sample_fraction)

    y_train = train_df[TARGET_COLUMN].copy()
    y_val = val_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    X_train = train_df.drop(columns=drop_columns)
    X_val = val_df.drop(columns=drop_columns)
    X_test = test_df.drop(columns=drop_columns)

    if encode_categorical:
        categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in X_train.columns]

        if categorical_columns:
            X_train = _fit_one_hot_encoding(X_train,categorical_columns)
            train_columns = X_train.columns
            X_val = _apply_one_hot_encoding(X_val, train_columns, categorical_columns)
            X_test = _apply_one_hot_encoding(X_test, train_columns, categorical_columns)

    if normalize:
        X_train, X_val, X_test = _normalize(X_train, X_val, X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test