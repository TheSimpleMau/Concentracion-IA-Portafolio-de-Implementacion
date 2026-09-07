from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=500, **rf_kwargs):
    params = {
        "n_estimators": 1,
        "max_depth": None,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "n_jobs": -1,
        "random_state": 42,
        "warm_start": True
    }
    params.update(rf_kwargs)

    model = RandomForestRegressor(**params)
    checkpoints = sorted(
        set(
            max(
                1,
                int(
                    n_estimators
                    * (i + 1)
                    / 5
                )
            )
            for i in range(5)
        )
    )

    train_mse_history = []
    val_mse_history = []

    for n in tqdm(checkpoints, desc="Random Forest"):
        model.n_estimators = n
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_mse_history.append(mean_squared_error(y_train, train_predictions))

        val_mse_history.append(mean_squared_error(y_val, val_predictions))

    return model, checkpoints, train_mse_history, val_mse_history


def predict(model, X_train, X_val, X_test):
    return {
        "train": model.predict(X_train),
        "val": model.predict(X_val),
        "test": model.predict(X_test)
    }


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, n_estimators=500, **rf_kwargs):
    model, estimators, train_mse_history, val_mse_history = train_random_forest(X_train, y_train, X_val, y_val,
                                                                                n_estimators=n_estimators,
                                                                                **rf_kwargs)

    predictions = predict(model, X_train, X_val, X_test)
    predictions.update({
        "y_train": y_train.tolist(),
        "y_val": y_val.tolist(),
        "y_test": y_test.tolist()
    })

    history = {
        "estimators": estimators,
        "train_mse": train_mse_history,
        "val_mse": val_mse_history
    }

    return model, predictions, history