from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=1000, **xgb_kwargs):
    params = {
        "n_estimators": n_estimators,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42
    }
    params.update(xgb_kwargs)

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[ (X_train, y_train), (X_val, y_val)], verbose=100)
    results = model.evals_result()
    history = {
        "iteration": list(range(1,len(results["validation_0"]["rmse"]) + 1)),
        "train_rmse": results["validation_0"]["rmse"],
        "val_rmse": results["validation_1"]["rmse"],
    }

    return model, history


def predict(model, X_train, X_val, X_test=None):
    predictions = {
        "train": model.predict(X_train),
        "val": model.predict(X_val),
    }
    if X_test is not None:
        predictions["test"] = model.predict(X_test)
    return predictions


def run_xgboost(X_train, y_train, X_val, y_val, X_test=None, y_test=None, n_estimators=1000, **xgb_kwargs):
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test y y_test deben proporcionarse juntos.")

    model, history = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=n_estimators,
        **xgb_kwargs,
    )

    predictions = predict(model, X_train, X_val, X_test)
    predictions.update({
        "y_train": y_train.tolist(),
        "y_val": y_val.tolist(),
    })
    if y_test is not None:
        predictions["y_test"] = y_test.tolist()

    return model, predictions, history
