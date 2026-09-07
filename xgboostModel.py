from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=1000, early_stopping_rounds=30, **xgb_kwargs):
    params = {
        "n_estimators": n_estimators,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42
    }
    params.update(xgb_kwargs)

    model = XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds)
    model.fit(X_train, y_train, eval_set=[ (X_train, y_train), (X_val, y_val)], verbose=False)
    results = model.evals_result()
    history = {
        "iteration": list(
            range(
                1,
                len(
                    results[
                        "validation_0"
                    ]["rmse"]
                ) + 1
            )
        ),

        "train_rmse": results["validation_0"]["rmse"],
        "val_rmse": results["validation_1"]["rmse"],
        "best_iteration": getattr(model,"best_iteration",None)
    }

    return model, history


def predict(model, X_train, X_val, X_test):
    return {
        "train": model.predict(X_train),
        "val": model.predict(X_val),
        "test": model.predict(X_test)
    }


def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, n_estimators=1000, early_stopping_rounds=30, **xgb_kwargs):
    model, history = train_xgboost(X_train, y_train, X_val, y_val,
                                n_estimators=n_estimators,
                                early_stopping_rounds=early_stopping_rounds, **xgb_kwargs)

    predictions = predict(model, X_train, X_val, X_test)
    predictions.update({
        "y_train": y_train.tolist(),
        "y_val": y_val.tolist(),
        "y_test": y_test.tolist()
    })

    return model, predictions, history