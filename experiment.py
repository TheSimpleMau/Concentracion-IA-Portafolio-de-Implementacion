from etlProcess import extract_data, transform_data, split_data
from preprocessing import prepare_model_data

from regresionModel import run_regression_model
from randomForestModel import run_random_forest
from xgboostModel import run_xgboost

from evaluation import evaluate_all_splits, create_comparison_dataframe, create_diagnostic_dataframe, print_diagnostic_summary

from experimentStorage import save_experiment

from plotUtil import (
    mean_delay_by_hour,
    mean_delay_by_month,
    mean_delay_by_week,
    correlation_matrix,
    predicted_vs_actual,
    residual_plot,
    error_distribution,
    plot_training_history,
    rf_cost_evolution,
    xgb_cost_evolution,
    compare_models_bar,
    binned_prediction_plot
)


def initial_graphs(df):
    mean_delay_by_hour(df)
    mean_delay_by_month(df)
    mean_delay_by_week(df)
    correlation_matrix(df)


def run_experiment():

    config = {
        "sample_fraction": 0.001,
        "shuffle_data": True,
        "random_state": 42,
        "evaluate_test": False,

        "run_manual": False,
        "run_random_forest": False,
        "run_xgboost": True,

        "generate_basic_graphs": False,
        "generate_individual_graphs": True,
        "generate_comparison_graphs": False,

        "manual": {
            "learning_rate": 0.01,
            "epochs": 20,
            "batch_size": 2048,
            "random_state": 42
        },

        "random_forest": {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_leaf": 10,
            "min_samples_split": 20,
            "max_features": 0.7,
            "bootstrap": True,
            "max_samples": 2_000_000,
            "n_jobs": -1,
            "random_state": 42,
            "criterion": "squared_error",
        },

        "xgboost": {
            "n_estimators": 400,
            "max_depth": 7,
            "learning_rate": 0.125,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 80,
            "reg_lambda": 50,
            "gamma": 0.0,
            "tree_method": "hist",
            "max_bin": 256,
            "eval_metric": "rmse",
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }
    }

    sample_fraction = config["sample_fraction"]
    evaluation_split = "test" if config["evaluate_test"] else "val"

    if config["evaluate_test"] and sample_fraction != 1.0:
        raise ValueError(
            "Test sólo puede evaluarse con sample_fraction=1.0."
        )

    print("\n" + "=" * 70)
    print("ETL")
    print("=" * 70)

    df, skip_transform = extract_data()

    if not skip_transform:
        df = transform_data(df)

    df, train_df, val_df, test_df = split_data(
        df,
        shuffle=config["shuffle_data"],
        random_state=config["random_state"],
    )

    print("\nDatos separados:")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")

    print(f"\nTamaño efectivo del entrenamiento piloto ({config['sample_fraction'] * 100}%) :")
    print(f"Train:      {int(len(train_df) * config["sample_fraction"]):,}")
    print(f"Validation: {int(len(val_df) * config["sample_fraction"]):,} (completo)")
    print(f"Test:       {int(len(test_df) * config["sample_fraction"]):,} (completo)")

    if config["generate_basic_graphs"]:
        print("\nGenerando gráficos iniciales...")
        initial_graphs(df)
        print("Gráficos iniciales terminados.")

    model_results = {}
    model_predictions = {}
    model_histories = {}
    trained_models = {}
    fitted_preprocessors = {}

    if config["run_manual"]:

        print("\n" + "=" * 70)
        print("REGRESIÓN MANUAL")
        print("=" * 70)

        X_train, y_train, X_val, y_val, X_test, y_test = prepare_model_data(train_df,val_df,test_df,
                                                                            sample_fraction=sample_fraction,
                                                                            drop_columns=["DepDelay","UniqueCarrier","Origin","Dest"],
                                                                            normalize=True,
                                                                            random_state=config["random_state"])

        model_X_test = X_test if config["evaluate_test"] else None
        model_y_test = y_test if config["evaluate_test"] else None
        manual_model, manual_predictions, manual_history = run_regression_model(X_train,y_train,X_val,y_val,model_X_test,model_y_test,**config["manual"])
        manual_metrics = evaluate_all_splits(
            y_train,
            manual_predictions["train"],
            y_val,
            manual_predictions["val"],
            model_y_test,
            manual_predictions.get("test"),
        )
        trained_models["Manual"] = manual_model
        model_results["Manual"] = manual_metrics
        model_predictions["Manual"] = manual_predictions
        model_histories["Manual"] = manual_history

        if config["generate_individual_graphs"]:
            y_evaluation = model_y_test if config["evaluate_test"] else y_val
            dataset_name = f"Manual_{evaluation_split.title()}"
            predicted_vs_actual(y_evaluation,manual_predictions[evaluation_split],manual_metrics[f"{evaluation_split}_r2"],dataset_name=dataset_name)
            residual_plot(y_evaluation,manual_predictions[evaluation_split],dataset_name=dataset_name)
            error_distribution(y_evaluation,manual_predictions[evaluation_split],dataset_name=dataset_name)
            plot_training_history(manual_history["epoch"],manual_history["train_mse"],manual_history["val_mse"],
                                xlabel="Época",
                                ylabel="MSE",
                                title="Evolución del MSE — Modelo Manual",
                                filename="manual_learning_curve")

    if config["run_random_forest"]:

        print("\n" + "=" * 70)
        print("RANDOM FOREST")
        print("=" * 70)

        X_train, y_train, X_val, y_val, X_test, y_test, rf_preprocessor = prepare_model_data(train_df,val_df,test_df,
                                                                            sample_fraction=sample_fraction,
                                                                            drop_columns=["DepDelay"],
                                                                            encode_categorical=True,
                                                                            normalize=False,
                                                                            random_state=config["random_state"],
                                                                            return_preprocessor=True)

        
        model_X_test = X_test if config["evaluate_test"] else None
        model_y_test = y_test if config["evaluate_test"] else None
        rf_model, rf_predictions, rf_history = run_random_forest(X_train, y_train, X_val, y_val, model_X_test, model_y_test, **config["random_forest"])

        rf_metrics = evaluate_all_splits(
            y_train,
            rf_predictions["train"],
            y_val,
            rf_predictions["val"],
            model_y_test,
            rf_predictions.get("test"),
        )

        trained_models["Random Forest"] = rf_model
        model_results["Random Forest"] = rf_metrics
        model_predictions["Random Forest"] = rf_predictions
        model_histories["Random Forest"] = rf_history
        fitted_preprocessors["Random Forest"] = rf_preprocessor

        if config["generate_individual_graphs"]:
            y_evaluation = model_y_test if config["evaluate_test"] else y_val
            dataset_name = f"RandomForest_{evaluation_split.title()}"
            predicted_vs_actual(y_evaluation, rf_predictions[evaluation_split], rf_metrics[f"{evaluation_split}_r2"], dataset_name=dataset_name)
            residual_plot(y_evaluation, rf_predictions[evaluation_split], dataset_name=dataset_name)
            error_distribution(y_evaluation, rf_predictions[evaluation_split], dataset_name=dataset_name)
            rf_cost_evolution(rf_history["train_mse"], rf_history["val_mse"], rf_history["estimators"])

    if config["run_xgboost"]:

        print("\n" + "=" * 70)
        print("XGBOOST")
        print("=" * 70)

        X_train, y_train, X_val, y_val, X_test, y_test, xgb_preprocessor = prepare_model_data(train_df, val_df, test_df,
                                                                            sample_fraction=sample_fraction,
                                                                            drop_columns=["DepDelay"],
                                                                            encode_categorical=True,
                                                                            normalize=False,
                                                                            random_state=config["random_state"],
                                                                            return_preprocessor=True)

        model_X_test = X_test if config["evaluate_test"] else None
        model_y_test = y_test if config["evaluate_test"] else None
        xgb_model, xgb_predictions, xgb_history = run_xgboost(X_train, y_train, X_val, y_val, model_X_test, model_y_test, **config["xgboost"])

        xgb_metrics = evaluate_all_splits(
            y_train,
            xgb_predictions["train"],
            y_val,
            xgb_predictions["val"],
            model_y_test,
            xgb_predictions.get("test"),
        )

        trained_models["XGBoost"] = xgb_model
        model_results["XGBoost"] = xgb_metrics
        model_predictions["XGBoost"] = xgb_predictions
        model_histories["XGBoost"] = xgb_history
        fitted_preprocessors["XGBoost"] = xgb_preprocessor

        if config["generate_individual_graphs"]:
            y_evaluation = model_y_test if config["evaluate_test"] else y_val
            dataset_name = f"XGBoost_{evaluation_split.title()}"
            predicted_vs_actual(y_evaluation, xgb_predictions[evaluation_split], xgb_metrics[f"{evaluation_split}_r2"], dataset_name=dataset_name)
            residual_plot(y_evaluation, xgb_predictions[evaluation_split], dataset_name=dataset_name)
            error_distribution(y_evaluation, xgb_predictions[evaluation_split], dataset_name=dataset_name)
            xgb_cost_evolution(xgb_history["train_rmse"], xgb_history["val_rmse"])

    if not model_results:

        print("\nNo se ejecutó ningún modelo.")
        return

    comparison = create_comparison_dataframe(
        model_results,
        split=evaluation_split,
    )

    print("\n")
    print("=" * 90)
    print(f"COMPARACIÓN DE MODELOS — {evaluation_split.upper()}")
    print("=" * 90)

    print(comparison.to_string(index=False))

    diagnostic = create_diagnostic_dataframe(model_results)

    print_diagnostic_summary(model_results)

    if config["generate_comparison_graphs"]:

        compare_models_bar(model_results,f"{evaluation_split}_mae",f"Comparación de MAE — {evaluation_split.title()}")
        compare_models_bar(model_results,f"{evaluation_split}_rmse",f"Comparación de RMSE — {evaluation_split.title()}")

        compare_models_bar(model_results,f"{evaluation_split}_r2",f"Comparación de R² — {evaluation_split.title()}")

        compare_models_bar(model_results,f"{evaluation_split}_within_15","Predicciones dentro de ±15 minutos")

        predictions_for_plot = {
            model_name: predictions[evaluation_split]
            for model_name, predictions
            in model_predictions.items()
        }

        first_model = next(iter(model_predictions))
        y_evaluation = model_predictions[first_model][f"y_{evaluation_split}"]

        binned_prediction_plot(predictions_for_plot,y_evaluation,n_bins=20,dataset_name=evaluation_split.title())

    experiment_data = {
        "config": config,
        "metrics": model_results,
        "histories": model_histories,
        "comparison": comparison,
        "diagnostic": diagnostic,
        "models": trained_models,
        "preprocessors": fitted_preprocessors
    }

    save_experiment(experiment_data)

    print("\nExperimento terminado.")
