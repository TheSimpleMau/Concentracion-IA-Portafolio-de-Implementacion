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

    # ========================================================
    # Configuración
    # ========================================================

    config = {
        "sample_fraction": 0.25,

        "run_manual": False,
        "run_random_forest": True,
        "run_xgboost": True,

        "generate_basic_graphs": False,
        "generate_individual_graphs": True,
        "generate_comparison_graphs": True,

        "manual": {
            "learning_rate": 0.01,
            "epochs": 20,
            "batch_size": 2048
        },

        "random_forest": {
            "n_estimators":50,
            "max_depth":12,
            "min_samples_leaf":20,
            "min_samples_split":40,
            "max_features":0.5,
            "bootstrap":True,
            "max_samples":0.5,
            "n_jobs":2,
            "random_state":42,
            "criterion":"squared_error",
            # "n_estimators": 50,
            # "max_depth": 20,
            # "min_samples_split": 10,
            # "min_samples_leaf": 10,
            # "max_features": 0.7,
            # "bootstrap": True,
            # "max_samples": 0.7,
            # "n_jobs": -1,
            # "random_state": 42,
            # "criterion": "squared_error"
        },

        "xgboost": {
            "n_estimators": 50,
            "max_depth": 20,
            "learning_rate": 0.001,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 100
        }
    }

    sample_fraction = config["sample_fraction"]

    # ========================================================
    # ETL
    # ========================================================

    print("\n" + "=" * 70)
    print("ETL")
    print("=" * 70)

    df, skip_transform = extract_data()

    if not skip_transform:
        df = transform_data(df)

    df, train_df, val_df, test_df = split_data(df)

    print("\nDatos separados:")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")

    print(f"\nProporcion separada ({config["sample_fraction"] * 100}%) :")
    print(f"Train:      {len(train_df) * config["sample_fraction"]:,}")
    print(f"Validation: {len(val_df) * config["sample_fraction"]:,}")
    print(f"Test:       {len(test_df) * config["sample_fraction"]:,}")

    # ========================================================
    # Gráficas básicas
    # ========================================================

    if config["generate_basic_graphs"]:
        print("\nGenerando gráficos iniciales...")
        initial_graphs(df)
        print("Gráficos iniciales terminados.")

    # ========================================================
    # Resultados
    # ========================================================

    model_results = {}
    model_predictions = {}
    model_histories = {}
    trained_models = {}

    # ========================================================
    # Modelo manual
    # ========================================================

    if config["run_manual"]:

        print("\n" + "=" * 70)
        print("REGRESIÓN MANUAL")
        print("=" * 70)

        X_train, y_train, X_val, y_val, X_test, y_test = prepare_model_data(train_df,val_df,test_df,
                                                                            sample_fraction=sample_fraction,
                                                                            drop_columns=["DepDelay","UniqueCarrier","Origin","Dest"],
                                                                            normalize=True)

        manual_model, manual_predictions, manual_history = run_regression_model(X_train,y_train,X_val,y_val,X_test,y_test,**config["manual"])
        manual_metrics = evaluate_all_splits(y_train,manual_predictions["train"],y_val,manual_predictions["val"],y_test,manual_predictions["test"])
        trained_models["Manual"] = manual_model
        model_results["Manual"] = manual_metrics
        model_predictions["Manual"] = manual_predictions
        model_histories["Manual"] = manual_history

        if config["generate_individual_graphs"]:
            predicted_vs_actual(y_test,manual_predictions["test"],manual_metrics["test_r2"],dataset_name="Manual_Test")
            residual_plot(y_test,manual_predictions["test"],dataset_name="Manual_Test")
            error_distribution(y_test,manual_predictions["test"],dataset_name="Manual_Test")
            plot_training_history(manual_history["epoch"],manual_history["train_mse"],manual_history["val_mse"],
                                xlabel="Época",
                                ylabel="MSE",
                                title="Evolución del MSE — Modelo Manual",
                                filename="manual_learning_curve")

    # ========================================================
    # Random forest
    # ========================================================

    if config["run_random_forest"]:

        print("\n" + "=" * 70)
        print("RANDOM FOREST")
        print("=" * 70)

        X_train, y_train, X_val, y_val, X_test, y_test = prepare_model_data(train_df,val_df,test_df,
                                                                            sample_fraction=sample_fraction,
                                                                            drop_columns=["DepDelay"],
                                                                            encode_categorical=True,
                                                                            normalize=False)

        
        rf_model, rf_predictions, rf_history = run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, **config["random_forest"])

        rf_metrics = evaluate_all_splits(y_train, rf_predictions["train"], y_val, rf_predictions["val"], y_test,rf_predictions["test"])

        trained_models["Random Forest"] = rf_model
        model_results["Random Forest"] = rf_metrics
        model_predictions["Random Forest"] = rf_predictions
        model_histories["Random Forest"] = rf_history

        if config["generate_individual_graphs"]:
            predicted_vs_actual(y_test, rf_predictions["test"], rf_metrics["test_r2"], dataset_name="RandomForest_Test")
            residual_plot(y_test, rf_predictions["test"], dataset_name="RandomForest_Test")
            error_distribution(y_test, rf_predictions["test"], dataset_name="RandomForest_Test")
            rf_cost_evolution(rf_history["train_mse"], rf_history["val_mse"], rf_history["estimators"])

    # ========================================================
    # XGBoost
    # ========================================================

    if config["run_xgboost"]:

        print("\n" + "=" * 70)
        print("XGBOOST")
        print("=" * 70)

        X_train, y_train, X_val, y_val, X_test, y_test = prepare_model_data(train_df, val_df, test_df,
                                                                            sample_fraction=sample_fraction,
                                                                            drop_columns=["DepDelay"],
                                                                            encode_categorical=True,
                                                                            normalize=False)

        xgb_model, xgb_predictions, xgb_history = run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, **config["xgboost"])

        xgb_metrics = evaluate_all_splits(y_train, xgb_predictions["train"], y_val, xgb_predictions["val"], y_test, xgb_predictions["test"])

        trained_models["XGBoost"] = xgb_model
        model_results["XGBoost"] = xgb_metrics
        model_predictions["XGBoost"] = xgb_predictions
        model_histories["XGBoost"] = xgb_history

        if config["generate_individual_graphs"]:
            predicted_vs_actual(y_test, xgb_predictions["test"], xgb_metrics["test_r2"], dataset_name="XGBoost_Test")
            residual_plot(y_test, xgb_predictions["test"], dataset_name="XGBoost_Test")
            error_distribution(y_test, xgb_predictions["test"], dataset_name="XGBoost_Test")
            xgb_cost_evolution(xgb_history["train_rmse"], xgb_history["val_rmse"], xgb_history["best_iteration"])

    # ========================================================
    # Validación
    # ========================================================

    if not model_results:

        print("\nNo se ejecutó ningún modelo.")
        return

    # ========================================================
    # Comparación
    # ========================================================

    comparison = create_comparison_dataframe(
        model_results
    )

    print("\n")
    print("=" * 90)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 90)

    print(comparison.to_string(index=False))

    # ========================================================
    # Diagnóstico validation / test
    # ========================================================

    diagnostic = create_diagnostic_dataframe(model_results)

    print_diagnostic_summary(model_results)

    # ========================================================
    # Gráficas comparativas
    # ========================================================

    if config["generate_comparison_graphs"]:

        compare_models_bar(model_results,"test_mae","Comparación de MAE — Test")
        compare_models_bar(model_results,"test_rmse","Comparación de RMSE — Test")

        compare_models_bar(model_results,"test_r2","Comparación de R² — Test")

        compare_models_bar(model_results,"test_within_15","Predicciones dentro de ±15 minutos")

        predictions_for_plot = {
            model_name: predictions["test"]
            for model_name, predictions
            in model_predictions.items()
        }

        first_model = next(iter(model_predictions))
        y_test = model_predictions[first_model]["y_test"]

        binned_prediction_plot(predictions_for_plot,y_test,n_bins=20,dataset_name="Test")

    # ========================================================
    # Guardar resultados
    # ========================================================

    experiment_data = {
        "config": config,
        "metrics": model_results,
        "histories": model_histories,
        "comparison": comparison,
        "diagnostic": diagnostic,
        "models": trained_models
    }

    save_experiment(experiment_data)

    print("\nExperimento terminado.")