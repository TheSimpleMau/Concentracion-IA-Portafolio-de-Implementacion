import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from etlProcess import extract_data, transform_data, split_data
from preprocessing import prepare_final_model_data, prepare_model_data
from regresionModel import run_regression_model
from xgboostModel import run_xgboost, train_final_xgboost
from evaluation import (
    create_comparison_dataframe,
    create_diagnostic_dataframe,
    evaluate_all_splits,
    print_diagnostic_summary,
    regression_metrics,
)
from experimentStorage import save_experiment
from plotUtil import (
    binned_prediction_plot,
    compare_models_bar,
    correlation_matrix,
    error_distribution,
    mean_delay_by_hour,
    mean_delay_by_month,
    mean_delay_by_week,
    plot_training_history,
    predicted_vs_actual,
    residual_plot,
    xgb_cost_evolution,
)


FINAL_COMPARISON_DIRECTORY = Path("resultados/evaluacion_final_xgboost")


def validate_final_comparison_config(config):
    expected_values = {
        "sample_fraction": 1.0,
        "shuffle_data": True,
        "random_state": 42,
    }
    expected_parameters = {
        "manual": {
            "learning_rate": 0.01,
            "epochs": 20,
            "batch_size": 2048,
            "random_state": 42,
        },
        "xgboost_base": {
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
        },
        "xgboost_improved": {
            "n_estimators": 400,
            "max_depth": 0,
            "max_leaves": 63,
            "grow_policy": "lossguide",
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
        },
    }
    errors = []
    for name, expected in expected_values.items():
        value = config.get(name)
        if value != expected:
            errors.append(f"{name}={value!r}; esperado {expected!r}.")
    for model_name, parameters in expected_parameters.items():
        for name, expected in parameters.items():
            value = config[model_name].get(name)
            if value != expected:
                errors.append(
                    f"{model_name}.{name}={value!r}; esperado {expected!r}."
                )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(
            "La comparación final no se ejecutará hasta restaurar los parámetros "
            f"congelados:\n{details}"
        )
    print("\nConfiguración final confirmada: se usarán los parámetros congelados.")


def create_final_output_directory():
    FINAL_COMPARISON_DIRECTORY.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = FINAL_COMPARISON_DIRECTORY / run_id
    output_dir.mkdir()
    return run_id, output_dir


def append_final_execution_history(run_id, output_dir, records,
                                   total_duration_seconds):
    history = pd.DataFrame(records)
    history.insert(0, "ejecucion", run_id)
    history.insert(1, "directorio", str(output_dir))
    history.insert(2, "total_duration_seconds", total_duration_seconds)
    history_path = FINAL_COMPARISON_DIRECTORY / "registro_ejecuciones.csv"
    history.to_csv(
        history_path,
        mode="a",
        header=not history_path.exists(),
        index=False,
    )


def initial_graphs(df):
    mean_delay_by_hour(df)
    mean_delay_by_month(df)
    mean_delay_by_week(df)
    correlation_matrix(df)


def run_manual_validation(train_df, val_df, test_df, config, model_results,
                          model_predictions, model_histories, trained_models):
    X_train, y_train, X_val, y_val, _, _ = prepare_model_data(
        train_df,
        val_df,
        test_df,
        sample_fraction=config["sample_fraction"],
        drop_columns=["DepDelay", "UniqueCarrier", "Origin", "Dest"],
        normalize=True,
        random_state=config["random_state"],
    )
    model, predictions, history = run_regression_model(
        X_train,
        y_train,
        X_val,
        y_val,
        **config["manual"],
    )
    metrics = evaluate_all_splits(
        y_train,
        predictions["train"],
        y_val,
        predictions["val"],
    )
    trained_models["Manual"] = model
    model_results["Manual"] = metrics
    model_predictions["Manual"] = predictions
    model_histories["Manual"] = history

    if config["generate_individual_graphs"]:
        predicted_vs_actual(
            y_val,
            predictions["val"],
            metrics["val_r2"],
            dataset_name="Manual_Validation",
        )
        residual_plot(y_val, predictions["val"], dataset_name="Manual_Validation")
        error_distribution(y_val, predictions["val"], dataset_name="Manual_Validation")
        plot_training_history(
            history["epoch"],
            history["train_mse"],
            history["val_mse"],
            xlabel="Época",
            ylabel="MSE",
            title="Evolución del MSE — Modelo Manual",
            filename="manual_learning_curve",
        )


def run_xgboost_validation(train_df, val_df, test_df, config, model_name,
                           parameters, add_time_categories, model_results,
                           model_predictions, model_histories, trained_models,
                           fitted_preprocessors):
    X_train, y_train, X_val, y_val, _, _, preprocessor = prepare_model_data(
        train_df,
        val_df,
        test_df,
        sample_fraction=config["sample_fraction"],
        drop_columns=["DepDelay"],
        encode_categorical=True,
        random_state=config["random_state"],
        return_preprocessor=True,
        add_time_categories=add_time_categories,
    )
    model, predictions, history = run_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        **parameters,
    )
    metrics = evaluate_all_splits(
        y_train,
        predictions["train"],
        y_val,
        predictions["val"],
    )
    trained_models[model_name] = model
    model_results[model_name] = metrics
    model_predictions[model_name] = predictions
    model_histories[model_name] = history
    fitted_preprocessors[model_name] = preprocessor

    if config["generate_individual_graphs"]:
        label = model_name.replace(" ", "") + "_Validation"
        predicted_vs_actual(y_val, predictions["val"], metrics["val_r2"], label)
        residual_plot(y_val, predictions["val"], dataset_name=label)
        error_distribution(y_val, predictions["val"], dataset_name=label)
        xgb_cost_evolution(history["train_rmse"], history["val_rmse"])


def final_metrics_record(model_name, metrics, duration_seconds, sample_fraction):
    return {
        "modelo": model_name,
        "sample_fraction": sample_fraction,
        "duration_seconds": duration_seconds,
        **metrics,
    }


def create_prediction_comparison(y_test, predictions_by_model):
    comparison = pd.DataFrame({
        "observacion": range(1, len(y_test) + 1),
        "valor_real": y_test.to_numpy(),
    })
    for model_name, predictions in predictions_by_model.items():
        comparison[f"prediccion_{model_name}"] = predictions
    return comparison.head(10)


def run_final_comparison(train_df, val_df, test_df, config, program_started):
    train_val_df = pd.concat([train_df, val_df], axis=0)
    records = []
    final_predictions = {}
    final_target = None
    improved_model = None
    improved_preprocessor = None

    print("\n" + "=" * 70)
    print("COMPARACIÓN FINAL SOBRE TEST")
    print("=" * 70)

    started = time.perf_counter()
    X_train, y_train, X_test, y_test = prepare_final_model_data(
        train_val_df,
        test_df,
        sample_fraction=config["sample_fraction"],
        drop_columns=["DepDelay", "UniqueCarrier", "Origin", "Dest"],
        normalize=True,
        random_state=config["random_state"],
    )
    manual_model, manual_predictions, _ = run_regression_model(
        X_train,
        y_train,
        X_train,
        y_train,
        X_test,
        y_test,
        **config["manual"],
    )
    final_predictions["manual"] = manual_predictions["test"]
    final_target = y_test
    records.append(
        final_metrics_record(
            "Manual",
            regression_metrics(y_test, manual_predictions["test"]),
            time.perf_counter() - started,
            config["sample_fraction"],
        )
    )

    for model_name, parameters, add_time_categories in (
        ("XGBoost base", config["xgboost_base"], False),
        ("XGBoost mejorado", config["xgboost_improved"], True),
    ):
        started = time.perf_counter()
        X_train, y_train, X_test, y_test, preprocessor = prepare_final_model_data(
            train_val_df,
            test_df,
            sample_fraction=config["sample_fraction"],
            drop_columns=["DepDelay"],
            encode_categorical=True,
            random_state=config["random_state"],
            return_preprocessor=True,
            add_time_categories=add_time_categories,
        )
        if not y_test.equals(final_target):
            raise ValueError("Los modelos no recibieron el mismo conjunto de test.")

        print(
            f"\nEntrenando {model_name}: {parameters['n_estimators']} árboles "
            "(avance cada 100 árboles)."
        )
        model = train_final_xgboost(X_train, y_train, **parameters)
        predictions = model.predict(X_test)
        final_predictions[model_name.lower().replace(" ", "_")] = predictions
        records.append(
            final_metrics_record(
                model_name,
                regression_metrics(y_test, predictions),
                time.perf_counter() - started,
                config["sample_fraction"],
            )
        )
        if model_name == "XGBoost mejorado":
            improved_model = model
            improved_preprocessor = preprocessor

    metrics_by_model = {
        record["modelo"]: {
            f"test_{name}": value
            for name, value in record.items()
            if name not in {"modelo", "sample_fraction", "duration_seconds"}
        }
        for record in records
    }
    comparison = create_comparison_dataframe(metrics_by_model, split="test")
    run_id, output_dir = create_final_output_directory()
    comparison.to_csv(output_dir / "comparacion_test.csv", index=False)
    prediction_comparison = create_prediction_comparison(final_target, final_predictions)
    prediction_comparison.to_csv(
        output_dir / "predicciones_test_primeras_10.csv",
        index=False,
        float_format="%.4f",
    )
    (output_dir / "configuraciones.json").write_text(
        json.dumps(
            {
                "manual": config["manual"],
                "xgboost_base": config["xgboost_base"],
                "xgboost_mejorado": config["xgboost_improved"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    improved_model.save_model(str(output_dir / "xgboost_mejorado.json"))
    with open(output_dir / "preprocessor_xgboost_mejorado.pkl", "wb") as file:
        pickle.dump(improved_preprocessor, file, protocol=pickle.HIGHEST_PROTOCOL)
    total_duration_seconds = time.perf_counter() - program_started
    (output_dir / "metricas.json").write_text(
        json.dumps(
            {
                "total_duration_seconds": total_duration_seconds,
                "models": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    append_final_execution_history(
        run_id,
        output_dir,
        records,
        total_duration_seconds,
    )

    print(comparison.to_string(index=False))
    print("\nPrimeras 10 predicciones sobre test:")
    print(prediction_comparison.to_string(index=False))
    print(f"\nTiempo total del programa: {total_duration_seconds / 60:.2f} minutos")
    print(f"\nResultados finales guardados en: {output_dir}")


def run_experiment():
    program_started = time.perf_counter()
    config = {
        "sample_fraction": 1.0,
        "shuffle_data": True,
        "random_state": 42,
        "run_final_comparison": True,
        "run_manual": False,
        "run_xgboost_base": False,
        "run_xgboost_improved": False,
        "generate_basic_graphs": True,
        "generate_individual_graphs": True,
        "generate_comparison_graphs": True,
        "manual": {
            "learning_rate": 0.01,
            "epochs": 20,
            "batch_size": 2048,
            "random_state": 42,
        },
        "xgboost_base": {
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
        },
        "xgboost_improved": {
            "n_estimators": 400,
            "max_depth": 0,
            "max_leaves": 63,
            "grow_policy": "lossguide",
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
        },
    }

    if config["run_final_comparison"]:
        validate_final_comparison_config(config)

    print("\n" + "=" * 70)
    print("ETL")
    print("=" * 70)
    df, skip_transform = extract_data()
    if not skip_transform:
        df = transform_data(df)
    _, train_df, val_df, test_df = split_data(
        df,
        shuffle=config["shuffle_data"],
        random_state=config["random_state"],
    )

    print("\nDatos separados:")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test reservado: {len(test_df):,}")

    if config["run_final_comparison"]:
        run_final_comparison(train_df, val_df, test_df, config, program_started)
        return

    print(
        f"\nTamaño efectivo del entrenamiento ({config['sample_fraction'] * 100}%): "
        f"{int(len(train_df) * config['sample_fraction']):,}"
    )
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
        run_manual_validation(
            train_df,
            val_df,
            test_df,
            config,
            model_results,
            model_predictions,
            model_histories,
            trained_models,
        )

    if config["run_xgboost_base"]:
        print("\n" + "=" * 70)
        print("XGBOOST BASE")
        print("=" * 70)
        run_xgboost_validation(
            train_df,
            val_df,
            test_df,
            config,
            "XGBoost base",
            config["xgboost_base"],
            False,
            model_results,
            model_predictions,
            model_histories,
            trained_models,
            fitted_preprocessors,
        )

    if config["run_xgboost_improved"]:
        print("\n" + "=" * 70)
        print("XGBOOST MEJORADO")
        print("=" * 70)
        run_xgboost_validation(
            train_df,
            val_df,
            test_df,
            config,
            "XGBoost mejorado",
            config["xgboost_improved"],
            True,
            model_results,
            model_predictions,
            model_histories,
            trained_models,
            fitted_preprocessors,
        )

    if not model_results:
        print("\nNo se ejecutó ningún modelo.")
        return

    comparison = create_comparison_dataframe(model_results, split="val")
    print("\n" + "=" * 90)
    print("COMPARACIÓN DE MODELOS — VALIDATION")
    print("=" * 90)
    print(comparison.to_string(index=False))
    diagnostic = create_diagnostic_dataframe(model_results)
    print_diagnostic_summary(model_results)

    if config["generate_comparison_graphs"]:
        compare_models_bar(model_results, "val_mae", "Comparación de MAE — Validation")
        compare_models_bar(model_results, "val_rmse", "Comparación de RMSE — Validation")
        compare_models_bar(model_results, "val_r2", "Comparación de R² — Validation")
        compare_models_bar(model_results, "val_within_15", "Predicciones dentro de ±15 minutos")
        predictions_for_plot = {
            model_name: predictions["val"]
            for model_name, predictions in model_predictions.items()
        }
        first_model = next(iter(model_predictions))
        binned_prediction_plot(
            predictions_for_plot,
            model_predictions[first_model]["y_val"],
            n_bins=20,
            dataset_name="Validation",
        )

    save_experiment(
        {
            "config": config,
            "metrics": model_results,
            "histories": model_histories,
            "comparison": comparison,
            "diagnostic": diagnostic,
            "models": trained_models,
            "preprocessors": fitted_preprocessors,
        }
    )
    print("\nExperimento terminado.")
