import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from etlProcess import extract_data, transform_data, split_data
from preprocessing import prepare_model_data
from regresionModel import run_regression_model
from xgboostModel import run_xgboost
from evaluation import (
    create_comparison_dataframe,
    create_diagnostic_dataframe,
    evaluate_all_splits,
    print_diagnostic_summary,
)
from experimentStorage import save_experiment
from plotUtil import (
    binned_prediction_plot,
    compare_models_bar,
    correlation_matrix,
    delays_hist,
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
FINAL_REQUIRED_VALUES = {
    "sample_fraction": 1.0,
    "shuffle_data": True,
    "random_state": 42,
}
MANUAL_PARAMETERS = {
    "learning_rate": 0.01,
    "epochs": 20,
    "batch_size": 2048,
    "random_state": 42,
}
XGBOOST_BASE_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "max_bin": 256,
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}
XGBOOST_IMPROVED_PARAMETERS = {
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
}
FINAL_MODEL_PARAMETERS = {
    "manual": MANUAL_PARAMETERS,
    "xgboost_base": XGBOOST_BASE_PARAMETERS,
    "xgboost_improved": XGBOOST_IMPROVED_PARAMETERS,
}


def validate_final_comparison_config(config):
    errors = []
    for name, expected in FINAL_REQUIRED_VALUES.items():
        value = config.get(name)
        if value != expected:
            errors.append(f"{name}={value!r}; esperado {expected!r}.")
    for model_name, parameters in FINAL_MODEL_PARAMETERS.items():
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
    delays_hist(df)
    mean_delay_by_hour(df)
    mean_delay_by_month(df)
    mean_delay_by_week(df)
    correlation_matrix(df)


def run_manual_model(train_df, val_df, test_df, config, evaluate_test=False):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_model_data(
        train_df,
        val_df,
        test_df,
        sample_fraction=config["sample_fraction"],
        drop_columns=["DepDelay", "UniqueCarrier", "Origin", "Dest"],
        normalize=True,
        random_state=config["random_state"],
    )
    model_X_test = X_test if evaluate_test else None
    model_y_test = y_test if evaluate_test else None
    model, predictions, history = run_regression_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model_X_test,
        model_y_test,
        **config["manual"],
    )
    metrics = evaluate_all_splits(
        y_train,
        predictions["train"],
        y_val,
        predictions["val"],
        model_y_test,
        predictions.get("test"),
    )
    return model, predictions, history, metrics


def run_xgboost_model(train_df, val_df, test_df, config, parameters,
                      add_time_categories, evaluate_test=False):
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = prepare_model_data(
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
    model_X_test = X_test if evaluate_test else None
    model_y_test = y_test if evaluate_test else None
    model, predictions, history = run_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        model_X_test,
        model_y_test,
        **parameters,
    )
    metrics = evaluate_all_splits(
        y_train,
        predictions["train"],
        y_val,
        predictions["val"],
        model_y_test,
        predictions.get("test"),
    )
    return model, predictions, history, metrics, preprocessor


def generate_individual_graphs(model_name, predictions, history, metrics, split):
    graph_suffix = "Test_Final" if split == "test" else "Validation"
    label = model_name.replace(" ", "") + f"_{graph_suffix}"
    y_evaluation = predictions[f"y_{split}"]
    predicted_vs_actual(
        y_evaluation,
        predictions[split],
        metrics[f"{split}_r2"],
        dataset_name=label,
    )
    residual_plot(y_evaluation, predictions[split], dataset_name=label)
    error_distribution(y_evaluation, predictions[split], dataset_name=label)

    if model_name == "Manual":
        filename = (
            "manual_learning_curve_test_final"
            if split == "test" else "manual_learning_curve"
        )
        plot_training_history(
            history["epoch"],
            history["train_mse"],
            history["val_mse"],
            xlabel="Época",
            ylabel="MSE",
            title="Evolución del MSE — Modelo Manual",
            filename=filename,
        )
        return

    filename_suffix = "test_final" if split == "test" else "validation"
    xgb_cost_evolution(
        history["train_rmse"],
        history["val_rmse"],
        title=f"Evolución del RMSE — {model_name}",
        filename=model_name.lower().replace(" ", "_")
        + f"_learning_curve_{filename_suffix}",
    )


def final_metrics_record(model_name, metrics, duration_seconds, sample_fraction):
    test_metrics = {
        name.removeprefix("test_"): value
        for name, value in metrics.items()
        if name.startswith("test_")
    }
    return {
        "modelo": model_name,
        "sample_fraction": sample_fraction,
        "duration_seconds": duration_seconds,
        **test_metrics,
    }


def create_prediction_comparison(y_test, predictions_by_model):
    comparison = pd.DataFrame({
        "observacion": range(1, len(y_test) + 1),
        "valor_real": y_test.to_numpy(),
    })
    for model_name, predictions in predictions_by_model.items():
        prediction_key = model_name.lower().replace(" ", "_")
        comparison[f"prediccion_{prediction_key}"] = predictions
    return comparison.head(10)


def generate_comparison_graphs(model_results, model_predictions, split,
                               dataset_name):
    title_suffix = "Test final" if split == "test" else "Validation"
    for metric, title in (
        ("mae", "Comparación de MAE"),
        ("rmse", "Comparación de RMSE"),
        ("r2", "Comparación de R²"),
        ("within_15", "Predicciones dentro de ±15 minutos"),
    ):
        compare_models_bar(
            model_results,
            f"{split}_{metric}",
            f"{title} — {title_suffix}",
        )
    predictions_for_plot = {
        model_name: predictions[split]
        for model_name, predictions in model_predictions.items()
    }
    first_model = next(iter(model_predictions))
    binned_prediction_plot(
        predictions_for_plot,
        model_predictions[first_model][f"y_{split}"],
        n_bins=20,
        dataset_name=dataset_name,
    )


def run_final_comparison(train_df, val_df, test_df, config, program_started):
    records = []
    model_results = {}
    model_predictions = {}
    improved_model = None
    improved_preprocessor = None

    print("\n" + "=" * 70)
    print("COMPARACIÓN FINAL SOBRE TEST")
    print("=" * 70)

    started = time.perf_counter()
    _, predictions, history, metrics = run_manual_model(
        train_df,
        val_df,
        test_df,
        config,
        evaluate_test=True,
    )
    model_results["Manual"] = metrics
    model_predictions["Manual"] = predictions
    records.append(
        final_metrics_record(
            "Manual",
            metrics,
            time.perf_counter() - started,
            config["sample_fraction"],
        )
    )
    if config["generate_individual_graphs"]:
        generate_individual_graphs("Manual", predictions, history, metrics, "test")

    for model_name, parameters, add_time_categories in (
        ("XGBoost base", config["xgboost_base"], False),
        ("XGBoost mejorado", config["xgboost_improved"], True),
    ):
        print(
            f"\nEntrenando {model_name}: {parameters['n_estimators']} árboles "
            "(avance cada 100 árboles)."
        )
        started = time.perf_counter()
        model, predictions, history, metrics, preprocessor = run_xgboost_model(
            train_df,
            val_df,
            test_df,
            config,
            parameters,
            add_time_categories,
            evaluate_test=True,
        )
        model_results[model_name] = metrics
        model_predictions[model_name] = predictions
        records.append(
            final_metrics_record(
                model_name,
                metrics,
                time.perf_counter() - started,
                config["sample_fraction"],
            )
        )
        if config["generate_individual_graphs"]:
            generate_individual_graphs(
                model_name,
                predictions,
                history,
                metrics,
                "test",
            )
        if model_name == "XGBoost mejorado":
            improved_model = model
            improved_preprocessor = preprocessor

    comparison = create_comparison_dataframe(model_results, split="test")
    if config["generate_comparison_graphs"]:
        print("\nGenerando gráficas comparativas de test...")
        generate_comparison_graphs(
            model_results,
            model_predictions,
            "test",
            "Test_Final_XGBoost",
        )
    run_id, output_dir = create_final_output_directory()
    comparison.to_csv(output_dir / "comparacion_test.csv", index=False)
    final_predictions = {
        model_name: predictions["test"]
        for model_name, predictions in model_predictions.items()
    }
    prediction_comparison = create_prediction_comparison(
        test_df["DepDelay"],
        final_predictions,
    )
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
        "manual": MANUAL_PARAMETERS.copy(),
        "xgboost_base": XGBOOST_BASE_PARAMETERS.copy(),
        "xgboost_improved": XGBOOST_IMPROVED_PARAMETERS.copy(),
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

    if config["generate_basic_graphs"]:
        print("\nGenerando gráficos iniciales...")
        initial_graphs(df)
        print("Gráficos iniciales terminados.")

    if config["run_final_comparison"]:
        run_final_comparison(train_df, val_df, test_df, config, program_started)
        return

    print(
        f"\nTamaño efectivo del entrenamiento ({config['sample_fraction'] * 100}%): "
        f"{int(len(train_df) * config['sample_fraction']):,}"
    )
    model_results = {}
    model_predictions = {}
    model_histories = {}
    trained_models = {}
    fitted_preprocessors = {}

    if config["run_manual"]:
        print("\n" + "=" * 70)
        print("REGRESIÓN MANUAL")
        print("=" * 70)
        model, predictions, history, metrics = run_manual_model(
            train_df,
            val_df,
            test_df,
            config,
        )
        trained_models["Manual"] = model
        model_results["Manual"] = metrics
        model_predictions["Manual"] = predictions
        model_histories["Manual"] = history
        if config["generate_individual_graphs"]:
            generate_individual_graphs(
                "Manual",
                predictions,
                history,
                metrics,
                "val",
            )

    for model_name, config_key, run_key, add_time_categories in (
        ("XGBoost base", "xgboost_base", "run_xgboost_base", False),
        ("XGBoost mejorado", "xgboost_improved", "run_xgboost_improved", True),
    ):
        if not config[run_key]:
            continue
        print("\n" + "=" * 70)
        print(model_name.upper())
        print("=" * 70)
        model, predictions, history, metrics, preprocessor = run_xgboost_model(
            train_df,
            val_df,
            test_df,
            config,
            config[config_key],
            add_time_categories,
        )
        trained_models[model_name] = model
        model_results[model_name] = metrics
        model_predictions[model_name] = predictions
        model_histories[model_name] = history
        fitted_preprocessors[model_name] = preprocessor
        if config["generate_individual_graphs"]:
            generate_individual_graphs(
                model_name,
                predictions,
                history,
                metrics,
                "val",
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
        generate_comparison_graphs(
            model_results,
            model_predictions,
            "val",
            "Validation",
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
