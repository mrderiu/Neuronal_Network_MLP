import json

import joblib
import torch

from src import config
from src.data_loader import load_data, split_data
from src.evaluate import compute_metrics, evaluate_model
from src.experiment_log import log_run
from src.models import MODEL_REGISTRY, build_model
from src.preprocessing import build_preprocessor, values_preprocessing
from src.train import train_model

ARTIFACTS_DIR = config.PROJECT_ROOT / "models" / "production"
LOG_PATH = config.PROJECT_ROOT / "logs" / "experiments.csv"


def run_experiment(
    model_name: str,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
) -> dict:
    """
    Entrena una única arquitectura y devuelve el modelo entrenado junto
    con sus métricas de validación y test. No decide todavía cuál es la
    mejor: eso se hace en main(), comparando entre arquitecturas.
    """
    model = build_model(model_name, input_dim=X_train.shape[1])
    print(f"\n--- Entrenando modelo '{model_name}' ---")

    model = train_model(model, X_train, y_train, X_val, y_val)

    val_metrics = compute_metrics(model, X_val, y_val)
    test_metrics = compute_metrics(model, X_test, y_test)

    print(
        f"'{model_name}' -> val_f1: {val_metrics['f1']:.4f} | "
        f"test_f1: {test_metrics['f1']:.4f}"
    )

    return {
        "model_name": model_name,
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def save_production_artifacts(model_name: str, model, preprocessor, input_dim: int) -> None:
    """
    Guarda en disco todo lo necesario para predecir en producción con el
    modelo ganador (ver predict.py):

    - model.pt             -> pesos del modelo (state_dict)
    - preprocessor.joblib  -> ColumnTransformer ya ajustado (fit sobre train)
    - metadata.json        -> qué arquitectura es y con qué input_dim se
                               construyó, para poder reconstruirla antes
                               de cargar los pesos
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

    metadata = {"model_variant": model_name, "input_dim": input_dim}
    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artefactos de producción guardados en: {ARTIFACTS_DIR}")


def main() -> dict:
    df = load_data()
    df = values_preprocessing(df)  # Preprocesamiento de valores faltantes y outliers
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(df)

    preprocessor = build_preprocessor(
        df,
        config.NUMERICAL_FEATURES,
        config.CATEGORICAL_FEATURES,
    )
    # El preprocesador se ajusta UNA sola vez, con el train
    X_train_processed = preprocessor.fit_transform(X_train)

    # Para validación y test solo se transforma, no se reajusta
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)

    # --- Entrena y compara todas las arquitecturas registradas en models.py ---
    results = []
    for model_name in MODEL_REGISTRY:
        result = run_experiment(
            model_name,
            X_train_processed, y_train,
            X_validation_processed, y_validation,
            X_test_processed, y_test,
        )
        results.append(result)

        # Se registra cada corrida en el log CSV. La selección del mejor
        # modelo se hace luego con la métrica de VALIDACIÓN, no de test:
        # el test se guarda solo para el reporte final del ganador, así
        # no se filtra información del test en la decisión.
        log_run(
            LOG_PATH,
            {
                "model_variant": result["model_name"],
                "val_accuracy": result["val_metrics"]["accuracy"],
                "val_precision": result["val_metrics"]["precision"],
                "val_recall": result["val_metrics"]["recall"],
                "val_f1": result["val_metrics"]["f1"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_precision": result["test_metrics"]["precision"],
                "test_recall": result["test_metrics"]["recall"],
                "test_f1": result["test_metrics"]["f1"],
            },
        )

    # --- Selección del ganador ---
    # F1 de validación en vez de accuracy: con clases desbalanceadas
    # (income > 50K es minoritaria), accuracy puede ser engañosa.
    best_result = max(results, key=lambda r: r["val_metrics"]["f1"])
    best_model_name = best_result["model_name"]
    best_model = best_result["model"]

    print(
        f"\nMejor arquitectura: '{best_model_name}' "
        f"(val_f1={best_result['val_metrics']['f1']:.4f})"
    )

    # Reporte final completo (accuracy, matriz de confusión, classification
    # report) SOLO para el modelo ganador.
    print(f"\n--- Evaluación final en test del modelo '{best_model_name}' ---")
    final_metrics = evaluate_model(best_model, X_test_processed, y_test)

    save_production_artifacts(
        best_model_name, best_model, preprocessor, input_dim=X_train_processed.shape[1]
    )

    return {
        "preprocessor": preprocessor,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "metrics": final_metrics,
        "all_results": results,
    }


if __name__ == "__main__":
    results = main()
