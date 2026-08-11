"""
Script de inferencia en PRODUCCIÓN.

Carga los artefactos guardados por main.py para la mejor arquitectura
(modelo entrenado + preprocesador ya ajustado) y predice sobre datos
nuevos, sin volver a entrenar ni a hacer fit del preprocesador.
"""
import json

import joblib
import pandas as pd
import torch

from src import config
from src.models import build_model
from src.train import to_tensor

ARTIFACTS_DIR = config.PROJECT_ROOT / "models" / "production"


def load_production_model():
    """
    Reconstruye el modelo ganador (misma arquitectura y input_dim con la
    que se entrenó, según metadata.json) y le carga los pesos guardados.
    """
    with open(ARTIFACTS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = build_model(metadata["model_variant"], input_dim=metadata["input_dim"])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu"))
    model.eval()

    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")

    return model, preprocessor, metadata


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predice sobre un DataFrame con las mismas columnas de entrada que el
    dataset original (sin la columna 'income'). Devuelve el DataFrame
    original con dos columnas nuevas: probabilidad predicha y clase
    predicha (0 = <=50K, 1 = >50K).
    """
    model, preprocessor, metadata = load_production_model()

    X_processed = preprocessor.transform(df)
    X_t = to_tensor(X_processed)

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).numpy().ravel()

    result = df.copy()
    result["income_pred_proba"] = probs
    result["income_pred"] = (probs >= 0.5).astype(int)

    print(f"Predicciones generadas con el modelo '{metadata['model_variant']}'.")
    return result


if __name__ == "__main__":
    # Ejemplo: predecir sobre el propio CSV de entrada (sin la columna target)
    sample = pd.read_csv(config.DATA_PATH).drop(columns=[config.TARGET_COLUMN]).head(5)
    print(predict(sample))
