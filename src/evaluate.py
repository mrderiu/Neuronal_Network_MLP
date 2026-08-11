import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.train import to_tensor


def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Runs a forward pass and returns binary predictions (0/1) as a numpy
    array. Shared by evaluate_model() and compute_metrics() so both use
    the exact same inference logic.
    """
    device = next(model.parameters()).device
    X_t = to_tensor(X).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    return (probs >= 0.5).astype(int)


def compute_metrics(model: nn.Module, X: np.ndarray, y: pd.Series) -> dict:
    """
    Lightweight, numeric-only version of evaluate_model(): returns a flat
    dict of scalar metrics (no confusion matrix / classification report),
    which makes it easy to log to a CSV or compare across models.

    Used to compare architectures on the VALIDATION set, and to log both
    validation and test metrics for every experiment run.
    """
    y_pred = _predict(model, X)
    y_true = y.values

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """
    Evalúa el modelo sobre el conjunto de test y devuelve métricas básicas.
    Pensada para el reporte FINAL de un único modelo ya seleccionado
    (no para comparar arquitecturas entre sí: para eso usar compute_metrics
    sobre el conjunto de validación).
    """
    y_pred = _predict(model, X_test)
    y_true = y_test.values

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])
    print("Reporte de clasificación:")
    print(metrics["classification_report"])

    return metrics

