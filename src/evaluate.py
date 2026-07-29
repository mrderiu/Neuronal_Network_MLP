import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.train import to_tensor


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """
    Evalúa el modelo sobre el conjunto de test y devuelve métricas básicas.
    """
    device = next(model.parameters()).device
    X_test_t, y_test_t = to_tensor(X_test, y_test)
    X_test_t = X_test_t.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    y_pred = (probs >= 0.5).astype(int)
    y_true = y_test.values

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])
    print("Reporte de clasificación:")
    print(metrics["classification_report"])

    return metrics
