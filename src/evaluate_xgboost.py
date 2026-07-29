import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


def evaluate_xgb_model(model: XGBClassifier, X_test, y_test: pd.Series) -> dict:
    """
    Evalúa el modelo XGBoost sobre el conjunto de test y devuelve métricas básicas.
    """
    y_pred = model.predict(X_test)
    y_true = y_test.values if hasattr(y_test, "values") else y_test

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
