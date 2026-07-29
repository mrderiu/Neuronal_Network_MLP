import pandas as pd
from xgboost import XGBClassifier

from src import config


def build_xgb_model(y_train: pd.Series) -> XGBClassifier:
    """
    Construye (sin entrenar) un XGBClassifier configurado para el
    problema de clasificación binaria con clases desbalanceadas.

    scale_pos_weight es el equivalente en XGBoost al pos_weight que
    usamos en BCEWithLogitsLoss: compensa que la clase 1 (>50K) es
    minoritaria (~24% de los datos).
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    return model


def train_xgb_model(
    model: XGBClassifier,
    X_train,
    y_train: pd.Series,
    X_val,
    y_val: pd.Series,
) -> XGBClassifier:
    """
    Entrena el modelo con early stopping sobre el conjunto de validación:
    si el logloss de validación no mejora en 20 rondas consecutivas,
    para el entrenamiento y se queda con el mejor punto.
    """
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    return model
