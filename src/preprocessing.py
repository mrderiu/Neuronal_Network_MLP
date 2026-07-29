from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def values_preprocessing(df):
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)
    return df

def feature_engineering_preprocessing(df):
    
    return df


def build_preprocessor(df, numerical_features: List[str], categorical_features: List[str],) -> ColumnTransformer:
    """
    Construye (sin ajustar) el ColumnTransformer que preprocesa
    variables numéricas y categóricas.

    Importante: esta función NO ajusta (fit) nada, solo devuelve
    el objeto. El ajuste se hace una única vez, en el conjunto de
    entrenamiento, llamando a .fit_transform(). Para validación y
    test se usa el mismo objeto ya ajustado, llamando solo a
    .transform(), para no filtrar información del train.
    """

    numerical_pipeline = Pipeline(
        steps=[
            
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor
