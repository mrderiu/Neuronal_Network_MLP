from src import config
from src.data_loader import load_data, split_data
from src.preprocessing import build_preprocessor, values_preprocessing
from src.models import mlp_model_simple
from src.train import train_model
from src.evaluate import evaluate_model


def main() -> dict:
    df = load_data()
    df = values_preprocessing(df)  # Preprocesamiento de valores faltantes y outliers
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(df)

    preprocessor = build_preprocessor(
        df,
        config.NUMERICAL_FEATURES,
        config.CATEGORICAL_FEATURES
    )
    # El preprocesador se ajusta UNA sola vez, con el train
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Para validación y test solo se transforma, no se reajusta
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)

    model = mlp_model_simple(input_dim=X_train_processed.shape[1])
    print("Modelo MLP construido con éxito.")

    model = train_model(
        model,
        X_train_processed,
        y_train,
        X_validation_processed,
        y_validation,
    )

    metrics = evaluate_model(model, X_test_processed, y_test)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "metrics": metrics,
    }


if __name__ == "__main__":
    results = main()
