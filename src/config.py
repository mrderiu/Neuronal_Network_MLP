from pathlib import Path

# --- Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "input" / "AdultCensusIncome.csv"

# --- Columnas ---
TARGET_COLUMN = "income"

# OJO: confirma si tu CSV usa "_" o "-" en estos nombres de columna.
# Si no coinciden exactamente, ColumnTransformer lanzará un KeyError.
NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# --- Split de datos ---
TEST_SIZE = 0.30          # proporción para (validation + test)
VALIDATION_TEST_SPLIT = 0.50  # de ese 30%, la mitad para validation y la mitad para test
RANDOM_STATE = 42

# --- Hiperparámetros del modelo ---
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
