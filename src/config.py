from pathlib import Path

# --- Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "input" / "AdultCensusIncome.csv"

# --- Columns ---
TARGET_COLUMN = "income"

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

# --- Data Split ---
TEST_SIZE = 0.30
VALIDATION_TEST_SPLIT = 0.50
RANDOM_STATE = 42

# --- Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# --- Early Stopping ---
# Nº de epochs consecutivos sin mejora en val_loss antes de detener
# el entrenamiento. Al finalizar, se restauran los pesos del mejor epoch.
EARLY_STOPPING_PATIENCE = 10