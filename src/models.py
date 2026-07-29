import torch.nn as nn


def mlp_model_simple(input_dim: int) -> nn.Module:
    """
    Construye un modelo de red neuronal multicapa (MLP) para clasificación binaria.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),

        nn.Linear(64, 32),
        nn.ReLU(),

        nn.Linear(32, 8),
        nn.ReLU(),

        nn.Linear(8, 1),
    )

    return model

def mlp_model_BatchNorm_Dropout(input_dim: int) -> nn.Module:
    """
    Construye un modelo de red neuronal multicapa (MLP) para clasificación binaria. Se agregan capas de Batch Normalization y Dropout para mejorar la generalización.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    return model

def mlp_model_ResNet18(input_dim: int) -> nn.Module:
    """
    Construye un modelo de red neuronal multicapa (MLP) para clasificación binaria.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(32, 1)  # sin Sigmoid
    )

    return model