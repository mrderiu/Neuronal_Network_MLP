import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


def to_tensor(X: np.ndarray, y: pd.Series = None):
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return X_t
    y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)
    return X_t, y_t

def train_model(model: nn.Module, X_train: np.ndarray, y_train: pd.Series, X_val: np.ndarray, y_val: pd.Series, epochs: int = config.EPOCHS, batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,) -> nn.Module:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_t, y_train_t = to_tensor(X_train, y_train)
    X_val_t, y_val_t = to_tensor(X_val, y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # El modelo ahora devuelve logits crudos (sin Sigmoid), por eso usamos
    # BCEWithLogitsLoss en vez de BCELoss: aplica el sigmoid internamente
    # de forma numéricamente más estable, y permite pesar la clase minoritaria.
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            X_val_device = X_val_t.to(device)
            y_val_device = y_val_t.to(device)
            val_logits = model(X_val_device)
            val_loss = criterion(val_logits, y_val_device).item()
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs >= 0.5).float()
            val_accuracy = (val_preds == y_val_device).float().mean().item()

        print(
            f"Epoch {epoch:03d}/{epochs} - "
            f"train_loss: {train_loss:.4f} - "
            f"val_loss: {val_loss:.4f} - "
            f"val_accuracy: {val_accuracy:.4f}"
        )

    return model
