"""
MLP para clasificación binaria real vs. IA generada.

Incluye arquitectura y loop de entrenamiento.

Uso:
    from src.mlp import train_mlp
    historia = train_mlp()
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.config import (
    DATA_PROC, OUTPUTS,
    BATCH_SIZE, LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    PCA_N_COMPONENTS,
)

torch.manual_seed(SEED)

_LABEL_TO_INT = {"real": 0, "fake": 1}


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """356 → 512 → 256 → 1 con ReLU y Dropout."""

    def __init__(self, input_dim: int = PCA_N_COMPONENTS, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Carga de datos con PCA
# ---------------------------------------------------------------------------

def _cargar_split_pca(split: str, pca) -> tuple[torch.Tensor, torch.Tensor]:
    """Carga imágenes de processed/{split}, aplana y transforma con PCA."""
    paths, labels = [], []
    for label in ("real", "fake"):
        for p in sorted((DATA_PROC / split / label).iterdir()):
            paths.append(p)
            labels.append(_LABEL_TO_INT[label])

    filas = []
    for p in tqdm(paths, desc=f"Cargando {split}", leave=False):
        with Image.open(p) as img:
            filas.append(np.array(img, dtype=np.uint8).flatten())

    X = pca.transform(np.stack(filas)).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_mlp() -> dict:
    OUTPUTS.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    pca = joblib.load(OUTPUTS / "pca_mlp.joblib")
    X_train, y_train = _cargar_split_pca("train", pca)
    X_val,   y_val   = _cargar_split_pca("val",   pca)
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    model     = MLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    historia = {"train_loss": [], "val_loss": [], "val_acc": []}
    mejor_val_loss    = float("inf")
    epochs_sin_mejora = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds     = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
                correct  += ((preds.sigmoid() >= 0.5) == y_batch).sum().item()
        val_loss /= len(X_val)
        val_acc   = correct / len(X_val)

        historia["train_loss"].append(train_loss)
        historia["val_loss"].append(val_loss)
        historia["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), OUTPUTS / "mlp_best.pt")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"Early stopping en epoch {epoch}.")
                break

    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    _plot_curvas(historia)
    return historia


def _plot_curvas(historia: dict) -> None:
    epochs = range(1, len(historia["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, historia["train_loss"], label="train")
    ax1.plot(epochs, historia["val_loss"],   label="val")
    ax1.set_title("Loss por epoch")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, historia["val_acc"], color="green", label="val accuracy")
    ax2.set_title("Val Accuracy por epoch")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUTS / "mlp_curvas.png", dpi=150)
    plt.show()
