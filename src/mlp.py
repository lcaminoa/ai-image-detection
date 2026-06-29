"""
MLP para clasificación binaria real vs. generado por IA.

El MLP no trabaja sobre píxeles crudos sino sobre componentes de PCA.
Las imágenes se cargan a 128x128x3 (49 152 features) y se comprimen con PCA
a aprox 350-500 componentes, haciendo el entrenamiento viable en RAM.
"""

import sys
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
    CLASES, LABEL_TO_INT, MLP_IMG_SIZE,
)

torch.manual_seed(SEED)



# == Arquitectura del MLP ==


class MLP(nn.Module):
    """
    Arquitectura: input_dim -> 512 -> 256 -> 1

    La capa de salida no tiene Sigmoid porque se usa BCEWithLogitsLoss,
    que combina sigmoid + BCE en una sola operación (pero numéricamente estable).
    Si usáramos Sigmoid + BCELoss por separado, el gradiente se anula cuando
    la salida está muy cerca de 0 o 1 (vanishing gradient).

    Usa Dropout en capas ocultas para regularizar: en cada paso apaga
    neuronas random, forzando a la red a no depender de features específicas.
    """

    def __init__(self, input_dim: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout), # Apaga 40% de neuronas random durante train
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1), # Salida -> logit (valor real, sin sigmoide)
        )

    def forward(self, x):
        return self.net(x)



# == Carga de datos con PCA ==


def _cargar_split_pca(split: str, pca) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Carga todas las imágenes de processed/{split}/, las aplana a vectores 1D
    y las transforma con PCA para obtener la representación comprimida.

    Devuelve tensores listos para TensorDataset: 'X' de shape (N, pca.n_components_)
    e 'y' de shape (N, 1) con etiquetas float para BCEWithLogitsLoss.
    """
    paths, labels = [], []
    for label in CLASES:
        for p in sorted((DATA_PROC / split / label).iterdir()):
            paths.append(p)
            labels.append(LABEL_TO_INT[label])

    filas = []
    for p in tqdm(paths, desc=f"Cargando {split}", leave=False, file=sys.stdout):
        with Image.open(p) as img:
            # Redimensiona a MLP_IMG_SIZE (128x128) antes de aplanar.
            # Las imágenes en processed/ están a 224x224 para la CNN, pero el MLP
            # necesita menos resolución para no reventar la RAM con el stack numpy.
            img = img.resize((MLP_IMG_SIZE, MLP_IMG_SIZE))
            filas.append(np.array(img, dtype=np.uint8).flatten())

    X = pca.transform(np.stack(filas)).astype(np.float32) # (N, PCA_N_COMPONENTS)
    y = np.array(labels, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1) # unsqueeze: hace (N,) -> (N,1)



# == Entrenamiento ==


def train_mlp() -> dict:
    """
    Entrena el MLP y devuelve el historial de métricas por epoch.

    Guarda el mejor modelo (el de menor val_loss) en outputs/mlp_best.pt.
    Early stopping frena el entrenamiento si val_loss no mejora en PATIENCE epochs.
    """
    OUTPUTS.mkdir(exist_ok=True)

    # Prioriza GPU: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # El PCA fue ajustado con whiten=True (componentes con varianza unitaria), lo que estabiliza el entrenamiento de la red
    pca = joblib.load(OUTPUTS / "pca_mlp.joblib")
    X_train, y_train = _cargar_split_pca("train", pca)
    X_val, y_val = _cargar_split_pca("val", pca)
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = MLP(input_dim=pca.n_components_).to(device)
    criterion = nn.BCEWithLogitsLoss() # Más estable que Sigmoid + BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    historia = {"train_loss": [], "val_loss": [], "val_acc": []}
    mejor_val_loss = float("inf")
    epochs_sin_mejora = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # = Training =
        model.train() # El modo train activa Dropout
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad() # Limpia gradientes del paso anterior
            loss = criterion(model(X_batch), y_batch)
            loss.backward() # Calcula gradientes
            optimizer.step() # Actualiza pesos
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train) # Promedio ponderado por tamaño de batch

        # = Validation =
        model.eval() # Desactiva Dropout para evaluación determinista
        val_loss, correct = 0.0, 0
        with torch.no_grad(): # Desactivar gradientes
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
                # Sigmoide convierte logit a probabilidad para comparar con umbral 0.5
                correct  += ((preds.sigmoid() >= 0.5) == y_batch).sum().item()
        val_loss /= len(X_val)
        val_acc = correct / len(X_val)

        historia["train_loss"].append(train_loss)
        historia["val_loss"].append(val_loss)
        historia["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)

        # = Early stopping y guardado =
        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), OUTPUTS / "mlp_best.pt") # Guarda solo pesos, no arquitectura
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"Early stopping en epoch {epoch}.")
                break

    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    _plot_curvas(historia)
    return historia


def _plot_curvas(historia: dict) -> None:
    """
    Grafica loss y accuracy por epoch.
    """
    epochs = range(1, len(historia["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, historia["train_loss"], label="train")
    ax1.plot(epochs, historia["val_loss"], label="val")
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
    plt.show()
