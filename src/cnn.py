"""
CNN entrenada desde cero para clasificación binaria real vs. IA generada.

A diferencia del MLP, la CNN trabaja directamente sobre los píxeles de la imagen
(128x128x3) sin necesidad de PCA. Las capas convolucionales detectan automáticamente
patrones espaciales (bordes, texturas, estructuras) que el MLP no puede capturar
porque trata a cada píxel como una feature independiente.

Uso:
    from src.cnn import train_cnn
    historia = train_cnn()
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.config import (
    OUTPUTS, IMG_SIZE, IMG_CHANNELS,
    BATCH_SIZE, LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
)
from src.data import ImageDataset

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """
    4 bloques convolucionales con filtros crecientes: 32 → 64 → 128 → 256.

    Cada bloque: Conv2d → BatchNorm → ReLU → MaxPool(2,2)
    Cada MaxPool reduce el mapa espacial a la mitad:
        128x128 → 64x64 → 32x32 → 16x16 → 8x8

    Los filtros se duplican en cada bloque porque los patrones más profundos
    (texturas complejas, estructuras de alto nivel) requieren más combinaciones
    que los patrones simples (bordes, gradientes) de las primeras capas.

    BatchNorm estabiliza el entrenamiento normalizando las activaciones de cada
    capa, lo que permite usar learning rates más altos y reduce la sensibilidad
    a la inicialización de pesos.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: detecta bordes y gradientes simples (32 filtros de 3x3x3)
            nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1),  # padding=1 mantiene el tamaño espacial
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128x128 → 64x64
            # Bloque 2: detecta texturas y patrones locales
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64x64 → 32x32
            # Bloque 3: detecta patrones más complejos y combinaciones de texturas
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32x32 → 16x16
            # Bloque 4: detecta estructuras de alto nivel (composición, coherencia global)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16 → 8x8
        )
        # Después de 4 MaxPool(2), el mapa espacial es IMG_SIZE / 2^4
        spatial = IMG_SIZE // (2 ** 4)  # 128 / 16 = 8
        self.classifier = nn.Sequential(
            nn.Flatten(),                              # (B, 256, 8, 8) → (B, 16384)
            nn.Linear(256 * spatial * spatial, 512),
            nn.ReLU(),
            nn.Dropout(dropout),  # regularización antes de la capa de salida
            nn.Linear(512, 1),    # logit de salida (sin sigmoid, se usa BCEWithLogitsLoss)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_cnn() -> dict:
    """
    Entrena la CNN y devuelve el historial de métricas por epoch.

    Guarda el mejor modelo (menor val_loss) en outputs/cnn_best.pt.
    Misma lógica de early stopping que el MLP para comparación justa.
    """
    OUTPUTS.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ImageDataset carga imágenes desde processed/ y aplica las transforms del split
    # (augmentation solo en train, normalización ImageNet en todos)
    train_loader = DataLoader(ImageDataset("train"), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ImageDataset("val"),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()  # más estable que Sigmoid + BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    historia = {"train_loss": [], "val_loss": [], "val_acc": []}
    mejor_val_loss    = float("inf")
    epochs_sin_mejora = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # — Fase de entrenamiento —
        model.train()  # activa Dropout y BatchNorm en modo entrenamiento
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # (B,) → (B,1) float para BCEWithLogitsLoss
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_loader.dataset)

        # — Fase de validación —
        model.eval()  # desactiva Dropout y pone BatchNorm en modo inferencia
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                preds  = model(imgs)
                val_loss += criterion(preds, labels).item() * len(imgs)
                # sigmoid convierte logit a probabilidad, umbral 0.5 para clasificación binaria
                correct  += ((preds.sigmoid() >= 0.5) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc   = correct / len(val_loader.dataset)

        historia["train_loss"].append(train_loss)
        historia["val_loss"].append(val_loss)
        historia["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # — Early stopping y checkpoint —
        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), OUTPUTS / "cnn_best.pt")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"Early stopping en epoch {epoch}.")
                break

    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    _plot_curvas(historia)
    return historia


def _plot_curvas(historia: dict) -> None:
    """Grafica loss y accuracy por epoch y guarda la figura en outputs/."""
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
    plt.savefig(OUTPUTS / "cnn_curvas.png", dpi=150)
    plt.show()
