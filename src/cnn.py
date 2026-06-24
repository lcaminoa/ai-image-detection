"""
CNN entrenada desde cero para clasificación binaria real vs. generada por IA.

A diferencia del MLP, la CNN trabaja directamente sobre los píxeles de la imagen
(224x224x3) sin necesidad de PCA. Las capas convolucionales detectan automáticamente
patrones espaciales (bordes, texturas, estructuras) que el MLP no puede capturar
porque trata a cada píxel como una feature independiente.

Uso:
    from src.cnn import train_cnn
    historial = train_cnn()
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



# === Arquitectura del modelo ===

class CNN(nn.Module):
    """
    4 bloques convolucionales con filtros crecientes: 32 -> 64 -> 128 -> 256.

    Cada bloque: Conv2d -> BatchNorm -> ReLU -> MaxPool(2,2)
    Cada MaxPool reduce el mapa espacial a la mitad:
        224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14

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
            # Capa 1: 32 filtros de 3x3x3
            nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1), # padding=1 mantiene tamaño de la img
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 224x224 -> 112x112

            # Capa 2: 64 filtros de 3x3x3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112x112 -> 56x56

            # Capa 3: 128 filtros de 3x3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56x56 -> 28x28

            # Capa 4: 256 filtros de 3x3x3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # 28x28 -> 14x14
        )
        # Después de 4 MaxPool(2), el mapa espacial es IMG_SIZE / 2^4
        final_size = IMG_SIZE // (2 ** 4)
        self.classifier = nn.Sequential(
            nn.Flatten(), # (BatchSize, 256, 14, 14) -> (BatchSize, 50176)
            nn.Linear(256 * final_size * final_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout), # Regularización antes de la capa de salida
            nn.Linear(512, 1), # Logit de salida (sin sigmoid, se usa BCEWithLogitsLoss)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_cnn() -> dict:
    """
    Entrena la CNN y devuelve el historiall de métricas por epoch.

    Guarda el mejor modelo (menor val_loss) en outputs/cnn_best.pt.
    """
    OUTPUTS.mkdir(exist_ok=True)
    # Prioriza GPU: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ImageDataset carga imágenes desde processed/ y aplica las transforms del split
    train_loader = DataLoader(ImageDataset("train"), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(ImageDataset("val"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CNN().to(device)
    criterion = nn.BCEWithLogitsLoss() # Más estable que Sigmoid + BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    historial = {"train_loss": [], "val_loss": [], "val_acc": []}
    mejor_val_loss = float("inf")
    epochs_sin_mejora = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # = Training =
        model.train() # Activa Dropout y BatchNorm en modo train
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device) # (B,) -> (B,1) float para BCEWithLogitsLoss
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_loader.dataset)

        # = Validation =
        model.eval() # Desactiva Dropout y pone BatchNorm en modo eval
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                preds  = model(imgs)
                val_loss += criterion(preds, labels).item() * len(imgs)
                # Sigmoid convierte logit a probabilidad, umbral 0.5 para clasificación binaria
                correct  += ((preds.sigmoid() >= 0.5) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        historial["train_loss"].append(train_loss)
        historial["val_loss"].append(val_loss)
        historial["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # = Early stopping y checkpoint =
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), OUTPUTS / "cnn_best.pt")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"Early stopping en epoch {epoch}.")
                break

    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    _plot_curvas(historial)
    return historial


def _plot_curvas(historial: dict) -> None:
    """
    Grafica loss y accuracy por epoch y guarda la figura en outputs/.
    """
    epochs = range(1, len(historial["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, historial["train_loss"], label="train")
    ax1.plot(epochs, historial["val_loss"], label="val")
    ax1.set_title("Loss por epoch")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, historial["val_acc"], color="green", label="val accuracy")
    ax2.set_title("Val Accuracy por epoch")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUTS / "cnn_curvas.png", dpi=150)
    plt.show()
