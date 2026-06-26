"""
Transfer learning con ResNet50 preentrenado en ImageNet.

Estrategia: Feature extraction — backbone completamente congelado, solo se
entrena el clasificador final. Los pesos de ResNet50 actúan como extractores
de features fijos que mapean cada imagen a un vector de 2048 dimensiones.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt

from src.config import OUTPUTS, BATCH_SIZE, LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED
from src.data import ImageDataset

torch.manual_seed(SEED)


# === Construcción de modelos ===

def build_feature_extractor() -> nn.Module:
    """
    ResNet50 con backbone completamente congelado.

    Solo el clasificador final (fc) es entrenable. Los pesos del backbone
    se usan como extractores de features genéricos tal como los aprendió
    ResNet50 en ImageNet.

    Parámetros entrenables: 2048 -> 1
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Congela todos los parámetros del backbone
    for param in model.parameters():
        param.requires_grad = False

    # Reemplaza el clasificador original (2048 -> 1000 clases ImageNet)
    # por uno binario (2048 -> 1 logit). Al ser una capa nueva,
    # requires_grad=True por defecto.
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model



# === Entrenamiento ===

def train_transfer(model: nn.Module, lr: float = LR) -> dict:
    """
    Loop de entrenamiento para feature extraction con ResNet50.

    Args:
        model: construido con build_feature_extractor()
        lr: learning rate (por defecto LR de config)

    Returns:
        dict con listas "train_loss", "val_loss", "val_acc" por epoch
    """
    OUTPUTS.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_loader = DataLoader(ImageDataset("train"), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(ImageDataset("val"),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    historial = {"train_loss": [], "val_loss": [], "val_acc": []}
    mejor_val_loss = float("inf")
    epochs_sin_mejora = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # = Train =
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_loader.dataset)

        # = Validation =
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                preds = model(imgs)
                val_loss += criterion(preds, labels).item() * len(imgs)
                correct += ((preds.sigmoid() >= 0.5) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        historial["train_loss"].append(train_loss)
        historial["val_loss"].append(val_loss)
        historial["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)

        # = Early stopping y checkpoint =
        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), OUTPUTS / "transfer_best.pt")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"Early stopping en epoch {epoch}.")
                break

    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    _plot_curvas(historial)
    return historial


def _plot_curvas(historial: dict) -> None:
    epochs = range(1, len(historial["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, historial["train_loss"], label="train")
    ax1.plot(epochs, historial["val_loss"],   label="val")
    ax1.set_title("Loss por epoch")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, historial["val_acc"], color="green", label="val accuracy")
    ax2.set_title("Val Accuracy por epoch")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("ResNet50 — Feature Extraction")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "transfer_curvas.png", dpi=150)
    plt.show()
