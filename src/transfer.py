"""
Transfer learning con ResNet18 preentrenado en ImageNet.

Dos estrategias comparadas:
  1. Feature extraction: backbone congelado, solo se entrena el clasificador.
     Rápido y sirve como piso de comparación.
  2. Fine-tuning: layer4 + clasificador descongelados con LR pequeño.
     Adapta las features profundas a los patrones específicos de imágenes IA.

Detectar imágenes IA tiene patrones únicos (artefactos de difusión, texturas
sintéticas) que ImageNet no captura directamente. Por eso el fine-tuning
debería superar al feature extraction.

Uso:
    from src.transfer import build_feature_extractor, build_fine_tuner, train_transfer

    model_fe = build_feature_extractor()
    historial_fe = train_transfer(model_fe, run_name="fe")

    model_ft = build_fine_tuner()
    historial_ft = train_transfer(model_ft, run_name="ft", lr=1e-4)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights  # ResNet18 solo tiene IMAGENET1K_V1
import matplotlib.pyplot as plt

from src.config import (
    OUTPUTS, BATCH_SIZE, LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
)
from src.data import ImageDataset

torch.manual_seed(SEED)


# =============================================================================
# Construcción de modelos
# =============================================================================

def build_feature_extractor() -> nn.Module:
    """
    ResNet18 con backbone completamente congelado.

    Solo el clasificador final (fc) es entrenable. Los pesos del backbone
    se usan como extractores de features genéricos tal como los aprendió
    ResNet18 en ImageNet — sin modificarlos.

    Parámetros entrenables: ~513 (fc: 512 -> 1)
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Congela todos los parámetros del backbone
    for param in model.parameters():
        param.requires_grad = False

    # Reemplaza el clasificador original (512 -> 1000 clases ImageNet)
    # por uno binario (512 -> 1 logit). Al ser una capa nueva,
    # requires_grad=True por defecto.
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model


def build_fine_tuner() -> nn.Module:
    """
    ResNet18 con layer4 + clasificador descongelados.

    Las primeras capas (layer1-3) detectan bordes y texturas genéricas que
    transfieren bien. Solo descongelamos layer4 (la más profunda, detecta
    estructuras de alto nivel) para que se adapte a los artefactos de
    imágenes IA sin destruir los features base.

    Parámetros entrenables: ~2.1M (layer4 ~2.1M + fc ~513)
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Congela todo el backbone
    for param in model.parameters():
        param.requires_grad = False

    # Descongela solo layer4 (las últimas capas residuales del backbone)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Nuevo clasificador binario (entrenable por defecto)
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model


# =============================================================================
# Entrenamiento
# =============================================================================

def train_transfer(model: nn.Module, run_name: str, lr: float = LR) -> dict:
    """
    Loop de entrenamiento genérico para ambas estrategias.

    Args:
        model:    construido con build_feature_extractor() o build_fine_tuner()
        run_name: "fe" o "ft" — determina el nombre del checkpoint y las curvas
        lr:       usar LR (1e-3) para feature extraction y 1e-4 para fine-tuning.
                  El LR chico en fine-tuning evita destruir los pesos preentrenados.

    Returns:
        dict con listas "train_loss", "val_loss", "val_acc" por epoch.
    """
    OUTPUTS.mkdir(exist_ok=True)
    # Prioriza GPU: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_loader = DataLoader(ImageDataset("train"), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ImageDataset("val"),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Optimiza solo los parámetros con requires_grad=True:
    # feature extraction -> solo fc | fine-tuning -> layer4 + fc
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    historial = {"train_loss": [], "val_loss": [], "val_acc": []}
    mejor_val_loss = float("inf")
    epochs_sin_mejora = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # — Train —
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_loader.dataset)

        # — Validation —
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                preds  = model(imgs)
                val_loss += criterion(preds, labels).item() * len(imgs)
                correct  += ((preds.sigmoid() >= 0.5) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        historial["train_loss"].append(train_loss)
        historial["val_loss"].append(val_loss)
        historial["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)

        # — Early stopping y checkpoint —
        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), OUTPUTS / f"{run_name}_best.pt")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"Early stopping en epoch {epoch}.")
                break

    print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
    _plot_curvas(historial, run_name)
    return historial


def _plot_curvas(historial: dict, run_name: str) -> None:
    """
    Grafica loss y accuracy por epoch y guarda la figura en outputs/.
    """
    titulo = "Feature Extraction" if run_name == "fe" else "Fine-tuning"
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

    plt.suptitle(f"ResNet18 — {titulo}")
    plt.tight_layout()
    plt.savefig(OUTPUTS / f"{run_name}_curvas.png", dpi=150)
    plt.show()
