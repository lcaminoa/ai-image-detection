"""
Preprocesamiento y carga de datos.

Se usan una sola vez para preparar el disco:
    build_splits_csv()
    preprocess_dataset()

Se usan cada vez que se entrena:
    dataset = ImageDataset("train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.config import (
    ROOT, DATA_RAW, DATA_PROC, SPLITS_CSV,
    IMG_SIZE, VAL_RATIO, SEED,
)

Image.MAX_IMAGE_PIXELS = 200_000_000

LABEL_TO_INT = {"real": 0, "fake": 1}
_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Parte 1: generar splits.csv
# ---------------------------------------------------------------------------

def build_splits_csv(
    raw: Path = DATA_RAW,
    splits_csv: Path = SPLITS_CSV,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> None:
    """
    Escanea raw/, hace el split estratificado train/val y guarda splits.csv.
    """
    paths, labels = [], []
    for label in ("real", "fake"):
        for p in sorted((raw / "train" / label).iterdir()):
            if p.is_file() and p.suffix.lower() in _EXTS:
                paths.append(p)
                labels.append(label)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
    )

    rows = []
    for p, l in zip(train_paths, train_labels):
        rows.append({"path": str(p.relative_to(ROOT)), "split": "train", "label": l})
    for p, l in zip(val_paths, val_labels):
        rows.append({"path": str(p.relative_to(ROOT)), "split": "val", "label": l})
    for label in ("real", "fake"):
        for p in sorted((raw / "test" / label).iterdir()):
            if p.is_file() and p.suffix.lower() in _EXTS:
                rows.append({"path": str(p.relative_to(ROOT)), "split": "test", "label": label})

    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(splits_csv, index=False)

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n_real = len(sub[sub["label"] == "real"])
        n_fake = len(sub[sub["label"] == "fake"])
        print(f"{split:5s}: {len(sub):>6} imgs  ({n_real} real / {n_fake} fake)")


# ---------------------------------------------------------------------------
# Parte 2: redimensionar y guardar en processed/
# ---------------------------------------------------------------------------

def preprocess_dataset(
    processed: Path = DATA_PROC,
    splits_csv: Path = SPLITS_CSV,
    img_size: int = IMG_SIZE,
) -> None:
    """
    Redimensiona todas las imágenes a img_size x img_size y las guarda en processed/.
    """
    df = pd.read_csv(splits_csv)
    total = len(df)

    for _, row in tqdm(df.iterrows(), total=total, desc="Procesando imágenes"):
        src = ROOT / row["path"]
        dst = processed / row["split"] / row["label"] / (src.stem + ".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            continue

        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img = img.resize((img_size, img_size), Image.LANCZOS)
                img.save(dst, format="JPEG", quality=95)
        except Exception as e:
            print(f"\nSkip imagen corrupta: {src.name} ({e})")

    print(f"Listo. {total} imágenes guardadas en {processed}")


# ---------------------------------------------------------------------------
# Parte 3: Dataset de PyTorch
# ---------------------------------------------------------------------------

def get_transforms(split: str):
    """
    train: augmentation liviana + normalización ImageNet.
    val / test: solo normalización ImageNet.
    """
    from torchvision import transforms
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            *normalize,
        ])
    return transforms.Compose(normalize)


class ImageDataset:
    """
    Carga imágenes desde processed/ y aplica las transformaciones del split.
    DataLoader solo requiere __len__ y __getitem__, no herencia explícita de Dataset.
    """

    def __init__(self, split: str, processed: Path = DATA_PROC, transform=None):
        df = pd.read_csv(SPLITS_CSV)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.processed = processed
        self.transform = transform if transform is not None else get_transforms(split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = (self.processed / row["split"] / row["label"] / (Path(row["path"]).stem + ".jpg"))
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        label = LABEL_TO_INT[row["label"]]
        return self.transform(img), label
