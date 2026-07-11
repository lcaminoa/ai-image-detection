# Preprocesamiento y carga de datos

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.config import (
    ROOT, DATA_RAW, DATA_PROC, SPLITS_CSV,
    IMG_SIZE, SEED,
    IMG_EXTS, CLASES, LABEL_TO_INT, IMAGENET_MEAN, IMAGENET_STD,
    # Image.MAX_IMAGE_PIXELS ya se configuró al importar config.py
)


# === 1) generar splits.csv ===

def build_splits_csv(raw: Path = DATA_RAW, splits_csv: Path = SPLITS_CSV, seed: int = SEED) -> None:
    """
    Genera splits.csv con proporción 80/10/10:
      - raw/train/ completo -> split "train" (48,000 imágenes)
      - raw/test/  50/50 estratificado -> split "val" y "test" (6,000 c/u)

    El split es estratificado para mantener 50% real / 50% fake en val y test.
    Los paths se guardan relativos a ROOT para que el CSV sea portable.
    """
    rows = []

    # Todo raw/train va directo a train sin dividir
    for label in CLASES:
        for p in sorted((raw / "train" / label).iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(p.relative_to(ROOT)), "split": "train", "label": label})

    # raw/test se parte 50/50 estratificado en val y test
    # stratify=test_labels garantiza que ambas mitades tengan el mismo balance de clases
    test_paths, test_labels = [], []
    for label in CLASES:
        for p in sorted((raw / "test" / label).iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                test_paths.append(p)
                test_labels.append(label)

    val_paths, held_paths, val_labels, held_labels = train_test_split(
        test_paths, test_labels,
        test_size=0.5,
        stratify=test_labels,
        random_state=seed,
    )
    for p, l in zip(val_paths, val_labels):
        rows.append({"path": str(p.relative_to(ROOT)), "split": "val", "label": l})
    for p, l in zip(held_paths, held_labels):
        rows.append({"path": str(p.relative_to(ROOT)), "split": "test", "label": l})

    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(splits_csv, index=False)

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n_real = len(sub[sub["label"] == "real"])
        n_fake = len(sub[sub["label"] == "fake"])
        print(f"{split:5s}: {len(sub):>6} imgs  ({n_real} real / {n_fake} fake)")


# === 2) redimensionar y guardar en processed/ ===

def preprocess_dataset(processed: Path = DATA_PROC, splits_csv: Path = SPLITS_CSV, img_size: int = IMG_SIZE) -> None:
    """
    Redimensiona todas las imágenes a img_size x img_size y las guarda en processed/.

    Se hace una sola vez: las imágenes procesadas son mucho más pequeñas que las
    originales (224x224 vs hasta 6000x4000), acelerando muchísimo el entrenamiento.
    """
    df = pd.read_csv(splits_csv)
    total = len(df)

    for _, row in tqdm(df.iterrows(), total=total, desc="Procesando imágenes"):
        src = ROOT / row["path"]
        # La imagen procesada se guarda con extensión .jpg para uniformidad,
        # independientemente del formato original (png, webp, etc.)
        dst = processed / row["split"] / row["label"] / (src.stem + ".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            continue # Ya procesada en una corrida anterior

        try:
            with Image.open(src) as img:
                img = img.convert("RGB") # normaliza paletas, RGBA, etc.
                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS) # LANCZOS = mejor calidad al reducir
                img.save(dst, format="JPEG", quality=95)
        except Exception as e:
            print(f"\nSkip imagen corrupta: {src.name} ({e})")

    # Elimina del CSV las entradas que no tienen archivo procesado
    # (imágenes corruptas que fueron skipeadas arriba)
    df["_dst"] = df.apply(
        lambda r: processed / r["split"] / r["label"] / (Path(r["path"]).stem + ".jpg"), axis=1
    )
    validas = df["_dst"].apply(lambda p: p.exists())
    n_corruptas = (~validas).sum()
    if n_corruptas:
        df = df[validas].drop(columns="_dst")
        df.to_csv(splits_csv, index=False)
        print(f"Se eliminaron {n_corruptas} imágenes corruptas del CSV.")
    else:
        df = df.drop(columns="_dst")

    print(f"Listo. {len(df)} imágenes en {processed}")


# === 3) Dataset de PyTorch ===

def get_transforms(split: str):
    """
    Devuelve las transformaciones de torchvision para el split indicado.

    - train: augmentation liviana (flip, rotación, brillo) + normalización ImageNet.
      El augmentation se aplica en el momento para que cada epoch vea variantes distintas.
    - val / test: solo normalización. No se augmenta para que las métricas sean estables.
    """
    from torchvision import transforms  # importado acá adentro y no al principio del archivo para que
                                        # build_splits_csv() y preprocess_dataset() funcionen sin torch
    normalize = [
        transforms.ToTensor(), # [0,255] uint8 -> [0,1] float32
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Centra en media ImageNet
    ]
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(), # Las imágenes reales/IA no tienen orientación preferida
            transforms.RandomRotation(10), # Rotación leve para más variedad
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simula distintas condiciones de iluminación
            *normalize,
        ])
    return transforms.Compose(normalize)


class ImageDataset:
    """
    Dataset de PyTorch que lee imágenes desde data/processed/ y aplica transformaciones.

    No hereda de torch.utils.data.Dataset explícitamente porque DataLoader solo
    requiere que el objeto tenga __len__ y __getitem__ (duck typing).
    El import de torch se hace lazy en get_transforms() para que este módulo
    pueda importarse sin torch instalado (útil para build_splits_csv y preprocess_dataset).
    """

    def __init__(self, split: str, processed: Path = DATA_PROC, transform=None):
        df = pd.read_csv(SPLITS_CSV)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.processed = processed
        # Si no se pasa transform, usa el estándar para el split (con o sin augmentation)
        self.transform = transform if transform is not None else get_transforms(split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Reconstruye la ruta en processed/ a partir del stem del path original
        img_path = (self.processed / row["split"] / row["label"] / (Path(row["path"]).stem + ".jpg"))
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        label = LABEL_TO_INT[row["label"]]
        return self.transform(img), label
