# Funciones de exploración del dataset

from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import random

from src.config import IMG_EXTS, CLASES, RAW_SPLITS
# Al importar config.py también se configura Image.MAX_IMAGE_PIXELS = 200_000_000


def listar_imagenes(carpeta: Path):
    """
    Devuelve una lista con los Paths de las imágenes en 'carpeta'.
    Filtra por extensión para ignorar otros archivos (.DS_Store, etc.).
    """
    if not carpeta.exists():
        print(f"No existe: {carpeta}")
        return []
    return [p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def contar_dataset(raw: Path):
    """
    Recorre las 4 combinaciones de carpeta (train/real, train/fake,
    test/real, test/fake) del dataset raw, cuenta cuántas imágenes hay
    en cada una, imprime el resumen, y devuelve un diccionario:
        {(split, clase): [paths]}
    """
    conteo = {}
    for split in RAW_SPLITS: # Para train y test (dataset raw)
        for clase in CLASES: # Para real y fake
            imgs = listar_imagenes(raw / split / clase)
            conteo[(split, clase)] = imgs
            print(f"{split}/{clase:5s}: {len(imgs):>6} imágenes")
    total = sum(len(v) for v in conteo.values())
    print(f"\nTOTAL: {total} imágenes")
    return conteo


def mostrar_muestras(conteo, split="train", n=4):
    """
    Muestra n imágenes 'real' y n 'fake' lado a lado.

    Conteo es el dict que devuelve contar_dataset
    Las imágenes son del split q se especifique.
    """
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    for fila, clase in enumerate(CLASES):
        muestras = random.sample(conteo[(split, clase)], n)
        for col, ruta in enumerate(muestras):
            img = Image.open(ruta)
            axes[fila, col].imshow(img)
            axes[fila, col].set_title(f"{clase}\n{img.size}", fontsize=9)
            axes[fila, col].axis("off")
    plt.tight_layout()
    plt.show()


def balance_clases(conteo):
    """
    Imprime la proporción real/fake en cada split.

    Conteo es el dict que devuelve contar_dataset.
    """
    for split in RAW_SPLITS:
        n_real = len(conteo[(split, "real")])
        n_fake = len(conteo[(split, "fake")])
        tot = n_real + n_fake
        if tot == 0:
            continue
        print(f"\n{split}:")
        print(f"  real: {n_real:>6} ({100*n_real/tot:.1f}%)")
        print(f"  fake: {n_fake:>6} ({100*n_fake/tot:.1f}%)")


def muestrear_dimensiones(paths, n=200):
    """
    Cuenta los tamaños (ancho, alto) sobre una muestra de imágenes.
    Muestreo porque abrir todas sería lentísimo con el dataset
    de 52 GB.
    """
    muestra = random.sample(paths, min(n, len(paths)))
    dims = defaultdict(int)
    for p in muestra:
        with Image.open(p) as img:
            dims[img.size] += 1
    return dims


def reporte_dimensiones(conteo, n=200):
    """
    Muestra los tamaños más comunes de cada carpeta.
    """
    for (split, clase), paths in conteo.items():
        if not paths:
            continue
        dims = muestrear_dimensiones(paths, n)
        print(f"\n{split}/{clase} - tamaños más comunes (sobre muestra):")
        for size, count in sorted(dims.items(), key=lambda x: -x[1])[:5]:
            print(f"  {size}: {count}")