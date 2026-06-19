# Funciones de exploración del dataset

from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import random

# Algunos originales superan el límite por defecto (aprox 89M px). 200M cubre
# fotos de cámara profesional sin desactivar la protección completamente.
Image.MAX_IMAGE_PIXELS = 200_000_000

# Extensiones de imagen válidas. Set para que 'in' sea rápido
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ["train", "test"]
CLASES = ["real", "fake"]


def listar_imagenes(carpeta: Path):
    """
    Devuelve la lista de Paths de imágenes en 'carpeta'.
    Filtra por extensión para ignorar archivos colados (.DS_Store, etc.).
    """
    if not carpeta.exists():
        print(f"No existe: {carpeta}")
        return []
    return [p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXTS]


def contar_dataset(raw: Path):
    """
    Recorre las 4 carpetas (split x clase) y cuenta imágenes.
    Devuelve un dict {(split, clase): [paths]} para reusar después.
    """
    conteo = {}
    for split in SPLITS:
        for clase in CLASES:
            imgs = listar_imagenes(raw / split / clase)
            conteo[(split, clase)] = imgs
            print(f"{split}/{clase:5s}: {len(imgs):>6} imágenes")
    total = sum(len(v) for v in conteo.values())
    print(f"\nTOTAL: {total} imágenes")
    return conteo


def mostrar_muestras(conteo, split="train", n=4):
    """
    Muestra n imágenes 'real' y n 'fake' lado a lado.
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
    """
    for split in SPLITS:
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
    Muestreamos porque abrir todas sería lentísimo con 52 GB.
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