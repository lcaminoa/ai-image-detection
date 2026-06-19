"""
Fuente única de verdad para rutas e hiperparámetros del proyecto.

Cualquier módulo (data.py, train.py, predict.py) importa desde acá en vez
de hardcodear valores. Así, cambiar el tamaño de imagen o la semilla se hace
en un solo lugar y se propaga a todo el pipeline automáticamente.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
SPLITS_CSV = ROOT / "data" / "splits.csv"

IMG_SIZE = 128
IMG_CHANNELS = 3

VAL_RATIO = 0.2
SEED = 42
