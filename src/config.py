"""
Fuente única de verdad para rutas, constantes e hiperparámetros del proyecto.

Cualquier módulo (data.py, mlp.py, cnn.py, predict.py) importa desde acá en vez
de hardcodear valores. Así, cambiar el tamaño de imagen o la semilla se hace
en un solo lugar y se propaga a todo el pipeline automáticamente.
"""
from pathlib import Path
from PIL import Image

# =============================================================================
# CONSTANTES - valores fijos que no se tocan entre experimentos
# =============================================================================

# __file__ es la ruta de este archivo (src/config.py)
# .parent sube a src/, .parent.parent sube a la raíz del proyecto.
ROOT = Path(__file__).parent.parent

# Rutas principales del proyecto
DATA_RAW   = ROOT / "data" / "raw"        # imágenes originales del dataset
DATA_PROC  = ROOT / "data" / "processed"  # imágenes redimensionadas a IMG_SIZE x IMG_SIZE
SPLITS_CSV = ROOT / "data" / "splits.csv" # asignación de cada imagen a train/val/test
OUTPUTS    = ROOT / "outputs"             # modelos guardados, curvas, predicciones

# Extensiones de imagen válidas (set para búsqueda O(1))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Clases del problema (el orden importa: índice 0 = real, índice 1 = fake)
CLASES = ["real", "fake"]

# Mapeo de etiquetas de texto a enteros para la loss function
LABEL_TO_INT = {"real": 0, "fake": 1}

# Splits del dataset raw de Kaggle (estructura de carpetas en data/raw/)
RAW_SPLITS = ["train", "test"]

# Splits del proyecto tras el preprocesamiento (80% / 10% / 10%)
PROC_SPLITS = ["train", "val", "test"]

# Stats de ImageNet para normalización. Necesarias para transfer learning ya que los
# modelos preentrenados fueron entrenados con estas stats. Se usan en todos los modelos
# para que el preprocesamiento sea idéntico y las comparaciones sean justas.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# PIL rechaza imágenes muy grandes por seguridad (protección contra "decompression bombs":
# archivos pequeños que explotan a gigas al decodificarse). El límite por defecto es aprox
# 89M px, pero algunas fotos del dataset superan eso. 200M cubre cámaras profesionales sin
# desactivar la protección completamente (None la desactivaría del todo).
Image.MAX_IMAGE_PIXELS = 200_000_000


# =============================================================================
# HIPERPARÁMETROS - valores que se ajustan entre experimentos
# =============================================================================

# Configuración de imágenes
IMG_SIZE     = 128  # todas las imágenes se redimensionan a este tamaño (cuadradas)
IMG_CHANNELS = 3    # RGB - el color es señal clave para distinguir real de IA

# PCA (solo para el MLP)
# Número de componentes que explican el 90% de la varianza (estimado sobre muestra de 5k imgs)
PCA_N_COMPONENTS = 356

# Split del dataset: 80% train / 10% val / 10% test
# raw/train/ completo -> train | raw/test/ partido 50/50 -> val + test
SEED = 26  # semilla para reproducibilidad de splits y modelos

# Hiperparámetros de entrenamiento (aplican a todos los modelos como punto de partida)
BATCH_SIZE   = 128   # número de imágenes por paso de gradient descentt
LR           = 1e-3  # learning rate del optimizador Adam
WEIGHT_DECAY = 1e-4  # regularización L2 para reducir overfitting
NUM_EPOCHS   = 50    # máximo de epochs
PATIENCE     = 10    # epochs sin mejora en val_loss antes de early stopping