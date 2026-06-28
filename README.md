# Clasificación de Imágenes: Real vs. Generada por IA

Proyecto final de la materia Machine Learning (I302) - Universidad de San Andrés.

Se comparan cuatro modelos de clasificación binaria para distinguir imágenes reales de imágenes generadas por inteligencia artificial: un MLP con reducción PCA, una CNN entrenada desde cero, y ResNet50 con dos estrategias de transfer learning (Feature Extraction y Fine-Tuning).

**Dataset:** [AI-Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data) - Kaggle

---

## Resultados (test set)

| Modelo | Accuracy | F1 |
|--------|----------|----|
| MLP + PCA | 0.7237 | 0.7382 |
| CNN (desde cero) | 0.8843 | 0.8815 |
| ResNet50 - Feature Extraction | 0.8808 | 0.8813 |
| ResNet50 - Fine-tuning | **0.9335** | **0.9341** |

---

## Estructura del proyecto

```
PF_ML_Krinisky_Caminoa/
├── data/
│   ├── raw/              # imágenes originales del dataset (no versionado)
│   ├── processed/        # imágenes redimensionadas a 224x224 (no versionado)
│   └── splits.csv        # asignación de cada imagen a train / val / test
├── notebooks/
│   ├── 01_eda.ipynb                      # análisis exploratorio del dataset
│   ├── 02_MLP.ipynb                      # PCA + entrenamiento MLP
│   ├── 03_CNN.ipynb                      # entrenamiento CNN
│   ├── 04_transfer_learning.ipynb        # feature extraction y fine-tuning con ResNet50
│   ├── 05_results.ipynb                  # evaluación y comparación de modelos
│   └── 06_custom_image_prediction.ipynb  # predicción sobre una imagen propia
├── outputs/
│   ├── mlp_best.pt       # pesos del mejor MLP
│   ├── cnn_best.pt       # pesos de la mejor CNN
│   ├── fe_best.pt        # pesos del mejor modelo de feature extraction
│   ├── ft_best.pt        # pesos del mejor modelo de fine-tuning
│   └── pca_mlp.joblib    # transformación PCA (ver nota más abajo)
├── src/
│   ├── config.py         # rutas, constantes e hiperparámetros centralizados
│   ├── data.py           # splits, preprocesamiento y Dataset de PyTorch
│   ├── mlp.py            # arquitectura y entrenamiento del MLP
│   ├── cnn.py            # arquitectura y entrenamiento de la CNN
│   └── transfer.py       # construcción y entrenamiento de modelos ResNet50
└── requirements.txt
```

---

## Instalación

Se requiere Python 3.10 o superior.

```bash
git clone <url-del-repositorio>
cd PF_ML_Krinisky_Caminoa
pip install -r requirements.txt
```

---

## Preparación del dataset

1. Descargar el dataset desde [Kaggle](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data).
2. Descomprimir y colocar las imágenes con la siguiente estructura:

```
data/raw/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

El preprocesamiento (generación de `splits.csv` y redimensionado a 224x224) se ejecuta automáticamente al correr `01_eda.ipynb`.

---

## Ejecución

### Solo inferencia (modelos ya entrenados)

Los pesos de los cuatro modelos están incluidos en el repositorio dentro de `outputs/`. Para evaluar los modelos o probar una imagen propia, solo es necesario generar `pca_mlp.joblib` corriendo la celda de ajuste de PCA en `02_MLP.ipynb` (sin necesidad de reentrenar el MLP).

Luego se pueden usar directamente:
- `05_results.ipynb` - evaluación completa sobre el test set
- `06_custom_image_prediction.ipynb` - predicción sobre cualquier imagen

### Entrenamiento completo desde cero

Correr los notebooks en orden:

| # | Notebook | Genera |
|---|----------|--------|
| 01 | `01_eda.ipynb` | `data/splits.csv`, `data/processed/` |
| 02 | `02_MLP.ipynb` | `outputs/pca_mlp.joblib`, `outputs/mlp_best.pt` |
| 03 | `03_CNN.ipynb` | `outputs/cnn_best.pt` |
| 04 | `04_transfer_learning.ipynb` | `outputs/fe_best.pt`, `outputs/ft_best.pt` |
| 05 | `05_results.ipynb` | - (solo lectura) |

> **GPU recomendada** para los notebooks 03 y 04. El código detecta automáticamente MPS (Apple Silicon), CUDA (NVIDIA) o cae a CPU.

---

## Nota sobre `pca_mlp.joblib`

Este archivo (aprox. 127 MB) supera el límite de tamaño de GitHub y no está versionado. Es necesario generarlo corriendo `02_MLP.ipynb` antes de usar el MLP en los notebooks de resultados o predicción.

---

## Limitación del modelo

El dataset de entrenamiento contiene imágenes de paisajes, naturaleza, arquitectura y arte. Los modelos **no generalizan a retratos ni selfies**, ya que ese tipo de imágenes está fuera de la distribución de entrenamiento. El propósito del modelo es detectar si una imagen de ese dominio específico fue generada o modificada por IA.

---

## Autores

- Lautaro Valentín Caminoa
- Francisco Krinisky

Universidad de San Andrés - Machine Learning (I302), 2026