# Clasificación de Imágenes: Real vs. Generada por IA

Se desarrollan y comparan cinco modelos de clasificación binaria para distinguir imágenes reales de imágenes generadas por inteligencia artificial: un MLP con reducción PCA, una CNN entrenada desde cero, ResNet50 con dos estrategias de transfer learning (Feature Extraction y Fine-Tuning), y un quinto modelo de fine-tuning incremental orientado a la adaptación de dominio sobre imágenes de redes sociales.

**Demo interactiva:** [panchokrinisky.github.io/web-ai-image-detection](https://panchokrinisky.github.io/web-ai-image-detection/) - subí una imagen y el modelo te dice si es real o generada por IA.

**Dataset principal:** [AI-Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data) - Kaggle

**Dataset de adaptación:** [itw-sm](https://huggingface.co/datasets/dkarageo/itw-sm) - HuggingFace (imágenes de Facebook, X, Instagram y LinkedIn)

---

## Resultados

### Dominio original (paisajes, naturaleza, arte)

| Modelo | Accuracy | F1 |
|--------|----------|----|
| MLP + PCA | 0.7237 | 0.7382 |
| CNN (desde cero) | 0.8843 | 0.8815 |
| ResNet50 - Feature Extraction | 0.8808 | 0.8813 |
| ResNet50 - Fine-tuning | **0.9335** | **0.9341** |
| ResNet50 - Fine-tuning Adaptado | 0.8669 | 0.8733 |

### Dominio redes sociales (itw-sm)

| Modelo | Accuracy | F1 |
|--------|----------|----|
| ResNet50 - Fine-tuning | 0.6250 | 0.7220 |
| ResNet50 - Fine-tuning Adaptado | **0.9450** | **0.9453** |

---

## Estructura del proyecto

```
PF_ML_Krinisky_Caminoa/
├── data/
│   ├── raw/              # imágenes originales del dataset (no versionado)
│   ├── processed/        # imágenes redimensionadas a 224x224 (no versionado)
│   └── splits.csv        # asignación de cada imagen a train / val / test
├── informe/
│   ├── Caminoa_Krinisky_Informe_PF.pdf   # informe final
│   ├── informe.tex                       # fuente LaTeX del informe
│   └── figs/                             # figuras incluidas en el informe
├── notebooks/
│   ├── 01_eda.ipynb                  # análisis exploratorio del dataset
│   ├── 02_MLP.ipynb                  # PCA + entrenamiento MLP
│   ├── 03_CNN.ipynb                  # entrenamiento CNN
│   ├── 04_transfer_learning.ipynb    # feature extraction y fine-tuning con ResNet50
│   ├── 05_results.ipynb              # evaluación y comparación de los cuatro modelos base
│   ├── 06_generalizacion_itwsm.ipynb # evaluación de los modelos base sobre imágenes de redes sociales
│   ├── 07_domain_adaptation.ipynb    # fine-tuning incremental sobre imágenes de redes sociales
│   ├── 08_prediccion_imagen_propia.ipynb  # predicción de los cinco modelos sobre una imagen propia
│   └── imgs/                         # carpeta para colocar la imagen a predecir
├── outputs/
│   ├── mlp_best.pt       # pesos del mejor MLP
│   ├── cnn_best.pt       # pesos de la mejor CNN
│   ├── fe_best.pt        # pesos del mejor modelo de feature extraction
│   ├── ft_best.pt        # pesos del mejor modelo de fine-tuning
│   ├── ft_adapted.pt     # pesos del modelo adaptado a redes sociales
│   └── pca_mlp.joblib    # transformación PCA (ver nota más abajo)
├── src/
│   ├── config.py         # rutas, constantes e hiperparámetros centralizados
│   ├── data.py           # splits, preprocesamiento y Dataset de PyTorch
│   ├── exploracion.py    # funciones auxiliares para el EDA (conteo, visualización, dimensiones)
│   ├── mlp.py            # arquitectura y entrenamiento del MLP
│   ├── cnn.py            # arquitectura y entrenamiento de la CNN
│   └── transfer.py       # construcción y entrenamiento de modelos ResNet50
└── requirements.txt
```

---

## Instalación

Se requiere Python 3.10 o superior.

```bash
git clone https://github.com/lcaminoa/ai-image-detection.git
cd ai-image-detection
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

### Dataset de adaptación (itw-sm)

El dataset `itw-sm` utilizado en los notebooks 06 y 07 requiere solicitar acceso a sus autores desde [HuggingFace](https://huggingface.co/datasets/dkarageo/itw-sm). Una vez aprobado el acceso, se descarga automáticamente al correr los notebooks.

---

## Ejecución

### Solo inferencia (modelos ya entrenados)

Los pesos de los cinco modelos están incluidos en el repositorio dentro de `outputs/`. Para evaluar los modelos o probar una imagen propia, solo es necesario generar `pca_mlp.joblib` corriendo la celda de ajuste de PCA en `02_MLP.ipynb` (sin necesidad de reentrenar el MLP).

Luego se pueden usar directamente:
- `05_results.ipynb` - evaluación completa de los cuatro modelos base sobre el test set
- `06_generalizacion_itwsm.ipynb` - evaluación de los modelos base sobre imágenes de redes sociales
- `07_domain_adaptation.ipynb` - evaluación del modelo adaptado a redes sociales
- `08_prediccion_imagen_propia.ipynb` - predicción de los cinco modelos sobre una imagen propia

### Entrenamiento completo desde cero

Correr los notebooks en orden:

| # | Notebook | Genera |
|---|----------|--------|
| 01 | `01_eda.ipynb` | `data/splits.csv`, `data/processed/` |
| 02 | `02_MLP.ipynb` | `outputs/pca_mlp.joblib`, `outputs/mlp_best.pt` |
| 03 | `03_CNN.ipynb` | `outputs/cnn_best.pt` |
| 04 | `04_transfer_learning.ipynb` | `outputs/fe_best.pt`, `outputs/ft_best.pt` |
| 05 | `05_results.ipynb` | - (solo lectura) |
| 06 | `06_generalizacion_itwsm.ipynb` | - (solo lectura) |
| 07 | `07_domain_adaptation.ipynb` | `outputs/ft_adapted.pt` |
| 08 | `08_prediccion_imagen_propia.ipynb` | - (solo lectura) |

> **GPU recomendada** para los notebooks 03, 04 y 07. El código detecta automáticamente MPS (Apple Silicon), CUDA (NVIDIA) o cae a CPU.

---

## Nota sobre `pca_mlp.joblib`

Este archivo (aprox. 127 MB) supera el límite de tamaño de GitHub y no está versionado. Es necesario generarlo corriendo `02_MLP.ipynb` antes de usar el MLP en los notebooks de resultados o predicción.

---

## Autores

- Lautaro Valentín Caminoa
- Francisco Krinisky

Universidad de San Andrés - Aprendizaje Automático y Aprendizaje Profundo, 2026