# Proyecto: Severstal Steel Defect Detection (Segmentación)

Este proyecto entrena y evalúa un modelo de **segmentación** (4 clases) para el dataset de la competencia **Severstal: Steel Defect Detection**.

## Importante (para tu caso)
- **No** descarga datos desde Kaggle (por requisito).
- Tú debes **descargar** el dataset manualmente y colocarlo en la carpeta indicada abajo.

---

## 1) Requisitos
- Python 3.10+ (recomendado)
- GPU opcional (si tienes CUDA instalado, entrenará mucho más rápido)

---

## 2) Estructura esperada de datos

Coloca los archivos del dataset en:

```
data/raw/severstal/
├── train.csv
├── sample_submission.csv
├── train_images/
│   ├── xxx.jpg
│   └── ...
└── test_images/
    ├── yyy.jpg
    └── ...
```

> Nota: este repo no incluye datos. Solo código.

---

## 3) Instalación

En VS Code (terminal):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 4) Flujo recomendado (paso a paso)

### A) Verificar y analizar el dataset (EDA)
Genera estadísticas y gráficas en `outputs/eda/`:

```bash
python -m src.cli.eda --data_dir data/raw/severstal --out_dir outputs/eda
```

### B) Crear splits: train / val / test
Divide por `ImageId` (evita fuga de información).

```bash
python -m src.cli.make_splits --data_dir data/raw/severstal --out_dir outputs/splits --seed 42
```

Esto crea:
- `outputs/splits/train.csv`
- `outputs/splits/val.csv`
- `outputs/splits/test.csv`

### C) Entrenar (baseline)
```bash
python -m src.cli.train --data_dir data/raw/severstal --splits_dir outputs/splits --out_dir outputs/run_baseline
```

### D) Evaluar en test
```bash
python -m src.cli.evaluate --data_dir data/raw/severstal --splits_dir outputs/splits --ckpt outputs/run_baseline/best.pt --out_dir outputs/run_baseline
```

---

## 5) Mejoras (opcional)
Entrenar un modelo mejorado (pérdida Tversky + búsqueda de umbral por clase):

```bash
python -m src.cli.train --data_dir data/raw/severstal --splits_dir outputs/splits --out_dir outputs/run_improved --loss tversky --encoder efficientnet_b2 --epochs 30
python -m src.cli.tune_thresholds --data_dir data/raw/severstal --splits_dir outputs/splits --ckpt outputs/run_improved/best.pt --out_dir outputs/run_improved
python -m src.cli.evaluate --data_dir data/raw/severstal --splits_dir outputs/splits --ckpt outputs/run_improved/best.pt --out_dir outputs/run_improved --thresholds outputs/run_improved/thresholds.json
```

---

## 6) Qué entrega el proyecto (lo que pide la tarea)
✅ División train/val/test  
✅ Análisis del dataset (EDA)  
✅ Preprocesamiento (normalizar, redimensionar, augmentations) explicado en `REPORT.md`  
✅ Arquitectura elegida y justificada (U-Net + backbone)  
✅ Curvas de aprendizaje (loss y Dice) guardadas en `outputs/.../plots/`  
✅ Métricas cuantitativas (Dice por clase, mean Dice) y análisis cualitativo (visualizaciones)  
✅ Mejoras aplicadas (opcional) y propuestas finales

---

## 7) Reporte
Lee y completa/ajusta `REPORT.md` (ya está redactado en formato académico y se llena con tus resultados).
