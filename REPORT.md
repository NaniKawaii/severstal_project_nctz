# Informe — Redes Neuronales: Severstal Steel Defect Detection (Segmentación)

> Este documento está listo para entregar. Solo reemplaza los valores numéricos (Dice, épocas, etc.) con los resultados que te genere el proyecto en `outputs/`.

---

## 1. Objetivo

Desarrollar un modelo de **segmentación semántica** para identificar defectos en láminas de acero en el dataset de la competencia **Severstal Steel Defect Detection**.  
El problema se plantea como segmentación multi-clase con **4 clases** y etiquetas en formato **RLE (Run-Length Encoding)**.

---

## 2. Análisis del dataset (EDA)

### 2.1 Estructura de etiquetas
El archivo `train.csv` contiene las máscaras por combinación `ImageId_ClassId`.  
Para evitar fuga de información, el proyecto agrupa el CSV para obtener **una fila por imagen** con hasta 4 máscaras (una por clase).

### 2.2 Distribución de defectos
En el EDA se reportan:
- Porcentaje de imágenes con al menos un defecto.
- Porcentaje por clase (1..4).
- Distribución aproximada de áreas (pixeles) por clase.

**Hallazgos típicos observados:**
- Existe un **desbalance**: muchas imágenes no tienen defectos.
- Algunas clases aparecen con menor frecuencia.
- Defectos delgados/pequeños son comunes, lo que dificulta la segmentación.

> Evidencia: gráficas y tablas generadas en `outputs/eda/`.

---

## 3. División Train / Validation / Test

Se realizó una división en:
- Train: 70%
- Validation: 15%
- Test: 15%

La división se hace por `ImageId` (no por filas del CSV) y estratifica por `has_defect` (imagen con/sin defectos) para mantener proporciones similares entre splits.

> Archivos generados: `outputs/splits/train.csv`, `val.csv`, `test.csv`.

---

## 4. Preprocesamiento

### 4.1 Normalización
Se normaliza con media y desviación estándar tipo ImageNet cuando se utiliza un encoder preentrenado.  
**Motivo:** alinear la distribución de entrada con el preentrenamiento mejora estabilidad y convergencia.

### 4.2 Redimensionamiento
Las imágenes se redimensionan a **256×1024** (configurable).  
**Motivo:** reducir costo computacional y permitir batch sizes manejables, manteniendo suficiente resolución para defectos finos.

### 4.3 Aumentación de datos
Se aplican aumentos suaves:
- Volteo horizontal
- Variación de brillo/contraste
- Rotaciones pequeñas y desplazamientos leves

**Motivo:** mejorar generalización sin deformar la geometría del defecto.

---

## 5. Arquitectura del modelo

### 5.1 Modelo base
Se utilizó un modelo tipo **U-Net** con backbone/encoder **EfficientNet** preentrenado en ImageNet.

**Razones de elección:**
- U-Net es un estándar fuerte en segmentación por su diseño encoder-decoder y skip connections.
- Backbones modernos (EfficientNet) extraen características con buena eficiencia.

### 5.2 Salida del modelo
- 4 canales (uno por clase).
- Activación final: **sigmoid** (multi-label por pixel).

---

## 6. Funciones de pérdida y métricas

### 6.1 Pérdida (baseline)
Se emplea una combinación:
- **BCEWithLogits**
- **Dice Loss**

\[
Loss = w_{BCE} \cdot BCE + w_{Dice} \cdot (1 - Dice)
\]

### 6.2 Métricas
- Dice por clase
- Mean Dice (promedio clases)

Estas métricas son apropiadas para segmentación con desbalance y se alinean con el objetivo del problema.

---

## 7. Curvas de aprendizaje y evaluación

### 7.1 Curvas (cualitativo)
Se analiza:
- Tendencia de la loss en train/val.
- Tendencia del Dice en train/val.
- Señales de overfitting (divergencia train vs val).

> Evidencia: `outputs/run_*/plots/loss_curve.png` y `dice_curve.png`.

### 7.2 Métricas (cuantitativo)
**Resultados baseline (ejemplo, reemplazar):**
- Val mean Dice: **X.XXXX**
- Test mean Dice: **X.XXXX**
- Dice por clase (test): **[c1, c2, c3, c4]**

---

## 8. Mejoras aplicadas (si se usó el modo improved)

Se probaron mejoras como:
- Pérdida **Tversky** para controlar el balance entre falsos positivos y falsos negativos.
- Backbone más fuerte (por ejemplo EfficientNet-B2).
- Ajuste de umbral por clase mediante búsqueda en validación.

**Comparación esperada:**
- Mayor mean Dice y/o mejor balance por clase, especialmente en clases raras.

---

## 9. Conclusiones y trabajo futuro

### Conclusiones
- La combinación U-Net + backbone preentrenado es adecuada para defectos industriales.
- El desbalance del dataset exige pérdidas robustas y/o umbrales calibrados.
- Aumentación suave contribuye a generalización.

### Mejoras propuestas (trabajo futuro)
- Entrenamiento por parches (tiling) para defectos pequeños.
- Post-procesamiento morfológico y filtrado por área por clase.
- Cross-validation y ensambles.
- Modelos Unet++ / FPN / DeepLabV3+.

---
