# 🧬 Co-MIL: Continual Multiple Instance Learning para Úlceras de Pie Diabético

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TorchMIL](https://img.shields.io/badge/TorchMIL-Standard-28a745?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)
![UNAM](https://img.shields.io/badge/UNAM-Facultad_de_Ciencias-C9A51F?style=for-the-badge)

**Proyecto de Práctica Profesional Supervisada (PPS)**  
**Autor:** Adrián Emiliano Beltrán Fernández  
**Institución:** Facultad de Ciencias, UNAM

---

## 📖 Descripción General

Este repositorio contiene la implementación oficial de **Co-MIL**, un framework de Aprendizaje Profundo diseñado para la **clasificación débilmente supervisada y localización espacial** de tejidos patológicos (Granulación, Fibrina, Callo, etc.) en imágenes clínicas de **Úlceras de Pie Diabético (UPD)**.

### 🎯 Problema Resuelto

El sistema aborda el **cuello de botella de la anotación densa a nivel de píxel** mediante el paradigma **MIML (Multi-Instance Multi-Label)**:

- **Entrada:** Anotación global de la úlcera por expertos clínicos (sin segmentación píxel a píxel)
- **Salida:** Mapas de Activación de Clase (CAMs) que localizan automáticamente cada tejido en la imagen

Esto reduce drasticamente el tiempo de anotación manteniendo la capacidad de localización espacial.

---

## ⚡ Características Arquitectónicas

### 🔹 Extracción Agnóstica a la Resolución
El sistema procesa imágenes clínicas en bruto (VGA a 12MP+) sin comprimir la imagen original, preservando gradientes biológicos a nivel celular.

### 🔹 Paradoja del Zero-Padding Resuelta
Implementa algoritmos de **Smart Crop** y **Reflection Padding** para expansión de ROI, eliminando bordes geométricos artificiales que podrían confundir a la red neuronal.

### 🔹 Atención MIML de K-Ramas
El mecanismo **Gated Attention** genera `K` mapas de calor independientes (uno por cada tejido anotado dinámicamente), habilitando la transición a Segmentación Semántica Débilmente Supervisada (WSSS).

### 🔹 Protección de Backbone en RAM
Interpola dinámicamente parches de alta densidad (ej. 56px) a la resolución operativa de **MobileNetV2** (224px), optimizando VRAM y viabilizando despliegue en dispositivos Edge.

### 🔹 Mitigación de Sesgo Tisular
Implementa cálculo dinámico de `pos_weight` para Entropía Cruzada Binaria, forzando convergencia en clases minoritarias (tejidos raros en las muestras clínicas).

---

## 📂 Estructura del Repositorio

```
CO-MIL/
├── generador_bolsas.py           # 🖱️  Herramienta interactiva de etiquetado
├── reprocesador_dataset.py        # ⚙️  Redimensionamiento masivo de bolsas
├── inspector_bolsas.py             # 🔍 Auditoría visual de tensores
├── validar_flujo_visual.py         # 📊 Validación de mapas de atención
├── dataset.py                      # 📦 Ingesta de datos (Dataset MIML)
├── entrenar_comil.py               # 🚀 Motor de optimización (Fase 1)
├── evaluar_comil.py                # 📈 Diagnóstico y métricas (Fase 2)
├── Models/
│   ├── attention_mil.py            # 🧠 Arquitectura Co-MIL completa
│   └── __init__.py
├── Data/                           # Carpeta para datos de entrada
└── requirements.txt                # Dependencias del proyecto
```

### Descripción Detallada de Módulos

| Archivo | Tipo | Descripción |
| :--- | :--- | :--- |
| `generador_bolsas.py` | 🖱️ UI Interactivo | Herramienta clínica con GUI para aislar ROI, etiquetar tejidos y serializar tensores de 224×224 px en archivos `.pt`. |
| `reprocesador_dataset.py` | ⚙️ Motor Headless | Redimensiona masivamente bolsas a resoluciones microscópicas (ej. 56×56 px) heredando coordenadas del experto. |
| `inspector_bolsas.py` | 🔍 Debugger Visual | Auditoría matemática que renderiza tensores, valida malla MIL y expone metadatos. |
| `validar_flujo_visual.py` | 📊 Validador WSSS | Genera los `K` mapas de atención independientes sobre topología real de úlcera. |
| `dataset.py` | 📦 Ingesta de Datos | Clase `CoMILDataset` adaptada a `torchmil.data.collate_fn` con protección geométrica. |
| `Models/attention_mil.py` | 🧠 Arquitectura | Implementación POO: *InstanceEncoder*, *AttentionAggregator*, *BagClassifier*. |
| `entrenar_comil.py` | 🚀 Motor de Opt. | Script maestro de entrenamiento con autodetección de clases y pesos compensatorios. |
| `evaluar_comil.py` | 📈 Diagnóstico | Calcula Hamming Loss, matrices de confusión MIML y métricas clínicas. |

---

## ⚙️ Instalación y Requisitos

### Requisitos del Sistema

- **Python:** 3.9 o superior
- **CUDA:** Opcional pero recomendado para GPU (NVIDIA)
- **Memoria RAM:** Mínimo 8GB (16GB recomendado para procesamiento de imágenes de alta resolución)

### Instalación Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/adrianBeltrn/co_mil_pps.git
cd co_mil_pps

# 2. Crear entorno virtual
python -m venv env
# En Windows:
env\Scripts\activate
# En macOS/Linux:
source env/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r CO-MIL/requirements.txt
```

### Instalación con Anaconda

```bash
conda create -n comil python=3.9
conda activate comil
cd CO-MIL
pip install -r requirements.txt
```

### Verificar Instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchmil; print('TorchMIL instalado correctamente')"
```

---

## 🚀 Guía Rápida de Uso

### Flujo Típico (5 pasos)

```mermaid
graph LR
    A[Imágenes Crudas] -->|generador_bolsas.py| B[Bolsas Etiquetadas 224px]
    B -->|reprocesador_dataset.py| C[Bolsas Multiresolución 56px]
    C -->|inspector_bolsas.py| D[✓ Auditoría]
    D -->|entrenar_comil.py| E[Modelo Entrenado]
    E -->|validar_flujo_visual.py| F[Mapas de Atención]
    E -->|evaluar_comil.py| G[Métricas MIML]
```

---

## 📋 Flujo de Trabajo Detallado

### **Fase A: Preparación de Datos (Clínica)**

#### 1️⃣ Anotación Débil Interactiva
```bash
python CO-MIL/generador_bolsas.py
```

**Qué hace:**
- Abre interfaz gráfica para cargar imágenes crudas
- Permite trazar el contorno de la úlcera (ROI)
- Permite marcar los tejidos presentes: Granulación, Fibrina, Callo, etc.
- Genera tensor `.pt` con estructura:
  ```python
  {
    'X': torch.Tensor,         # [N_parches, 3, 224, 224]
    'Y': torch.Tensor,         # [num_clases] vector multietiqueta binario
    'grid_shape': tuple,       # (rows, cols) para reconstrucción espacial
    'metadata': {...}          # timestamp, nombre original, etc.
  }
  ```

#### 2️⃣ (Opcional) Densificación Multi-resolución WSSS
```bash
python CO-MIL/reprocesador_dataset.py
```

**Qué hace:**
- Redimensiona bolsas a resoluciones microscópicas (56×56, 112×112 px)
- Pruebas de análisis más fino de patologías
- Crea subcarpetas `56px/`, `112px/` en `Bolsas_MIL_Procesadas/`

#### 3️⃣ Auditoría y Validación
```bash
python CO-MIL/inspector_bolsas.py
```

**Qué hace:**
- Renderiza cada tensor `.pt` en visualización
- Valida que la malla MIL sea correcta
- Detecta anomalías: Zero-Padding, dimensiones inconsistentes
- Exporta reportes de integridad

---

### **Fase B: Optimización y Evaluación (Computacional)**

#### 4️⃣ Entrenamiento del Modelo
```bash
python CO-MIL/entrenar_comil.py
```

**Qué hace:**
- Autodetecta cantidad de clases (dinámico según datos)
- Calcula pesos compensatorios (`pos_weight`) para clases desbalanceadas
- Congela backbone de MobileNetV2 (protege pesos ImageNet)
- Optimiza cabezal MIML con BCE ponderada
- Guarda mejor modelo en `Pesos_Entrenados/comil_miml_fase1.pth`

**Configuración disponible:**
- `BATCH_SIZE`: Tamaño de lote (default: 4)
- `LEARNING_RATE`: Tasa de aprendizaje (default: 1e-4)
- `NUM_EPOCHS`: Épocas de entrenamiento (default: 50)
- `DISPOSITIVO`: "cuda" o "cpu" (autodetectado)

#### 5️⃣ Validación de Mapas WSSS
```bash
python CO-MIL/validar_flujo_visual.py
```

**Qué hace:**
- Genera mapas de atención K-independientes
- Renderiza heatmaps sobrepuestos a imagen original
- Valida que la red enfoque en regiones clínicamente relevantes
- Exporta PNG para inspección visual

#### 6️⃣ Evaluación Cuantitativa
```bash
python CO-MIL/evaluar_comil.py
```

**Qué hace:**
- Calcula **Hamming Loss** (error promedio por clase)
- Calcula **Subset Accuracy** (exactitud exacta multietiqueta)
- Genera **Matrices de Confusión** por cada tejido
- Exporta reportes PDF con métricas clínicas

---

## 📐 Formalización Matemática

### Mecanismo de Atención (Gated Attention)

El sistema evalúa la relevancia clínica de cada parche mediante un sistema de compuertas probabilísticas:

$$\alpha_{i}^{k} = \frac{\exp\left\{w_k^{T}(\tanh(Vh_{i}^{T}) \odot \sigma(Uh_{i}^{T}))\right\}}{\sum_{j=1}^{N}\exp\left\{w_k^{T}(\tanh(Vh_{j}^{T}) \odot \sigma(Uh_{j}^{T}))\right\}}$$

**Donde:**
- $\alpha_{i}^{k}$: Peso de atención del parche $i$ para la clase $k$ (tejido)
- $h_i$: Vector de características del parche $i$ (1280-D desde MobileNetV2)
- $V, U \in \mathbb{R}^{D \times L}$: Matrices de transformación lineal
- $\odot$: Producto elemento a elemento (Hadamard)
- $\sigma(\cdot)$: Función sigmoide
- $\tanh(\cdot)$: Tangente hiperbólica

---

### Función de Pérdida Multi-Etiqueta Ponderada

La optimización de las $K$ ramas independientes se realiza usando Entropía Cruzada Binaria ponderada por clase:

$$\mathcal{L} = -\sum_{c=1}^{K} w_{c} \left[ y_{c} \log(\sigma(z_{c})) + (1 - y_{c}) \log(1 - \sigma(z_{c})) \right]$$

**Donde:**
- $w_c$: Peso compensatorio para la clase $c$ (inversamente proporcional a frecuencia de positivos)
- $y_c \in \{0,1\}$: Etiqueta binaria del tejido $c$ en la bolsa
- $z_c$: Logit predicho para la clase $c$ (salida del clasificador)
- $K$: Número total de clases (tejidos) en el dataset

**Cálculo dinámico de pesos:**
$$w_c = \frac{\text{Total de muestras}}{\text{Muestras positivas de clase } c} \cdot \text{factor de escala}$$

---

### Reconstrucción de Mapas de Activación (CAM)

Para visualización clínica, los pesos de atención se proyectan a la resolución espacial original:

$$\text{CAM}_{k}(x, y) = \text{Resize}\left(\sum_{i=1}^{N} \alpha_{i}^{k} \cdot M_i(x,y), \text{original\_resolution}\right)$$

Donde $M_i(x,y)$ es la máscara spatial que indica la región del parche $i$ en la imagen original.

---

## 🔬 Casos de Uso y Ejemplos

### Ejemplo 1: Entrenamiento Rápido en Google Colab

Para evitar limitaciones de GPU local, el proyecto está optimizado para Google Colab:

```python
# En Google Colab (primera celda)
from google.colab import drive
drive.mount('/content/drive')
!pip install torchmil

# Segunda celda
%cd "/content/drive/MyDrive/Co_MIL_PPS"
!python CO-MIL/entrenar_comil.py
```

Los pesos se guardan automáticamente en Google Drive.

### Ejemplo 2: Evaluación en Nuevo Conjunto de Datos

```python
# dataset.py se puede reutilizar
from CO_MIL.dataset import CoMILDataset
from torch.utils.data import DataLoader

dataset = CoMILDataset(pt_folder="ruta/a/nuevas/bolsas", target_size=224)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Iterar sobre bolsas MIML
for bag_X, bag_Y in dataloader:
    print(f"Bolsa: {bag_X.shape}, Etiquetas: {bag_Y.shape}")
```

---

## 🔧 Solución de Problemas

### ❌ "ModuleNotFoundError: No module named 'torchmil'"
```bash
pip install --upgrade torchmil
```

### ❌ "CUDA out of memory"
Reducir tamaño de batch en `entrenar_comil.py`:
```python
BATCH_SIZE = 2  # Reducir de 4 a 2
```

### ❌ "No se encuentran archivos .pt en la carpeta"
Verificar que `generador_bolsas.py` se ejecutó correctamente:
```bash
python CO-MIL/inspector_bolsas.py  # Auditar bolsas existentes
```

### ❌ Los mapas de atención no convergen a regiones clínicas
Verificar:
1. Anotaciones débiles correctas en `generador_bolsas.py`
2. `pos_weight` se calcula automáticamente (revisar terminal durante entrenamiento)
3. Aumentar `NUM_EPOCHS` o reducir `LEARNING_RATE`

---

## 📊 Estructura de Salidas

### Estructura de Carpetas Generadas

```
Pesos_Entrenados/
└── comil_miml_fase1.pth           # Modelo entrenado

Bolsas_MIL_Procesadas/
├── 224px/
│   ├── imagen1_bag.pt
│   └── imagen2_bag.pt
└── 56px/                           # Opcional (multiresolución)
    └── ...

Resultados/
├── heatmaps_validacion/           # Mapas CAM generados
└── metricas_evaluacion.json        # Reporte cuantitativo
```

### Formato de Archivo `.pt`

Cada archivo `.pt` es un diccionario Python serializado con:
```python
{
    'X': torch.Tensor              # Shape: [N, 3, 224, 224] (N = parches en bolsa)
    'Y': torch.Tensor              # Shape: [num_clases] (valores 0 o 1)
    'grid_shape': (rows, cols),    # Para reconstrucción espacial
    'metadata': {
        'timestamp': str,
        'imagen_original': str,
        'roi_coords': [(x1, y1, x2, y2)]
    }
}
```

---

## 🤝 Flujo de Colaboración Clínica-Computacional

Este proyecto implementa un ciclo iterativo:

1. **Clínico (Semana 1):** Genera 50 bolsas etiquetadas con `generador_bolsas.py`
2. **Ingeniero (Semana 2):** Entrena modelo con `entrenar_comil.py` en GPU
3. **Validación (Semana 2):** Ejecuta `validar_flujo_visual.py` para inspecionar localizaciones
4. **Feedback (Semana 3):** Clínico revisa heatmaps y refina anotaciones
5. **Iteración:** Volver a paso 1 con datos mejorados

---

## 📚 Referencias Académicas

Este trabajo se basa en los siguientes paradigmas:

- **Multiple Instance Learning (MIL):** Dietterich, T. G., et al. (1997). "Solving the Multiple Instance Problem with Axis-Parallel Rectangles"
- **Multi-Label Learning:** Sorower, M. S. (2010). "A Literature Survey on Algorithms for Multi-Label Learning"
- **Weakly Supervised Segmentation (WSSS):** Hong, S., et al. (2015). "Learning Deep Features for Discriminative Localization"
- **Medical Image Analysis:** Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"

---

## 🚀 Trabajo a Futuro

Este proyecto se encuentra en **desarrollo activo**. Las próximas fases incluyen:

- ✅ **Integración de Búfer de Repetición** para Aprendizaje Continuo Real (Co-MIL)
- ✅ **Refinamiento Espacial** de mapas mediante DenseCRF para cuantificación nanométrica en cm²
- ✅ **Modelo Clínico Embarcado** (ONNX) para tablets y dispositivos móviles
- ✅ **Arquitecturas Alternativas** (Vision Transformers, YOLO-MIL hybrid)
- ✅ **Base de Datos Multicéntrica** con datos de clínicas colaboradoras

---

## 📄 Licencia

Este proyecto está bajo licencia **MIT**. Ver archivo `LICENSE` para detalles.

---

## 📧 Contacto y Contribuciones

**Autor:** Adrián Emiliano Beltrán Fernández  
**Institución:** Facultad de Ciencias, UNAM  
**Correo:** [tu-email@unam.mx]

Para reportar bugs o sugerir mejoras, por favor abre un **Issue** o **Pull Request** en el repositorio.

---

**Actualizado:** Abril 2026  
**Estado del Proyecto:** 🟡 En desarrollo (Fase 1 y 2 completadas)
