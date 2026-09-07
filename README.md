# 🧬 Co-MIL — Análisis de tejidos en úlceras de pie diabético

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![segmentation--models--pytorch](https://img.shields.io/badge/smp-0.5-28a745?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)
![UNAM](https://img.shields.io/badge/UNAM-Facultad_de_Ciencias-C9A51F?style=for-the-badge)

**Autor:** Adrián Emiliano Beltrán Fernández · Estudiante de Física Biomédica, Facultad de Ciencias, UNAM
**Tutor:** Dr. José Eduardo Chairez Veloz · Facultad de Ciencias, UNAM
**Marco:** proyecto derivado de PAPIIT IN115825 — *Innovaciones tecnológicas en la evaluación
oportuna y el manejo de lesiones en pie diabético*

> **Estado:** 🟡 en desarrollo · última actualización **7 de septiembre de 2026**
> El proyecto cambió de estrategia en la reunión del 27 de agosto de 2026 (de anotación débil por
> cajas a **anotación densa** por contorno de tejido). El registro completo está en la
> [bitácora](#-documentos-clave).

---

## 📖 Objetivo

Desarrollar y evaluar modelos para la **clasificación y localización de tejidos** (granulación,
fibrina, callo, necrótico, piel perilesional) en fotografías clínicas de úlceras de pie diabético,
con tres énfasis:

1. **Eficiencia de anotación** — cuántas máscaras densas hacen falta realmente, frente a etiquetas
   débiles baratas.
2. **Aprendizaje continuo** — integrar lotes de datos que llegan en el tiempo sin olvido catastrófico
   (el componente que da nombre al proyecto, *Co*-MIL).
3. **Viabilidad embebida** — modelos ligeros (MobileNetV2) para dispositivos de bajo consumo.

El proyecto se sostiene sobre un núcleo computacional (arquitecturas, funciones de pérdida,
evaluación) acorde con un perfil de física biomédica, y sobre un conjunto de datos regional mexicano
en construcción.

---

## 🧭 Planteamiento (reformulado, sep-2026)

De una **anotación densa** (contorno de cada tejido) se derivan de forma automática todas las
versiones más simples de etiqueta. Tres brazos de entrenamiento las consumen, y los tres se evalúan
de forma idéntica contra las máscaras:

| Brazo | Anota el humano | Rol |
| :--- | :--- | :--- |
| **Supervisado** | contorno de cada tejido en cada foto | cota alta; segmentación semántica ligera (FPN + MobileNetV2) |
| **Débil (MIL)** | solo *qué tejidos hay* en la foto | anotación barata; mecanismo de atención / CAM |
| **Mixto** | pocas máscaras densas + muchas etiquetas | la pregunta central: la curva Dice vs. nº de máscaras |

Encima, el **aprendizaje continuo** re-entrena el modelo conforme llegan lotes; al final se exporta a
ONNX y se mide la latencia. Diagrama completo: `documentación/figuras/fig_proyecto_reformulado.png`.

---

## 📊 Resultados sobre DFUTissue (dataset público)

Mientras se anota el conjunto mexicano, el modelado se adelanta sobre **DFUTissue** (110 imágenes,
3 tejidos, partición oficial).

**Brazo supervisado** — segmentación semántica, Dice medio en tejidos (partición oficial):

| Modelo | Parámetros | Dice medio | Fibrina / Granul. / Callo |
| :--- | ---: | ---: | :--- |
| Baseline previo del grupo (Maldonado-Oclica *et al.* 2025) | — | 0.545 | 0.333 / 0.786 / 0.515 |
| FPN + MobileNetV2, receta reforzada (6 semillas) | 4.2 M | **0.710** | 0.58 / 0.87 / 0.68 |
| SegFormer-B0, misma receta (1 semilla) | 3.7 M | **0.744** | 0.61 / 0.90 / 0.72 |
| U-Net++ · ResNet-34 (referencia pesada) | 26 M | 0.719 | — |

Le gana al trabajo previo en las tres clases; el modelo pequeño pierde solo 0.02–0.04 de Dice
frente a modelos hasta 6× mayores. *(Los conjuntos de prueba no son idénticos; la comparación es indicativa.)*

**Frontera de eficiencia de anotación (5 semillas):** con ~20 máscaras densas (26 % de las imágenes)
se recupera ~80 % del desempeño de la supervisión densa completa. **Añadir etiquetas de imagen
baratas sobre las máscaras densas no ayuda a esta escala** (y a N alto estorba); filtrarlas tampoco.
Curva: `documentación/figuras/fig_curva_eficiencia_dfutissue.png`.

**Aprendizaje continuo (prototipo, `CO-MIL/continual/`):** entre lotes con distribución parecida el
olvido es leve (extractor congelado); entre poblaciones **de verdad distintas** (DFUTissue → lote
mexicano) el entrenamiento secuencial ingenuo colapsa el rendimiento previo (F1 0.74 → 0.02) y la
**repetición latente** lo evita (0.76) a coste de almacenamiento ~0. Ni la aumentación de cámara/color
ni la normalización de color cierran el salto: adaptar a una población nueva necesita anotación real.
Figura: `documentación/figuras/fig_continual_salto.png`.

---

## 📂 Estructura del repositorio

```
CO-MIL/
├── segmentacion/                  # ← brazo supervisado + mixto (post-pivote)
│   ├── descargar_datos.py         #   descarga reproducible de DFUTissue / WoundTissue
│   ├── dataset_seg.py             #   ingesta de DFUTissue (imagen + máscara)
│   ├── entrenar_seg.py            #   brazo supervisado (FPN / Unet++ / SegFormer; Dice / Tversky / Focal)
│   ├── entrenar_eficiencia.py     #   brazos débil y mixto; curva de eficiencia; --max_tejidos_debil
│   └── colab_estudio_dfutissue.ipynb   #   los tres experimentos en GPU (Colab)
│
├── continual/                     # ← objetivo 3: aprendizaje continuo (prototipo)
│   ├── datos_continual.py         #   caché de características + segundo dominio (lote mexicano)
│   ├── estrategias.py             #   cabeza MIL + naive / joint / experience / latent replay
│   ├── correr_continual.py        #   bucle domain- y class-incremental + métricas de olvido
│   ├── transferencia.py           #   el salto DFUTissue → lote mexicano + continual
│   └── transferencia_filtro.py    #   ¿el salto es cámara/color o población?
│
├── attention_mil.py  · Models/    # pipeline MIL original (brazo débil); se conserva
├── dataset.py · entrenar_comil.py · evaluar_comil.py · ...
├── catalogo_tejidos.py            # catálogo canónico y resolución de etiquetas
├── generador_bolsas.py            # herramienta de anotación por cajas (en desuso tras el pivote)
└── refinamiento_crf.py            # refinamiento DenseCRF de mapas de atención

documentación/
├── co_mil_bitacora.pdf · _parte2.pdf · _parte3.pdf   # bitácora I–III
├── co_mil_bitacora_parte4.tex/.pdf                   # bitácora IV (actual)
├── propuesta_dataset_tejidos_UPD.pdf                 # propuesta de anotación para el equipo clínico
├── Plan_PPS.pdf · Manual_usuario_COMIL.pdf
└── figuras/                                          # todas las figuras, vectoriales

Pesos_Entrenados/                  # metadatos de cada experimento (los .pth no se versionan)
```

---

## 📄 Documentos clave

| Documento | Para qué |
| :--- | :--- |
| [`documentación/co_mil_bitacora_parte4.pdf`](documentación/co_mil_bitacora_parte4.tex) | **Empezar aquí.** Registro del pivote, la investigación de estado del arte, el planteamiento reformulado y los resultados T1–T4. |
| [`documentación/propuesta_dataset_tejidos_UPD.pdf`](documentación/propuesta_dataset_tejidos_UPD.tex) | Propuesta de las 5 clases de tejido y el protocolo de anotación densa, para el equipo médico y la FENO. |
| `documentación/co_mil_bitacora.pdf` · `_parte2.pdf` · `_parte3.pdf` | Registro previo (marzo–agosto de 2026): marco teórico, experimentación MIL, auditoría y primer PoC válido. |

---

## ⚙️ Reproducir los experimentos

**En GPU (recomendado):** abrir `CO-MIL/segmentacion/colab_estudio_dfutissue.ipynb` en Google Colab
con GPU T4 y ejecutar las celdas. El notebook clona este repositorio, descarga DFUTissue y guarda los
resultados en Google Drive.

**Localmente:**

```bash
python -m pip install -r CO-MIL/requirements.txt segmentation-models-pytorch
python CO-MIL/segmentacion/descargar_datos.py          # baja DFUTissue a datasets_publicos/
python CO-MIL/segmentacion/entrenar_seg.py             # brazo supervisado (baseline)
python CO-MIL/segmentacion/entrenar_eficiencia.py \
    --modos mixto,solo_supervisado --semillas 42,1,7   # curva de eficiencia + ablación
python CO-MIL/continual/correr_continual.py --modo clase   # prototipo continuo (class-incremental)
python CO-MIL/continual/transferencia.py               # el salto DFUTissue → lote mexicano
```

---

## 🗓️ Próximos pasos

- Sesión con la Facultad de Enfermería y Obstetricia (FENO) de la UNAM para fijar el catálogo de
  5 clases y los criterios de anotación densa (sin fecha confirmada).
- Anotación densa del conjunto mexicano (~60–270 imágenes).
- Repetir el test de transferencia entre poblaciones con **máscaras densas** (muestra de WoundTissue).
- Preentrenamiento auto-supervisado del extractor sobre fotografías de herida sin etiquetar.
- Factibilidad embebida: exportación a ONNX y medición de latencia.

---

## 📚 Trabajo previo del grupo

Maldonado-Oclica A., Rios-López R., **Beltrán-Fernández A.**, *et al.* (2025).
*AI-based Mobile App for Segmentation and Tissue Classification on Diabetic Foot Ulcer: A Step
Forward in Patient Care.* Springer. DOI: 10.1007/978-3-031-95841-0_47.

---

## 📧 Contacto

**Adrián Emiliano Beltrán Fernández** — adrian_beltran@ciencias.unam.mx
Facultad de Ciencias, UNAM.

## 📄 Licencia

Código bajo licencia **MIT** (ver [`LICENSE`](LICENSE)). Los conjuntos de datos públicos que se
descargan a `datasets_publicos/` conservan su propia licencia y no se redistribuyen en este
repositorio.
