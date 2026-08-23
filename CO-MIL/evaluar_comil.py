"""
=========================================================================================
FASE 2: EVALUACIÓN MIML Y GENERACIÓN DE EVIDENCIA CLÍNICA
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (MÓDULO DE EVALUACIÓN).
CÓMO EJECUTAR: python CO-MIL/evaluar_comil.py

Objetivo:
1. Evaluar el modelo SOBRE EL SPLIT DE PRUEBA (nunca sobre los datos de entrenamiento).
2. Calcular Hamming Loss, Exactitud de Subconjunto, y por cada tejido: precisión,
   recall/sensibilidad, especificidad y AUC-ROC (cuando el split de evaluación tiene al
   menos un caso positivo y uno negativo de esa clase; si no, se reporta como N/D).
3. Generar Matrices de Confusión por cada tejido individual.
4. Producir el Heatmap WSSS final con los pesos de atención ya optimizados.

Requiere haber corrido antes:
    python CO-MIL/particionar_dataset.py --bolsas "<ruta a 224px>"
=========================================================================================
"""

import math
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    hamming_loss,
    multilabel_confusion_matrix,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

_DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(_DIRECTORIO_ACTUAL)

from dataset import CoMILDataset
from Models.attention_mil import CoMILNetwork
from torchmil.data import collate_fn


def calcular_sensibilidad_especificidad(y_true_col: np.ndarray, y_pred_col: np.ndarray):
    """A partir de una sola columna (una clase) de etiquetas binarias reales y
    predichas, calcula sensibilidad (recall de la clase positiva) y
    especificidad (recall de la clase negativa)."""
    tp = int(np.sum((y_true_col == 1) & (y_pred_col == 1)))
    fn = int(np.sum((y_true_col == 1) & (y_pred_col == 0)))
    tn = int(np.sum((y_true_col == 0) & (y_pred_col == 0)))
    fp = int(np.sum((y_true_col == 0) & (y_pred_col == 1)))

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sensibilidad, especificidad


def evaluar_modelo_miml(split: str = "test"):
    # --- CONFIGURACIÓN ---
    RAIZ_REPO = os.path.dirname(_DIRECTORIO_ACTUAL)
    RUTA_PESOS = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "comil_miml_fase1.pth")
    UMBRAL = 0.5  # Sensibilidad del diagnóstico clínico

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. CARGA DEL DICCIONARIO DE ENTRENAMIENTO
    if not os.path.exists(RUTA_PESOS):
        print(f"[!] Error: No se encontraron los pesos en {RUTA_PESOS}. Entrena primero con entrenar_comil.py.")
        return

    checkpoint = torch.load(RUTA_PESOS, map_location=dispositivo)
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    ruta_bolsas = checkpoint.get("ruta_bolsas")
    ruta_manifest = checkpoint.get("ruta_manifest")

    if not ruta_bolsas or not ruta_manifest:
        print("[!] Este checkpoint fue entrenado con una versión anterior del script (sin rutas "
              "de dataset/manifiesto guardadas). Vuelve a entrenar con entrenar_comil.py para "
              "poder evaluar sobre un split real.")
        return

    # 2. INSTANCIACIÓN DE LA ARQUITECTURA
    modelo = CoMILNetwork(num_classes=num_classes).to(dispositivo)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    # 3. CARGA DE DATOS — SOLO el split indicado (test por defecto), nunca el de
    # entrenamiento. Sin aumento de datos: se quiere medir desempeño real.
    try:
        dataset = CoMILDataset(
            pt_folder=ruta_bolsas,
            target_size=224,
            manifest_path=ruta_manifest,
            split=split,
            augment=False,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[!] Error: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    y_true = []
    y_pred_logits = []

    print(f"\n[+] Evaluando sobre el split '{split}': {len(dataset)} bolsas (no usadas para entrenar).")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_Y = batch["Y"]
            batch = batch.to(dispositivo)
            batch_X, mask = batch["X"], batch["mask"].bool()
            logits, _ = modelo(batch_X, mask)

            y_true.append(batch_Y.numpy())
            y_pred_logits.append(torch.sigmoid(logits).cpu().numpy())

    # Concatenamos resultados
    y_true = np.vstack(y_true)
    y_pred_probs = np.vstack(y_pred_logits)
    y_pred_bin = (y_pred_probs > UMBRAL).astype(int)

    # --- CÁLCULO DE MÉTRICAS MIML ---
    hl = hamming_loss(y_true, y_pred_bin)
    print(f"\n=== REPORTE DE EVALUACIÓN FASE 2 (split='{split}', {len(dataset)} bolsas) ===")
    print(f"-> Hamming Loss: {hl:.4f} (Menor es mejor)")
    print(f"-> Subset Accuracy (Exactitud Total): {np.all(y_true == y_pred_bin, axis=1).mean():.4f}")

    print("\n--- Desempeño por Tejido (precision/recall/F1, sklearn) ---")
    print(classification_report(y_true, y_pred_bin, target_names=class_names, zero_division=0))

    print("\n--- Sensibilidad, Especificidad y AUC-ROC por Tejido ---")
    print("(N/D = no definido: el split de evaluación no tiene ejemplos positivos Y negativos de esa clase)")
    for i, nombre in enumerate(class_names):
        col_true = y_true[:, i]
        col_pred = y_pred_bin[:, i]
        col_prob = y_pred_probs[:, i]

        sens, esp = calcular_sensibilidad_especificidad(col_true, col_pred)
        n_pos = int(col_true.sum())
        n_neg = int(len(col_true) - n_pos)

        if n_pos > 0 and n_neg > 0:
            auc = roc_auc_score(col_true, col_prob)
            auc_str = f"{auc:.3f}"
        else:
            auc_str = "N/D"

        sens_str = f"{sens:.3f}" if not np.isnan(sens) else "N/D"
        esp_str = f"{esp:.3f}" if not np.isnan(esp) else "N/D"

        print(f"  {nombre:35s} | positivos={n_pos:2d}/{len(col_true):2d} | "
              f"sensibilidad={sens_str:>5s} | especificidad={esp_str:>5s} | AUC-ROC={auc_str:>5s}")

    # --- VISUALIZACIÓN 1: MATRICES DE CONFUSIÓN MULTIETIQUETA ---
    mcm = multilabel_confusion_matrix(y_true, y_pred_bin)
    plt.style.use('default')

    # Cuadricula (no una sola fila) para que siga siendo legible al insertarla en un
    # documento a ancho de página -- con 12 clases, una fila de 12 paneles queda
    # ilegible en miniatura.
    cols = min(4, num_classes)
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows))
    axes_list = [axes] if num_classes == 1 else np.array(axes).flatten()

    for i, (matrix, name) in enumerate(zip(mcm, class_names)):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes_list[i], cbar=False, annot_kws={"size": 11})
        axes_list[i].set_title(name, fontsize=10)
        axes_list[i].set_xlabel("Predicción", fontsize=9)
        axes_list[i].set_ylabel("Realidad", fontsize=9)

    for i in range(num_classes, len(axes_list)):
        axes_list[i].axis("off")

    fig.suptitle(f"Matrices de confusión por tejido — split '{split}' ({len(dataset)} bolsas)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    ruta_figura = os.path.join(RAIZ_REPO, "Pesos_Entrenados", f"matrices_confusion_{split}.png")
    plt.savefig(ruta_figura, dpi=150)
    print(f"\n[+] Matrices de confusión guardadas en: {ruta_figura}")

    plt.show()


if __name__ == "__main__":
    split_a_evaluar = sys.argv[1] if len(sys.argv) > 1 else "test"
    evaluar_modelo_miml(split=split_a_evaluar)
