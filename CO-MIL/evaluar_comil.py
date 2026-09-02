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

import glob
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
from dataset_cam import CoMILDatasetCAM
from Models.attention_mil import CoMILNetwork
from Models.attention_mil_cam import CoMILNetworkCAM
from torchmil.data import collate_fn
import catalogo_tejidos
import experimentos


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


def evaluar_modelo_miml(split: str = "test", ruta_experimento: str = None):
    # --- CONFIGURACIÓN ---
    RAIZ_REPO = os.path.dirname(_DIRECTORIO_ACTUAL)
    RUTA_PESOS_DIR = os.path.join(RAIZ_REPO, "Pesos_Entrenados")
    UMBRAL = 0.5  # Sensibilidad del diagnóstico clínico

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Por default se evalúa el experimento más reciente; se puede apuntar a uno
    # viejo explícitamente (--experimento exp_20260822_172100) para reproducir
    # resultados de una corrida anterior sin mezclarlo con la más nueva.
    if ruta_experimento is None:
        ruta_experimento = experimentos.experimento_mas_reciente(RUTA_PESOS_DIR)
    elif not os.path.isabs(ruta_experimento):
        ruta_experimento = os.path.join(RUTA_PESOS_DIR, ruta_experimento)

    if not ruta_experimento or not os.path.isdir(ruta_experimento):
        print(f"[!] No se encontró ningún experimento en {RUTA_PESOS_DIR}. Entrena primero con entrenar_comil.py.")
        return

    RUTA_PESOS = os.path.join(ruta_experimento, "modelo.pth")
    if not os.path.exists(RUTA_PESOS):
        print(f"[!] Error: no se encontraron pesos en {RUTA_PESOS}.")
        return

    meta_experimento = experimentos.cargar_metadata(ruta_experimento)
    info_patch = meta_experimento.get("patch_size", {})
    print(f"[+] Experimento: {os.path.basename(ruta_experimento)}")
    if info_patch:
        print(f"    -> Resolución de parche: {info_patch.get('patch_sizes_encontrados')} px "
              f"({'MEZCLA DE RESOLUCIONES -- revisar' if info_patch.get('mezcla_de_resoluciones') else 'consistente'}), "
              f"{info_patch.get('parches_por_bolsa_min')}-{info_patch.get('parches_por_bolsa_max')} parches por bolsa")

    # 1. CARGA DEL DICCIONARIO DE ENTRENAMIENTO
    checkpoint = torch.load(RUTA_PESOS, map_location=dispositivo)
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    ruta_bolsas = checkpoint.get("ruta_bolsas")
    ruta_manifest = checkpoint.get("ruta_manifest")
    # Un checkpoint de entrenar_comil_cam.py guarda "arquitectura": "cam" -- con eso
    # basta para que este mismo script sirva para las dos arquitecturas sin
    # necesitar una bandera manual ni un script aparte que se pueda desincronizar.
    es_cam = checkpoint.get("arquitectura") == "cam"

    if not ruta_bolsas or not ruta_manifest:
        print("[!] Este checkpoint fue entrenado con una versión anterior del script (sin rutas "
              "de dataset/manifiesto guardadas). Vuelve a entrenar con entrenar_comil.py para "
              "poder evaluar sobre un split real.")
        return

    if es_cam:
        print("    -> Arquitectura: estilo CAM (1 forward pass por ROI completo; instancias = "
              "celdas del mapa de features nativo, no parches recortados a mano)")

    # 2. INSTANCIACIÓN DE LA ARQUITECTURA
    if es_cam:
        modelo = CoMILNetworkCAM(num_classes=num_classes).to(dispositivo)
    else:
        modelo = CoMILNetwork(num_classes=num_classes).to(dispositivo)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    # 3. CARGA DE DATOS — SOLO el split indicado (test por defecto), nunca el de
    # entrenamiento. Sin aumento de datos: se quiere medir desempeño real.
    try:
        if es_cam:
            dataset = CoMILDatasetCAM(
                pt_folder=ruta_bolsas,
                manifest_path=ruta_manifest,
                split=split,
            )
        else:
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

    # Auditoría de etiquetas: si el catálogo se vuelve a fragmentar (una
    # etiqueta cruda que ya no matchea nada del catálogo vigente), dataset.py
    # la ignora en silencio y una clase puede parecer "sin ejemplos" sin
    # serlo -- exactamente el bug de antes de la Etapa 0. Se avisa aquí en
    # vez de dejarlo para que alguien lo note manualmente en los resultados.
    problemas_etiquetas = catalogo_tejidos.auditar_etiquetas_no_reconocidas(
        ruta_bolsas, dataset.class_catalog, dataset.renombres
    )
    if problemas_etiquetas:
        print(f"\n[!] ALERTA: {len(problemas_etiquetas)} bolsa(s) con etiquetas que no matchean "
              "el catálogo vigente (se están ignorando en silencio, pueden estar deflactando "
              "el conteo de alguna clase):")
        for archivo, etiquetas in list(problemas_etiquetas.items())[:10]:
            print(f"    {archivo}: {etiquetas}")
        if len(problemas_etiquetas) > 10:
            print(f"    ... y {len(problemas_etiquetas) - 10} más.")

    # Conteo de positivos por clase en TODO el dataset (no solo este split),
    # para poder distinguir "esta clase es rara de verdad" de "este split en
    # particular tuvo mala suerte" al leer el reporte de abajo.
    conteo_global = catalogo_tejidos.contar_positivos_por_clase(
        ruta_bolsas, dataset.class_catalog, dataset.renombres
    )
    total_bolsas_dataset = len(glob.glob(os.path.join(ruta_bolsas, "*.pt")))

    # El estilo CAM entrena y evalúa una bolsa a la vez (batch_size=1, sin
    # collate_fn) porque cada ROI reconstruido tiene un tamaño físico distinto y no
    # se puede empaquetar en un lote parejo como los parches de tamaño uniforme.
    if es_cam:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    y_true = []
    y_pred_logits = []

    print(f"\n[+] Evaluando sobre el split '{split}': {len(dataset)} bolsas (no usadas para entrenar).")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if es_cam:
                imagen = batch["imagen"][0].to(dispositivo)
                batch_Y = batch["Y"]
                logits, _, _ = modelo(imagen)
            else:
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
    print("(dataset=positivos en TODO el dataset, no solo este split -- distingue 'clase rara de verdad' de "
          "'mala suerte en el split'; ver catalogo_tejidos.contar_positivos_por_clase)")
    for i, nombre in enumerate(class_names):
        col_true = y_true[:, i]
        col_pred = y_pred_bin[:, i]
        col_prob = y_pred_probs[:, i]

        sens, esp = calcular_sensibilidad_especificidad(col_true, col_pred)
        n_pos = int(col_true.sum())
        n_neg = int(len(col_true) - n_pos)
        n_pos_dataset = conteo_global.get(nombre)

        if n_pos > 0 and n_neg > 0:
            auc = roc_auc_score(col_true, col_prob)
            auc_str = f"{auc:.3f}"
        else:
            auc_str = "N/D"

        sens_str = f"{sens:.3f}" if not np.isnan(sens) else "N/D"
        esp_str = f"{esp:.3f}" if not np.isnan(esp) else "N/D"
        dataset_str = f"{n_pos_dataset:3d}/{total_bolsas_dataset}" if n_pos_dataset is not None else "N/D"

        print(f"  {nombre:35s} | positivos={n_pos:2d}/{len(col_true):2d} (dataset={dataset_str}) | "
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

    if es_cam:
        subtitulo_patch = "estilo CAM (instancias = mapa de features nativo, no parches recortados)"
    else:
        patch_sizes = info_patch.get("patch_sizes_encontrados") if info_patch else None
        subtitulo_patch = f"parche {patch_sizes} px" if patch_sizes else "resolución de parche desconocida"
    fig.suptitle(
        f"Matrices de confusión por tejido — split '{split}' ({len(dataset)} bolsas)\n"
        f"{os.path.basename(ruta_experimento)} · {subtitulo_patch}",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Se guarda DENTRO de la carpeta del experimento, no en Pesos_Entrenados/ a
    # secas -- así nunca se mezcla con la figura de otra corrida.
    ruta_figura = os.path.join(ruta_experimento, f"matrices_confusion_{split}.png")
    plt.savefig(ruta_figura, dpi=150)
    print(f"\n[+] Matrices de confusión guardadas en: {ruta_figura}")

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluacion Co-MIL sobre un split de datos")
    parser.add_argument("split", nargs="?", default="test", choices=["train", "val", "test"])
    parser.add_argument("--experimento", default=None,
                         help="Nombre o ruta de la carpeta de experimento a evaluar (default: el mas reciente)")
    args = parser.parse_args()
    evaluar_modelo_miml(split=args.split, ruta_experimento=args.experimento)
