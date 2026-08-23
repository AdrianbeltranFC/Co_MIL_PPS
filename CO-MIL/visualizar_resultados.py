"""
=========================================================================================
GALERÍA DE MAPAS DE ATENCIÓN (Co-MIL) — Reporte visual para revisión humana
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (REPORTE VISUAL).
CÓMO EJECUTAR: python CO-MIL/visualizar_resultados.py [test|val|train] [--refinar] [--max N]

Objetivo:
Misma lógica de dibujo que validar_flujo_visual.py (K mapas de atención independientes
por tejido, superpuestos a la reconstrucción de la bolsa), pero sin diálogo de archivo:
recorre TODAS las bolsas de un split y guarda cada panel como PNG. Pensado para revisar
en bloque si el modelo atiende a las regiones correctas y si las anotaciones tienen
sentido clínico, sin abrir la GUI bolsa por bolsa.

Con --refinar, además de la malla gruesa (un valor por parche), cada mapa de la(s)
clase(s) realmente presente(s) en la bolsa se refina con refinamiento_crf.py (afinado
tipo DenseCRF guiado por la imagen real). Solo se refinan las clases positivas de cada
bolsa (no las 12), porque el refinamiento es lento (unos 10-20s por mapa) -- para una
galería completa en 12 clases x 37 bolsas tomaría horas; usa --max para generar una
muestra chica primero.

Requiere haber entrenado antes con entrenar_comil.py.
=========================================================================================
"""

import argparse
import os
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

_DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(_DIRECTORIO_ACTUAL)

from dataset import CoMILDataset
from Models.attention_mil import CoMILNetwork
from refinamiento_crf import refinar_mapa_atencion


def dibujar_panel_heatmaps(bolsa_X: torch.Tensor, pesos_atencion: torch.Tensor, meta: dict, vector_Y: list,
                            out_path: str, refinar: bool = False):
    """Idéntica lógica de dibujo que validar_flujo_visual.visualizar_heatmaps_miml,
    pero guarda a archivo en vez de mostrar en pantalla. Si refinar=True, las clases
    positivas se afinan con refinamiento_crf.refinar_mapa_atencion antes de dibujarse."""
    if not meta.get("grid_shape"):
        print(f"[!] {os.path.basename(out_path)}: sin metadata espacial, se omite.")
        return False

    grid_h, grid_w = meta["grid_shape"]
    patch_size = meta["patch_size"]
    class_names = meta.get("class_names", [])
    num_classes = len(class_names)

    full_recon = np.zeros((grid_h * patch_size, grid_w * patch_size, 3))
    idx = 0
    for r in range(grid_h):
        for c in range(grid_w):
            if idx < bolsa_X.shape[0]:
                parche = bolsa_X[idx].permute(1, 2, 0).numpy()
                full_recon[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size, :] = parche
                idx += 1

    plt.style.use("dark_background")
    cols = min(num_classes, 3)
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows + 1.5), constrained_layout=True)
    axes_list = [axes] if num_classes == 1 else axes.flatten()

    etiquetas_activas = [class_names[i] for i, val in enumerate(vector_Y) if val == 1]
    diagnostico = ", ".join(etiquetas_activas) if etiquetas_activas else "Negativo"
    fig.suptitle(
        f"{os.path.basename(out_path)}\nDiagnóstico Clínico (Y): [{diagnostico}]",
        fontsize=14, fontweight="bold", color="#00ffcc",
    )

    for k in range(num_classes):
        ax = axes_list[k]
        tejido_nombre = class_names[k]
        realidad_binaria = int(vector_Y[k])
        pesos_clase_k = pesos_atencion[k]

        attention_map_2d = np.zeros((grid_h, grid_w))
        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                if idx < bolsa_X.shape[0]:
                    attention_map_2d[r, c] = pesos_clase_k[idx].item()
                    idx += 1

        heatmap_upscaled = cv2.resize(attention_map_2d, (grid_w * patch_size, grid_h * patch_size), interpolation=cv2.INTER_NEAREST)

        titulo_extra = ""
        if refinar and realidad_binaria == 1:
            imagen_clip = np.clip(full_recon, 0, 1).astype(np.float32)
            heatmap_upscaled = refinar_mapa_atencion(imagen_clip, heatmap_upscaled.astype(np.float32))
            titulo_extra = " (refinado)"

        ax.imshow(np.clip(full_recon, 0, 1))
        im = ax.imshow(heatmap_upscaled, cmap="jet", alpha=0.55)

        color_titulo = "#28a745" if realidad_binaria == 1 else "#dc3545"
        ax.set_title(
            f"Detector: {tejido_nombre}{titulo_extra}\nPresencia Real: {'SÍ (1)' if realidad_binaria == 1 else 'NO (0)'}",
            fontsize=11, pad=8, color=color_titulo,
        )
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

    for k in range(num_classes, len(axes_list)):
        axes_list[k].axis("off")

    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def generar_galeria(split: str = "test", max_bolsas: int = None, refinar: bool = False):
    RAIZ_REPO = os.path.dirname(_DIRECTORIO_ACTUAL)
    RUTA_PESOS = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "comil_miml_fase1.pth")

    if not os.path.exists(RUTA_PESOS):
        print(f"[!] No se encontraron pesos en {RUTA_PESOS}. Entrena primero con entrenar_comil.py.")
        return

    checkpoint = torch.load(RUTA_PESOS, map_location="cpu")
    num_classes = checkpoint["num_classes"]
    ruta_bolsas = checkpoint.get("ruta_bolsas")
    ruta_manifest = checkpoint.get("ruta_manifest")

    if not ruta_bolsas or not ruta_manifest:
        print("[!] Este checkpoint no tiene rutas de dataset/manifiesto guardadas. Reentrena con la versión actual de entrenar_comil.py.")
        return

    dataset = CoMILDataset(pt_folder=ruta_bolsas, target_size=224, manifest_path=ruta_manifest, split=split, augment=False)

    modelo = CoMILNetwork(num_classes=num_classes)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    sufijo = "_refinado" if refinar else ""
    out_dir = os.path.join(RAIZ_REPO, "Pesos_Entrenados", f"heatmaps_{split}{sufijo}")
    os.makedirs(out_dir, exist_ok=True)

    n = len(dataset) if max_bolsas is None else min(max_bolsas, len(dataset))
    if refinar:
        print(f"[i] Refinamiento activado: solo se afinan las clases positivas de cada "
              f"bolsa (más lento, ~10-20s por mapa). Generando {n} bolsas.")

    generadas = 0
    for idx in range(n):
        item = dataset[idx]
        bolsa_X = item["X"]
        vector_Y = item["Y"].tolist()
        meta = dataset.get_metadata(idx)

        with torch.no_grad():
            batch_X = bolsa_X.unsqueeze(0)
            mask = torch.ones((1, bolsa_X.shape[0]), dtype=torch.bool)
            _, pesos_atencion = modelo(batch_X, mask)

        nombre_base = os.path.splitext(os.path.basename(dataset.file_paths[idx]))[0]
        out_path = os.path.join(out_dir, f"{nombre_base}.png")
        if dibujar_panel_heatmaps(bolsa_X, pesos_atencion[0], meta, vector_Y, out_path, refinar=refinar):
            generadas += 1
            print(f"[{idx + 1}/{n}] {nombre_base} -> guardado")

    print(f"\n[+] {generadas} paneles de heatmap guardados en: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Galeria de mapas de atencion Co-MIL")
    parser.add_argument("split", nargs="?", default="test", choices=["train", "val", "test"])
    parser.add_argument("--refinar", action="store_true", help="Afina las clases positivas con refinamiento_crf (mas lento)")
    parser.add_argument("--max", type=int, default=None, help="Limite de bolsas a procesar (util junto con --refinar)")
    args = parser.parse_args()
    generar_galeria(args.split, max_bolsas=args.max, refinar=args.refinar)
