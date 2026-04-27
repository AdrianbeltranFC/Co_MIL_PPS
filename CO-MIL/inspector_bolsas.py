"""
=========================================================================================
INSPECTOR DE BOLSAS TENSORIALES (DEBUGGER VISUAL PARA Co-MIL)
=========================================================================================

Permite:
1. Validar la estructura matematica N x C x H x W que exige MobileNet V2.
2. Confirmar que el vector de etiquetas MIML (Y) se guardo correctamente.
3. Visualizar la reconstruccion de la bolsa y sus primeras instancias de forma limpia.

Ejecucion:
    python CO-MIL/inspector_bolsas.py
=========================================================================================
"""

import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch


def configurar_estilo_profesional():
    """Establece parametros de visualizacion mas limpios para revision tecnica."""
    plt.style.use("dark_background")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.facecolor"] = "#0e1117"
    plt.rcParams["axes.facecolor"] = "#111827"


def reconstruir_bolsa(bolsa_x, spatial_meta):
    """Reconstruye la imagen completa a partir de la malla MIL."""
    grid_h, grid_w = spatial_meta["grid_shape"]
    patch_size = spatial_meta["patch_size"]
    canvas = np.zeros((grid_h * patch_size, grid_w * patch_size, 3), dtype=np.float32)

    idx = 0
    for row in range(grid_h):
        for col in range(grid_w):
            if idx >= bolsa_x.shape[0]:
                break
            patch = bolsa_x[idx].permute(1, 2, 0).numpy()
            y0 = row * patch_size
            x0 = col * patch_size
            canvas[y0 : y0 + patch_size, x0 : x0 + patch_size, :] = patch
            idx += 1

    return canvas


def dibujar_malla(ax, grid_h, grid_w, patch_size):
    """Dibuja contornos finos sin relleno para evitar artefactos visuales."""
    for row in range(grid_h):
        for col in range(grid_w):
            rect = Rectangle(
                (col * patch_size, row * patch_size),
                patch_size,
                patch_size,
                fill=False,
                edgecolor="#19d3da",
                linewidth=0.45,
                alpha=0.35,
                joinstyle="round",
            )
            ax.add_patch(rect)


def inspeccionar_bolsa_pro():
    configurar_estilo_profesional()

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Seleccionar Tensor de Ulcera (.pt)",
        filetypes=[("PyTorch Tensors", "*.pt")],
    )

    if not file_path:
        print("Operacion cancelada.")
        return

    data = torch.load(file_path, map_location="cpu")
    bolsa_x = data["X"]
    vector_y = data["Y"]
    class_names = data.get("class_names", ["Granulación", "Fibrina", "Callo"])
    spatial_meta = data.get("spatial_metadata")

    fig = plt.figure(figsize=(16, 9), facecolor="#0e1117")
    gs = gridspec.GridSpec(2, 4, height_ratios=[2.4, 1], hspace=0.28, wspace=0.12)

    ax_recon = fig.add_subplot(gs[0, :2])
    ax_recon.set_facecolor("#0b1220")

    if spatial_meta:
        grid_h, grid_w = spatial_meta["grid_shape"]
        patch_size = spatial_meta["patch_size"]
        full_recon = reconstruir_bolsa(bolsa_x, spatial_meta)
        ax_recon.imshow(np.clip(full_recon, 0, 1), interpolation="nearest")
        dibujar_malla(ax_recon, grid_h, grid_w, patch_size)
        ax_recon.set_xlim(0, grid_w * patch_size)
        ax_recon.set_ylim(grid_h * patch_size, 0)
    else:
        ax_recon.text(
            0.5,
            0.5,
            "Sin metadatos espaciales.\nVuelve a generar la bolsa.",
            ha="center",
            va="center",
            fontsize=14,
            color="#d1d5db",
        )

    ax_recon.set_title("Reconstruccion Anatomica", fontsize=14, fontweight="bold", pad=14)
    ax_recon.axis("off")

    ax_meta = fig.add_subplot(gs[0, 2:])
    ax_meta.set_facecolor("#0b1220")
    ax_meta.axis("off")

    etiquetas_activas = [class_names[i] for i, val in enumerate(vector_y) if float(val) == 1.0]
    info_text = (
        f"ARCHIVO ORIGINAL:\n{os.path.basename(file_path)}\n\n"
        f"INSTANCIAS (N):\n{bolsa_x.shape[0]} parches extraidos\n\n"
        f"RESOLUCION DEL PARCHE:\n{bolsa_x.shape[3]} x {bolsa_x.shape[2]} px\n\n"
        f"BACKBONE OBJETIVO:\nMobileNet V2 (3 canales RGB)\n\n"
        f"DIAGNOSTICO GLOBAL (Vector Y):\n"
        f"{', '.join(etiquetas_activas) if etiquetas_activas else 'Caso Negativo'}"
    )

    ax_meta.text(
        0.08,
        0.5,
        info_text,
        transform=ax_meta.transAxes,
        fontsize=13,
        verticalalignment="center",
        horizontalalignment="left",
        color="#e5e7eb",
        bbox=dict(facecolor="#111827", alpha=0.98, edgecolor="#19d3da", boxstyle="round,pad=1.2"),
    )

    max_samples = min(4, bolsa_x.shape[0])
    for idx in range(max_samples):
        ax_sample = fig.add_subplot(gs[1, idx])
        ax_sample.set_facecolor("#0b1220")
        patch = bolsa_x[idx].permute(1, 2, 0).numpy()
        ax_sample.imshow(np.clip(patch, 0, 1), interpolation="nearest")
        ax_sample.set_title(f"Instancia #{idx + 1}", fontsize=11, color="#d1d5db", pad=8)
        ax_sample.axis("off")

        border = Rectangle(
            (0, 0),
            patch.shape[1] - 1,
            patch.shape[0] - 1,
            fill=False,
            edgecolor="#334155",
            linewidth=1.2,
        )
        ax_sample.add_patch(border)

    plt.suptitle(
        "Sistema de Inspeccion Co-MIL - PPS Facultad de Ciencias",
        fontsize=18,
        fontweight="bold",
        y=0.96,
        color="#67e8f9",
    )
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

    print(f"Inspeccion finalizada para: {os.path.basename(file_path)}")
    plt.show()


if __name__ == "__main__":
    inspeccionar_bolsa_pro()
