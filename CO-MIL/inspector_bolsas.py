"""
=========================================================================================
INSPECTOR DE BOLSAS TENSORIALES (DEBUGGER VISUAL PARA Co-MIL)
=========================================================================================

CONTEXTO Y OBJETIVO:
Una vez que las imágenes crudas de las úlceras son procesadas, dejan de ser archivos 
visuales convencionales (.jpg o .png) y se convierten en tensores multidimensionales 
empaquetados en archivos binarios de PyTorch (.pt). 

Este script funciona como un "microscopio" para inspeccionar la integridad de los datos 
antes de inyectarlos a la red neuronal. Permite:
1. Validar la estructura matemática (Dimensiones N x C x H x W) que exige MobileNet V2.
[N, C, H, W] (Número, Canales, Alto, Ancho):
EJEMPLO: 
9 (N - Número de instancias): Es la cantidad total de recortes individuales que el algoritmo extrajo de tu Bounding Box.
3 (C - Canales): Representa los canales de color RGB (Red, Green, Blue). Como este número está "adentro" en la jerarquía de cada parche, 
significa que cada uno de los 9 recortes requiere 3 matrices numéricas superpuestas para formar los colores reales.
224 (H - Height / Alto): La altura en píxeles de cada parche.
224 (W - Width / Ancho): La anchura en píxeles de cada parche.

2. Confirmar que el vector de etiquetas MIML (Y) se haya guardado correctamente.
3. Visualizar gráficamente los primeros recortes (instancias) para comprobar empíricamente 
   que el 'Reflection Padding' y el algoritmo de 'Smart Crop' funcionaron sin distorsionar 
   la biología del tejido.

INSTRUCCIONES DE EJECUCIÓN:
Para correr este script desde la raíz de tu proyecto en la terminal de VS Code 
(asegurándote de tener activado tu entorno virtual 'env'), ejecuta el siguiente comando:

    python CO-MIL/inspector_bolsas.py
=========================================================================================
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tkinter as tk
from tkinter import filedialog

def configurar_estilo_profesional():
    """Establece parámetros de visualización limpios para presentaciones académicas."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.style.use('dark_background')

def inspeccionar_bolsa_pro():
    configurar_estilo_profesional()
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) 
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar Tensor de Úlcera (.pt)",
        filetypes=[("PyTorch Tensors", "*.pt")]
    )
    
    if not file_path:
        print("Operación cancelada.")
        return
        
    # Carga de datos y metadatos espaciales
    data = torch.load(file_path)
    bolsa_X = data['X']                 
    vector_Y = data['Y']                
    class_names = data.get('class_names', ["Granulación", "Fibrina", "Callo"])
    spatial_meta = data.get('spatial_metadata', None)
    
    # --- CREACIÓN DEL LIENZO DIAGNÓSTICO ESTRICTO ---
    fig = plt.figure(figsize=(16, 9))
    
    # Definimos una grilla estricta: 2 filas, 4 columnas.
    # La fila superior (reconstrucción + info) es 2.5 veces más alta que la inferior (parches).
    gs = gridspec.GridSpec(2, 4, height_ratios=[2.5, 1], hspace=0.3, wspace=0.1)
    
    # 1. VISOR DE RECONSTRUCCIÓN ANATÓMICA (Ocupa la mitad izquierda de la fila superior)
    ax_recon = fig.add_subplot(gs[0, :2])
    
    if spatial_meta:
        grid_h, grid_w = spatial_meta['grid_shape']
        patch_size = spatial_meta['patch_size']
        full_recon = np.zeros((grid_h * patch_size, grid_w * patch_size, 3))
        
        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                if idx < bolsa_X.shape[0]:
                    parche = bolsa_X[idx].permute(1, 2, 0).numpy()
                    full_recon[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size, :] = parche
                    idx += 1
        
        ax_recon.imshow(np.clip(full_recon, 0, 1))
        # Dibujo de la malla MIL técnica
        for r in range(grid_h + 1):
            ax_recon.axhline(y=r*patch_size, color='#00ffcc', linestyle='-', linewidth=0.5, alpha=0.4)
        for c in range(grid_w + 1):
            ax_recon.axvline(x=c*patch_size, color='#00ffcc', linestyle='-', linewidth=0.5, alpha=0.4)
    else:
        ax_recon.text(0.5, 0.5, "Sin Metadatos Espaciales.\nVuelve a generar la bolsa.", ha='center', va='center')

    ax_recon.set_title("RECONSTRUCCIÓN ANATÓMICA (Malla MIL)", fontsize=14, fontweight='bold', pad=15)
    ax_recon.axis('off') # Apagamos ejes estrictamente

    # 2. PANEL DE METADATOS TÉCNICOS (Ocupa la mitad derecha de la fila superior)
    ax_meta = fig.add_subplot(gs[0, 2:])
    ax_meta.axis('off')
    
    etiquetas_activas = [class_names[i] for i, val in enumerate(vector_Y) if val == 1]
    info_text = (
        f"ARCHIVO ORIGINAL:\n{os.path.basename(file_path)}\n\n"
        f"INSTANCIAS ($N$):\n{bolsa_X.shape[0]} parches extraídos\n\n"
        f"RESOLUCIÓN DEL PARCHE:\n{bolsa_X.shape[3]}x{bolsa_X.shape[2]} px\n\n"
        f"BACKBONE OBJETIVO:\nMobileNet V2 (3 canales RGB)\n\n"
        f"DIAGNÓSTICO GLOBAL (Vector Y):\n{', '.join(etiquetas_activas) if etiquetas_activas else 'Caso Negativo'}"
    )
    
    # Cuadro de texto alineado a la izquierda para mejor lectura
    ax_meta.text(0.1, 0.5, info_text, transform=ax_meta.transAxes, fontsize=13, 
                 verticalalignment='center', horizontalalignment='left',
                 bbox=dict(facecolor='#1a1a1a', alpha=0.9, edgecolor='#00ffcc', boxstyle='round,pad=1.5'))

    # 3. MUESTRAS DE INSTANCIAS (Fila inferior, ocupando las 4 columnas secuencialmente)
    for i in range(4):
        if i < bolsa_X.shape[0]:
            ax_sample = fig.add_subplot(gs[1, i])
            parche = bolsa_X[i].permute(1, 2, 0).numpy()
            ax_sample.imshow(np.clip(parche, 0, 1))
            ax_sample.set_title(f"Instancia #{i+1}", fontsize=11, color='#cccccc', pad=8)
            ax_sample.axis('off') # Apagamos los números horribles de 0.0 a 1.0

    # Título principal de la ventana
    plt.suptitle(f"SISTEMA DE INSPECCIÓN Co-MIL - PPS FACULTAD DE CIENCIAS", fontsize=18, fontweight='bold', y=0.95, color='#00ffcc')
    
    # Ajuste fino de los márgenes para que nada se corte
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    
    print(f"Inspección finalizada para: {os.path.basename(file_path)}")
    plt.show()

if __name__ == "__main__":
    inspeccionar_bolsa_pro()