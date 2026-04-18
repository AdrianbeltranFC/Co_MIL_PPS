"""
=========================================================================================
SCRIPT DE VALIDACIÓN DE FLUJO: HEATMAPS WSSS CLASE-ESPECÍFICOS (MIML)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (DEBUGGING VISUAL).
CÓMO EJECUTAR DESDE VS CODE: python CO-MIL/validar_flujo_visual.py

Objetivo:
Demostrar visualmente que la arquitectura emite K mapas de atención 
independientes, adaptándose a cualquier cantidad de tejidos. Cuenta con 
renderizado responsivo (anti-desbordamiento) para pantallas estándar.
=========================================================================================
"""

import os
import sys
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torchvision.transforms as transforms
import cv2

directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)

from Models.attention_mil import CoMILNetwork

def visualizar_heatmaps_miml(bolsa_X: torch.Tensor, pesos_atencion: torch.Tensor, meta: dict, vector_Y: list):
    """
    Dibuja N mapas de calor simultáneos utilizando una cuadrícula adaptativa 
    para evitar recortes en la pantalla del usuario.
    """
    if not meta.get('grid_shape'):
        print("[!] Advertencia: Bolsa sin metadata espacial.")
        return

    grid_h, grid_w = meta['grid_shape']
    patch_size = meta['patch_size']
    class_names = meta.get('class_names', ["Granulación", "Fibrina", "Callo"])
    num_classes = len(class_names)
    
    # 1. Reconstrucción de la Fotografía Base
    full_recon = np.zeros((grid_h * patch_size, grid_w * patch_size, 3))
    idx = 0
    for r in range(grid_h):
        for c in range(grid_w):
            if idx < bolsa_X.shape[0]:
                parche = bolsa_X[idx].permute(1, 2, 0).numpy()
                full_recon[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size, :] = parche
                idx += 1

    # 2. Configuración Estética y Adaptativa de Matplotlib
    plt.style.use('dark_background')
    
    # Restricción arquitectónica: Máximo 3 columnas para que encaje en monitores 1080p
    cols = min(num_classes, 3) 
    rows = math.ceil(num_classes / cols)
    
    # constrained_layout=True evita colisiones entre títulos y barras de color
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows + 1.5), constrained_layout=True)
    
    # Normalizamos el arreglo de ejes para poder iterarlo sin importar su forma
    if num_classes == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()
    
    # Título Principal
    etiquetas_activas = [class_names[i] for i, val in enumerate(vector_Y) if val == 1]
    diagnostico = ', '.join(etiquetas_activas) if etiquetas_activas else 'Negativo'
    fig.suptitle(f"Validación WSSS MIML: Mapas de Activación Independientes\nDiagnóstico Clínico (Y): [{diagnostico}]", 
                 fontsize=16, fontweight='bold', color='#00ffcc')

    # 3. Iteración sobre cada tejido para dibujar su mapa
    for k in range(num_classes):
        ax = axes_list[k]
        tejido_nombre = class_names[k]
        realidad_binaria = int(vector_Y[k])
        
        pesos_clase_k = pesos_atencion[k] 
        
        # Construimos el mapa 2D
        attention_map_2d = np.zeros((grid_h, grid_w))
        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                if idx < bolsa_X.shape[0]:
                    attention_map_2d[r, c] = pesos_clase_k[idx].item()
                    idx += 1
                    
        # Escalamiento
        heatmap_upscaled = cv2.resize(attention_map_2d, (grid_w * patch_size, grid_h * patch_size), interpolation=cv2.INTER_NEAREST)
        
        # Renderizado de capas
        ax.imshow(np.clip(full_recon, 0, 1))
        im = ax.imshow(heatmap_upscaled, cmap='jet', alpha=0.55)
        
        # Diseño de Títulos (Verde = Presente, Rojo = Ausente)
        color_titulo = "#28a745" if realidad_binaria == 1 else "#dc3545"
        titulo = f"Detector: {tejido_nombre}\nPresencia Real: {'SÍ (1)' if realidad_binaria == 1 else 'NO (0)'}"
        ax.set_title(titulo, fontsize=12, pad=10, color=color_titulo)
        ax.axis('off')
        
        # Barra de color VERTICAL (ahorra espacio y previene que la herida se aplane)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
        if k == 0: 
            cbar.set_label("Probabilidad de Atención (α)", fontsize=10)

    # 4. Limpieza visual: Apagamos sub-gráficos vacíos (ej. si hay 4 tejidos en una grilla de 2x3)
    for k in range(num_classes, len(axes_list)):
        axes_list[k].axis('off')

    plt.show()

def validar_pipeline_manual():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) 
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar Tensor de Úlcera (.pt) para Validación MIML",
        filetypes=[("PyTorch Tensors", "*.pt")]
    )
    
    if not file_path: return
        
    data = torch.load(file_path)
    bolsa_X = data['X']                 
    vector_Y = data['Y']                
    meta = data.get('spatial_metadata', {})
    meta['class_names'] = data.get('class_names', ["Granulación", "Fibrina", "Callo"])
    
    num_tejidos = len(meta['class_names'])
    print(f"\n[+] Archivo: {os.path.basename(file_path)}")
    print(f"[+] Clases detectadas ({num_tejidos}): {meta['class_names']}")

    target_size = 224
    if bolsa_X.shape[-1] != target_size:
        resize_op = transforms.Resize((target_size, target_size), antialias=True)
        bolsa_X_procesada = resize_op(bolsa_X)
    else:
        bolsa_X_procesada = bolsa_X

    batch_X = bolsa_X_procesada.unsqueeze(0)
    mask = torch.ones((1, bolsa_X.shape[0]), dtype=torch.bool)

    modelo = CoMILNetwork(num_classes=num_tejidos)
    modelo.eval()
    
    with torch.no_grad(): 
        logits, probabilidades_atencion = modelo(batch_X, mask)
    
    pesos_limpios = probabilidades_atencion[0] 
    vector_Y_list = vector_Y.tolist()
    
    visualizar_heatmaps_miml(bolsa_X, pesos_limpios, meta, vector_Y_list)

if __name__ == "__main__":
    validar_pipeline_manual()