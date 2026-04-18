"""
=========================================================================================
SCRIPT DE VALIDACIÓN DE FLUJO: MAPA DE CALOR WSSS (SELECCIÓN MANUAL)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (DEBUGGING VISUAL).
CÓMO EJECUTAR DESDE VS CODE: python CO-MIL/validar_flujo_visual.py

Objetivo: 
1. Cargar una bolsa tensorial (.pt) específica seleccionada por el usuario.
2. Aplicar la protección geométrica (reescalado a 224px) requerida por MobileNetV2.
3. Simular un lote de tamaño 1 (Batch=1) para ejecutar el Forward Pass en la red.
4. Proyectar los pesos de atención (aleatorios pre-entrenamiento) sobre la fotografía 
   real para generar el Heatmap diagnóstico WSSS.
=========================================================================================
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torchvision.transforms as transforms
import cv2

# =====================================================================
# Configuración de Rutas para importar módulos locales
# =====================================================================
directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)

from Models.attention_mil import CoMILNetwork

def visualizar_heatmap_anatomico(bolsa_X: torch.Tensor, pesos_atencion: torch.Tensor, meta: dict, vector_Y: list):
    """
    Fusiona los tensores biológicos y los coeficientes de atención en un solo lienzo.
    """
    if not meta.get('grid_shape'):
        print("[!] Advertencia: Bolsa sin metadata espacial. Genera las bolsas nuevamente con la UI.")
        return

    grid_h, grid_w = meta['grid_shape']
    patch_size = meta['patch_size']
    class_names = meta.get('class_names', ["Granulación", "Fibrina", "Callo"])
    
    # 1. Lienzo para reconstruir la anatomía visual
    full_recon = np.zeros((grid_h * patch_size, grid_w * patch_size, 3))
    
    # 2. Matriz 2D para mapear matemáticamente la atención de la red
    attention_map_2d = np.zeros((grid_h, grid_w))
    
    idx = 0
    for r in range(grid_h):
        for c in range(grid_w):
            if idx < bolsa_X.shape[0]:
                # Inyección visual: de [C, H, W] a [H, W, C]
                parche = bolsa_X[idx].permute(1, 2, 0).numpy()
                full_recon[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size, :] = parche
                
                # Inyección del peso de atención extraído del tensor del modelo
                attention_map_2d[r, c] = pesos_atencion[idx].item()
                idx += 1

    # 3. Escalamiento del mapa de calor para que coincida exactamente con la fotografía
    attention_heatmap_upscaled = cv2.resize(attention_map_2d, (grid_w * patch_size, grid_h * patch_size), interpolation=cv2.INTER_NEAREST)

    # 4. Renderizado Final
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.imshow(np.clip(full_recon, 0, 1))
    
    # Superposición de atención (Capa Alpha con Colormap 'jet')
    im = ax.imshow(attention_heatmap_upscaled, cmap='jet', alpha=0.5)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Peso Probabilístico de Atención (Alfa)")
    
    # Decodificamos el vector Y a texto
    etiquetas_activas = [class_names[i] for i, val in enumerate(vector_Y) if val == 1]
    diagnostico_texto = ', '.join(etiquetas_activas) if etiquetas_activas else 'Negativo / Sin Tejidos'
    
    ax.set_title(f"Validación WSSS: Gated Attention Heatmap\nDiagnóstico Global (MIML): {diagnostico_texto}", fontsize=12, pad=15)
    ax.axis('off')
    
    print("\n[+] Renderizando Mapa de Calor WSSS.")
    print("    (Nota: Los puntos de calor son aleatorios porque la red aún no está entrenada).")
    plt.tight_layout()
    plt.show()

def validar_pipeline_manual():
    # --- 1. SELECCIÓN MANUAL DEL ARCHIVO ---
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) 
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar Tensor de Úlcera (.pt) para Validación",
        filetypes=[("PyTorch Tensors", "*.pt")]
    )
    
    if not file_path:
        print("Operación cancelada.")
        return
        
    print(f"\n--- Iniciando Validación MIL para: {os.path.basename(file_path)} ---")

    # --- 2. CARGA Y EXTRACCIÓN DE DATOS ---
    data = torch.load(file_path)
    bolsa_X = data['X']                 # Tensor original [N, 3, H, W]
    vector_Y = data['Y']                # Tensor MIML [Num_Clases]
    meta = data.get('spatial_metadata', {})
    meta['class_names'] = data.get('class_names', ["Granulación", "Fibrina", "Callo"])
    
    num_tejidos = len(meta['class_names'])
    print(f"-> Clases dinámicas detectadas ({num_tejidos}): {meta['class_names']}")

    # --- 3. PROTECCIÓN GEOMÉTRICA (Emulación del Dataset) ---
    # MobileNetV2 exige 224px. Si el usuario elige un parche de 56px, lo escalamos en RAM.
    target_size = 224
    if bolsa_X.shape[-1] != target_size:
        print(f"-> Redimensionando parche de {bolsa_X.shape[-1]}px a {target_size}px para MobileNetV2...")
        resize_op = transforms.Resize((target_size, target_size), antialias=True)
        bolsa_X_procesada = resize_op(bolsa_X)
    else:
        bolsa_X_procesada = bolsa_X

    # --- 4. PREPARACIÓN DEL LOTE (BATCH = 1) ---
    # La red espera [Batch, N, C, H, W]. Le agregamos la dimensión extra (unsqueeze).
    batch_X = bolsa_X_procesada.unsqueeze(0)
    
    # Creamos una máscara llena de "Trues" (1s) porque aquí no hay padding que ocultar, 
    # ya que es una sola bolsa sin agrupar con otras más pequeñas.
    mask = torch.ones((1, bolsa_X.shape[0]), dtype=torch.bool)

    # --- 5. INFERENCIA EN LA RED Co-MIL ---
    print("-> Ejecutando Forward Pass (MobileNetV2 + Gated Attention)...")
    modelo = CoMILNetwork(num_classes=num_tejidos)
    modelo.eval()
    
    with torch.no_grad(): 
        logits, probabilidades_atencion = modelo(batch_X, mask)
    
    # --- 6. VISUALIZACIÓN ---
    # Le pasamos a la función visualizadora la bolsa original (sin reescalar a 224) 
    # para que la reconstrucción anatómica se vea perfecta a la resolución que elegiste (ej. 56px).
    pesos_limpios = probabilidades_atencion[0]
    vector_Y_list = vector_Y.tolist()
    
    visualizar_heatmap_anatomico(bolsa_X, pesos_limpios, meta, vector_Y_list)

if __name__ == "__main__":
    validar_pipeline_manual()