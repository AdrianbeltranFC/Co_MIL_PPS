"""
=========================================================================================
SCRIPT DE VALIDACIÓN DE FLUJO Y VISUALIZACIÓN DE ATENCIÓN (MVP)
=========================================================================================
Objetivo: 
Confirmar que el pipeline de datos (Dataset -> DataLoader -> Modelo Co-MIL) funciona 
sin errores de dimensionalidad y generar un gráfico de los mapas de atención 
para evaluación visual en documentos académicos (Tesis).

Nota: Al usar un modelo no entrenado, los pesos de atención serán aleatorios, pero 
la estructura del tensor será matemáticamente correcta.

# Correr con: python CO-MIL\validar_flujo_visual.py
=========================================================================================
"""

import os
import sys # Importante para manipular las rutas del sistema
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Pedirle a Python que busque mis módulos en la misma carpeta 
# donde está guardado este script (CO-MIL), sin importar desde 
# dónde abrí la terminal. 
# =====================================================================
directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)
# Importamos las clases que construiste previamente
from dataset import CoMILDataset, collate_fn_comil
from Models.attention_mil import CoMILNetwork

def visualizar_atencion_tesis(bolsa_X: torch.Tensor, pesos_atencion: torch.Tensor, vector_Y: torch.Tensor, max_parches: int = 5):
    """
    Genera un gráfico que muestra las instancias (parches)
    junto con el peso matemático que la red neuronal le asignó a cada uno.
    
    Args:
        bolsa_X: Tensor de la bolsa con forma [N, 3, 224, 224].
        pesos_atencion: Tensor 1D con los pesos probabilísticos de atención.
        vector_Y: Las etiquetas reales de la bolsa.
        max_parches: Límite de recortes a dibujar para mantener la estética.
    """
    # Convertimos los tensores a NumPy para trabajar con Matplotlib
    pesos = pesos_atencion.detach().cpu().numpy()
    Y = vector_Y.detach().cpu().numpy()
    
    # Ordenamos los parches de mayor a menor atención
    # argsort devuelve los índices ordenados de menor a mayor, [::-1] los invierte
    indices_top = np.argsort(pesos)[::-1][:max_parches]
    
    fig, axes = plt.subplots(1, len(indices_top), figsize=(15, 4))
    fig.suptitle(
        f'Validación de Flujo: Parches con Mayor Nivel de Atención\nEtiqueta Global (Granulación, Fibrina, Callo): {Y.tolist()}', 
        fontsize=14, fontweight='bold', color='#333333'
    )
    
    for i, idx in enumerate(indices_top):
        # Desnormalizamos y permutamos el parche de [C, H, W] a [H, W, C]
        parche = bolsa_X[idx].permute(1, 2, 0).numpy()
        
        # Clip para asegurar que los valores RGB estén entre 0 y 1 para matplotlib
        parche = np.clip(parche, 0, 1) 
        
        ax = axes[i] if len(indices_top) > 1 else axes
        ax.imshow(parche)
        
        # Diseño académico: Mostrar el peso probabilístico con 4 decimales
        peso_actual = pesos[idx]
        ax.set_title(f"Parche #{idx}\nAtención (α): {peso_actual:.4f}", fontsize=11)
        ax.axis('off')
        
        # Añadimos un borde sutil para enmarcar el parche
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()

def validar_pipeline():
    # 1. Configuración de Rutas (Tu directorio específico)
    ruta_bolsas = r"C:\Users\silvi\OneDrive\Documents\Co_MIL_PPS\Heridas\Bolsas_MIL_Procesadas"
    
    print(f"--- Iniciando Validación MVP en: {ruta_bolsas} ---")
    
    # 2. Instanciación del Dataset y DataLoader
    try:
        dataset = CoMILDataset(pt_folder=ruta_bolsas)
        print(f"Éxito: Se detectaron {len(dataset)} bolsas (.pt).")
    except FileNotFoundError as e:
        print(f"Error crítico: {e}")
        return

    # Usamos batch_size=2 para forzar al collate_fn a manejar el padding dinámico
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_comil)
    
    # 3. Instanciación de la Red Co-MIL
    print("Cargando arquitectura de red (InstanceEncoder + Attention + Classifier)...")
    modelo = CoMILNetwork()
    modelo.eval() # Modo evaluación para congelar Dropouts y BatchNorms
    
    # 4. Extracción de un Mini-lote (Batch)
    batch_X, batch_Y, mask = next(iter(dataloader))
    print(f"\nRadiografía del Lote:")
    print(f"- Forma del Tensor X (Padded): {batch_X.shape} -> [Batch, Max_N, Canales, Alto, Ancho]")
    print(f"- Forma de la Máscara: {mask.shape} -> Indica qué tensores son reales y cuáles son relleno negro")
    
    # 5. Pasaje hacia adelante (Forward Pass)
    print("\nEjecutando Forward Pass a través de MobileNetV2 y Gated Attention...")
    with torch.no_grad(): # Desactivamos el cálculo de gradientes para ahorrar memoria
        logits, probabilidades_atencion = modelo(batch_X, mask)
    
    print(f"Forma de Logits de Salida: {logits.shape} -> [Batch, 3 Clases]")
    print(f"Forma de Pesos de Atención: {probabilidades_atencion.shape} -> [Batch, Max_N]")
    
    # 6. Visualización de Resultados para la primera imagen del lote
    print("\nGenerando gráfico de diagnóstico...")
    # Extraemos solo las instancias reales de la primera imagen (ignorando el padding)
    instancias_reales_idx = mask[0].nonzero(as_tuple=True)[0]
    
    bolsa_limpia = batch_X[0][instancias_reales_idx]
    pesos_limpios = probabilidades_atencion[0][instancias_reales_idx]
    
    visualizar_atencion_tesis(bolsa_limpia, pesos_limpios, batch_Y[0])

if __name__ == "__main__":
    validar_pipeline()