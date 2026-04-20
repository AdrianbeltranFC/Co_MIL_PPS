"""
=========================================================================================
FASE 2: EVALUACIÓN MIML Y GENERACIÓN DE EVIDENCIA CLÍNICA
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (MÓDULO DE EVALUACIÓN).
CÓMO EJECUTAR: python CO-MIL/evaluar_comil.py

Objetivo:
1. Calcular Hamming Loss y Exactitud de Subconjunto (Subset Accuracy).
2. Generar Matrices de Confusión por cada tejido individual.
3. Producir el Heatmap WSSS final con los pesos de atención ya optimizados.
=========================================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import hamming_loss, multilabel_confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importaciones locales
from dataset import CoMILDataset
from Models.attention_mil import CoMILNetwork
from torchmil.data import collate_fn 

def evaluar_modelo_miml():
    # --- CONFIGURACIÓN ---
    RUTA_PESOS = "Pesos_Entrenados/comil_miml_fase1.pth"
    RUTA_TEST = r"Bolsas_MIL_Procesadas\224px" # Opcional: usar carpeta separada de validación
    UMBRAL = 0.5 # Sensibilidad del diagnóstico clínico
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. CARGA DEL DICCIONARIO DE ENTRENAMIENTO
    if not os.path.exists(RUTA_PESOS):
        print(f"[!] Error: No se encontraron los pesos en {RUTA_PESOS}. Entrena primero.")
        return
        
    checkpoint = torch.load(RUTA_PESOS, map_location=dispositivo)
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    
    # 2. INSTANCIACIÓN DE LA ARQUITECTURA
    modelo = CoMILNetwork(num_classes=num_classes).to(dispositivo)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()
    
    # 3. CARGA DE DATOS
    dataset = CoMILDataset(pt_folder=RUTA_TEST, target_size=224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    y_true = []
    y_pred_logits = []
    
    print(f"\n[+] Iniciando Inferencia en {len(dataset)} imágenes...")
    
    with torch.no_grad():
        for batch_X, batch_Y, mask in tqdm(dataloader):
            batch_X, mask = batch_X.to(dispositivo), mask.to(dispositivo)
            logits, _ = modelo(batch_X, mask)
            
            y_true.append(batch_Y.numpy())
            y_pred_logits.append(torch.sigmoid(logits).cpu().numpy())
            
    # Concatenamos resultados
    y_true = np.vstack(y_true)
    y_pred_probs = np.vstack(y_pred_logits)
    y_pred_bin = (y_pred_probs > UMBRAL).astype(int)
    
    # --- CÁLCULO DE MÉTRICAS MIML ---
    hl = hamming_loss(y_true, y_pred_bin)
    print(f"\n=== REPORTE DE EVALUACIÓN FASE 2 ===")
    print(f"-> Hamming Loss: {hl:.4f} (Menor es mejor)")
    print(f"-> Subset Accuracy (Exactitud Total): {np.all(y_true == y_pred_bin, axis=1).mean():.4f}")
    
    print("\n--- Desempeño por Tejido ---")
    print(classification_report(y_true, y_pred_bin, target_names=class_names))
    
    # --- VISUALIZACIÓN 1: MATRICES DE CONFUSIÓN MULTIETIQUETA ---
    mcm = multilabel_confusion_matrix(y_true, y_pred_bin)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 4))
    plt.style.use('default')
    
    for i, (matrix, name) in enumerate(zip(mcm, class_names)):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"Matriz: {name}")
        axes[i].set_xlabel("Predicción")
        axes[i].set_ylabel("Realidad")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluar_modelo_miml()