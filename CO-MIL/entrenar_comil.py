"""
=========================================================================================
FASE 1: MOTOR DE ENTRENAMIENTO Co-MIL (OPTIMIZACIÓN MULTI-ETIQUETA DINÁMICA)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (NÚCLEO DE ENTRENAMIENTO).
CÓMO EJECUTAR DESDE VS CODE: python CO-MIL/entrenar_comil.py

Objetivo:
Ejecutar el ciclo de aprendizaje de PyTorch para la arquitectura MIML.
Calcular automáticamente el desbalanceo de clases para penalizar errores en tejidos raros,
congela el extractor visual para proteger los pesos de ImageNet, y optimiza 
las ramas independientes usando Entropía Cruzada Binaria (BCE).
=========================================================================================

Notas para mi mismo:
1.-
Una vez que termines de generar todas tus bolsas con el etiquetador y las redimensiones,debo subir mi 
carpeta completa del proyecto (la que tiene los scripts y la carpeta de Bolsas_MIL_Procesadas) a mi Google Drive.
Sugerencia: Poner la carpeta directamente en la raíz del Drive y asegurar de que se llame Co_MIL_PPS.
2.-
Ir a colab.research.google.com y crea un nuevo cuaderno (New Notebook).
En el menú superior, hacer clic en Entorno de ejecución > Cambiar tipo de entorno de ejecución.
En Acelerador de hardware, seleccionar T4 GPU y dale a Guardar.
3.-
En la primera celda de tu cuaderno, pegar este código para darle acceso a Colab a mis archivos y 
descargar la librería médica que necesitas:
-----------------------------------------------------------
# Conectar Google Drive a Colab
from google.colab import drive
drive.mount('/content/drive')
# Instalar la librería oficial para manejo de tensores MIL
!pip install torchmil
-----------------------------------------------------------
4.-
En la segunda celda, simplemente debo llamar al script exactamente como lo haría en la terminal de 
VS Code, pero apuntando a la ruta de Drive:
-----------------------------------------------------------
# Movernos a la carpeta del proyecto
%cd "/content/drive/MyDrive/Co_MIL_PPS"
# Iniciar el motor de entrenamiento de la Fase 1
!python CO-MIL/entrenar_comil.py
-----------------------------------------------------------
Al terminar, los pesos de la red se guardarán automáticamente en mi propia carpeta de Google Drive 
(Pesos_Entrenados/comil_miml_fase1.pth), seguros y listos para que los descargue a la laptop.
ouuuyeah

"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm 

# Aseguramos que Python encuentre nuestros módulos
directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)

from dataset import CoMILDataset
from Models.attention_mil import CoMILNetwork

# Importación estandarizada para manejo de tensores asimétricos
from torchmil.data import collate_fn 

# =======================================================================================
# 1. FUNCIONES DE APOYO Y MITIGACIÓN DE SESGO
# =======================================================================================

def calcular_pos_weights(dataset: CoMILDataset, num_classes: int) -> torch.Tensor:
    """
    Escanea todo el dataset antes de entrenar para calcular el peso W_c de cada tejido.
    Fórmula: pos_weight = Muestras_Negativas / Muestras_Positivas
    """
    print("\n[+] Escaneando dataset para calcular 'pos_weights' (Mitigación de Sesgo)...")
    conteo_positivos = torch.zeros(num_classes)
    total_muestras = len(dataset)

    for i in tqdm(range(total_muestras), desc="Analizando distribuciones"):
        _, y = dataset[i]
        conteo_positivos += y

    conteo_negativos = total_muestras - conteo_positivos
    # Se suma 1e-5 para evitar divisiones por cero en caso de tejidos no anotados
    pos_weights = conteo_negativos / (conteo_positivos + 1e-5)
    
    print(f"    -> Muestras con presencia de tejido: {conteo_positivos.int().tolist()}")
    print(f"    -> Multiplicadores de Castigo (W_c): {pos_weights.round(decimals=2).tolist()}")
    return pos_weights

def congelar_backbone(modelo: nn.Module):
    """
    Congela las capas convolucionales de MobileNetV2. Ahorra VRAM y 
    preserva el conocimiento visual genérico extraído de ImageNet.
    """
    print("[+] Congelando pesos del Extractor Visual (MobileNet V2)...")
    for parametro in modelo.encoder.parameters():
        parametro.requires_grad = False
    
    params_entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"    -> Extractor bloqueado. Entrenando exclusivamente {params_entrenables:,} parámetros (Ramas de Atención + Clasificadores).")

# =======================================================================================
# 2. MOTOR PRINCIPAL DE ENTRENAMIENTO
# =======================================================================================

def entrenar_modelo():
    # --- HIPERPARÁMETROS DE LA FASE 1 ---
    # Ajusta esta ruta a donde guardes tus tensores de 224px (o los redimensionados de 56px)
    RUTA_BOLSAS = r"Bolsas_MIL_Procesadas\224px" 
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INICIANDO MOTOR Co-MIL EN DISPOSITIVO: {dispositivo.type.upper()} ===")

    # --- INGESTA DE DATOS Y TOPOLOGÍA DINÁMICA ---
    try:
        # Se fuerza el target_size a 224px para proteger la entrada del backbone
        dataset = CoMILDataset(pt_folder=RUTA_BOLSAS, target_size=224)
    except FileNotFoundError:
        print(f"[!] Error: No se encontraron bolsas en {RUTA_BOLSAS}")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # La red averigua cuántas ramas crear leyendo el primer tensor
    meta_info = dataset.get_metadata(0)
    num_tejidos = len(meta_info['class_names'])
    print(f"-> Arquitectura configurada dinámicamente para {num_tejidos} clases: {meta_info['class_names']}")
    
    # --- CONSTRUCCIÓN DEL MODELO ---
    modelo = CoMILNetwork(num_classes=num_tejidos).to(dispositivo)
    congelar_backbone(modelo)
    
    # --- EL JUEZ (Loss) Y EL MECÁNICO (Optimizer) ---
    pesos_clase = calcular_pos_weights(dataset, num_tejidos).to(dispositivo)
    criterio_loss = nn.BCEWithLogitsLoss(pos_weight=pesos_clase)
    
    # AdamW con decaimiento de pesos para evitar que la red memorice las fotos (Overfitting)
    optimizador = AdamW(filter(lambda p: p.requires_grad, modelo.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)

    # ===================================================================================
    # 3. BUCLE DE OPTIMIZACIÓN (ÉPOCAS)
    # ===================================================================================
    print(f"\n[+] Iniciando entrenamiento para {EPOCHS} épocas...\n")
    
    # Historial para saber si estamos mejorando
    historial_loss = []

    for epoch in range(EPOCHS):
        modelo.train()
        loss_acumulada = 0.0
        
        barra_batches = tqdm(dataloader, desc=f"Época {epoch+1:02d}/{EPOCHS}")
        
        for batch_X, batch_Y, mask in barra_batches:
            batch_X, batch_Y, mask = batch_X.to(dispositivo), batch_Y.to(dispositivo), mask.to(dispositivo)

            # 1. Limpieza de memoria matemática
            optimizador.zero_grad()

            # 2. Forward Pass: La red lanza sus predicciones (Logits)
            logits, _ = modelo(batch_X, mask)

            # 3. Cálculo del Castigo con Entropía Cruzada
            loss = criterio_loss(logits, batch_Y)

            # 4. Backward Pass: Cálculo de las derivadas (¿En qué dirección me equivoqué?)
            loss.backward()

            # 5. Ajuste de Pesos: El optimizador corrige las ramas de atención y el clasificador
            optimizador.step()

            loss_acumulada += loss.item()
            barra_batches.set_postfix({'Loss': f"{loss.item():.4f}"})

        loss_promedio = loss_acumulada / len(dataloader)
        historial_loss.append(loss_promedio)
        print(f" -> Fin Época {epoch+1:02d} | Loss Promedio: {loss_promedio:.4f}")
        
    print("\n=== ENTRENAMIENTO FASE 1 FINALIZADO CON ÉXITO ===")
    
    # --- GUARDADO ESTRUCTURADO ---
    os.makedirs("Pesos_Entrenados", exist_ok=True)
    ruta_modelo = os.path.join("Pesos_Entrenados", "comil_miml_fase1.pth")
    
    # Guardamos los pesos y la configuración clave para no perderla en la Fase 2
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': modelo.state_dict(),
        'optimizer_state_dict': optimizador.state_dict(),
        'loss': historial_loss[-1],
        'num_classes': num_tejidos,
        'class_names': meta_info['class_names']
    }, ruta_modelo)
    
    print(f"[+] Diccionario del modelo guardado en: {ruta_modelo}")
    print("[+] Listo para la Fase 2 (Evaluación con Hamming Loss y Precisión).")

if __name__ == "__main__":
    entrenar_modelo()