"""
=========================================================================================
INSPECTOR DE BOLSAS TENSORIALES (DEBUGGER VISUAL PARA Co-MIL)
=========================================================================================
Proyecto de Práctica Profesional Supervisada (PPS) - Análisis de Úlceras de Pie Diabético

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
import tkinter as tk
from tkinter import filedialog

def inspeccionar_bolsa():
    """
    Función principal que despliega la interfaz de selección, carga el archivo binario
    a la memoria RAM, desglosa su dimensionalidad matemática y renderiza una muestra
    visual de los parches utilizando Matplotlib.
    """
    print("Iniciando inspector tensorial...")
    print("Abriendo el explorador de archivos... (Revisa tu barra de tareas si no lo ves)")
    
    # =========================================================
    # 1. GESTIÓN DE LA INTERFAZ DE SELECCIÓN (TKINTER)
    # =========================================================
    # Inicializamos una instancia oculta de Tkinter solo para usar su cuadro de diálogo.
    root = tk.Tk()
    root.withdraw() # Evita que aparezca una ventana gris vacía e inútil de fondo.
    
    # [SOLUCIÓN DE BUG DE VS CODE]: 
    # VS Code tiende a secuestrar el foco del sistema operativo, dejando la ventana 
    # del explorador atrapada detrás del editor. El atributo '-topmost' fuerza a nivel 
    # de sistema que la ventana salte por encima de cualquier otro programa abierto.
    root.attributes('-topmost', True) 
    
    # Abrimos el cuadro de diálogo filtrando estrictamente por archivos de PyTorch.
    file_path = filedialog.askopenfilename(
        title="Selecciona una Bolsa procesada (.pt)",
        filetypes=[("PyTorch Tensors", "*.pt")]
    )
    
    # Manejo de cancelación por parte del usuario.
    if not file_path:
        print("No seleccionaste ningún archivo. Cancelando inspección...")
        return
        
    # =========================================================
    # 2. CARGA DEL DICCIONARIO DE DATOS (I/O)
    # =========================================================
    print(f"\n--- CARGANDO BOLSA: {os.path.basename(file_path)} ---")
    
    # torch.load deserializa el archivo binario y reconstruye el diccionario 
    # de Python exactamente como lo armamos en el generador de bolsas.
    data = torch.load(file_path)
    
    # Extracción de las variables clave
    bolsa_X = data['X']                 # El tensor masivo con los recortes de la foto.
    vector_Y = data['Y']                # El tensor 1D con las etiquetas [Granulación, Fibrina, Callo].
    archivo_original = data['original_file'] # La ruta de trazabilidad a la foto .jpg original.
    
    # =========================================================
    # 3. ANÁLISIS MATEMÁTICO DE DIMENSIONALIDAD EN TERMINAL
    # =========================================================
    # Imprimimos la radiografía estructural del tensor. Esto es vital para asegurar 
    # que la matriz empata con las dimensiones de entrada que esperará el InstanceEncoder.
    print(f"1. Archivo Original: {archivo_original}")
    print(f"2. Etiqueta (Vector Y): {vector_Y.tolist()} -> [Granulación, Fibrina, Callo]")
    print(f"3. Forma de la Bolsa X: {bolsa_X.shape}")
    print(f"   - N (Total de Instancias/Parches): {bolsa_X.shape[0]}")
    print(f"   - C (Canales de color RGB):        {bolsa_X.shape[1]}")
    print(f"   - H (Alto en píxeles):             {bolsa_X.shape[2]}")
    print(f"   - W (Ancho en píxeles):            {bolsa_X.shape[3]}\n")
    
    # =========================================================
    # 4. RENDERIZADO VISUAL CON MATPLOTLIB
    # =========================================================
    # Para no saturar la pantalla (ya que una bolsa puede tener 200 parches),
    # limitamos la visualización gráfica solo a las primeras 4 instancias.
    num_parches_a_mostrar = min(4, bolsa_X.shape[0])
    
    # Creamos una figura (lienzo) con una fila y múltiples columnas.
    fig, axes = plt.subplots(1, num_parches_a_mostrar, figsize=(15, 4))
    
    # Si la herida era tan pequeña que solo generó 1 parche, 'axes' no será una lista. 
    # Lo forzamos a ser lista para que el bucle 'for' de abajo no colapse.
    if num_parches_a_mostrar == 1:
        axes = [axes] 
        
    # Título principal de la ventana con el vector diagnóstico.
    fig.suptitle(f'Inspección Visual: Primeros {num_parches_a_mostrar} parches\nVector Y: {vector_Y.tolist()} | Archivo: {os.path.basename(archivo_original)}', fontsize=14)
    
    # Iteramos sobre los primeros parches para dibujarlos uno por uno.
    for i in range(num_parches_a_mostrar):
        
        # [CONVERSIÓN CRÍTICA DE EJES]:
        # PyTorch almacena imágenes en el formato [Canal, Alto, Ancho] -> [3, 224, 224].
        # Matplotlib exige el formato [Alto, Ancho, Canal] -> [224, 224, 3].
        # La función .permute(1, 2, 0) mueve el eje de los colores al final.
        # .numpy() lo convierte de Tensor a un arreglo estándar que matplotlib pueda dibujar.
        parche_img = bolsa_X[i].permute(1, 2, 0).numpy()
        
        # Inyectamos la imagen en el sub-gráfico correspondiente.
        axes[i].imshow(parche_img)
        axes[i].set_title(f"Instancia {i+1}")
        
        # Apagamos las reglas y números de los ejes XY (no nos sirven los píxeles aquí).
        axes[i].axis('off')
        
    # Ajusta los márgenes automáticamente para que no se traslapen los títulos.
    plt.tight_layout()
    print("Generando ventana de visualización (Matplotlib)...")
    
    # Detiene la ejecución en terminal y abre la ventana gráfica. 
    # El script terminará cuando el usuario cierre esta ventana.
    plt.show()

# Punto de entrada. Asegura que la función solo se llame si ejecutamos este script directamente.
if __name__ == "__main__":
    inspeccionar_bolsa()