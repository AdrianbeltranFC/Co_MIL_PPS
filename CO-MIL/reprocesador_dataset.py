"""
=========================================================================================
REDIMENSIONADOR MASIVO DE DATASETS (Co-MIL WSSS)
=========================================================================================
Objetivo: Generar bolsas de instancias a resoluciones microscópicas (ej. 56x56 px) 
heredando las coordenadas y diagnósticos trazados previamente por el experto clínico.

Ejecución: python CO-MIL/reprocesador_dataset.py
=========================================================================================
"""

import os
import math
import torch
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def smart_expansion_headless(original_img, bbox, patch_size):
    """
    Motor matemático aislado. Lee la IMAGEN ORIGINAL y recalcula la expansión 
    para la nueva métrica de parches. 
    """
    img_w, img_h = original_img.size
    x1, y1, x2, y2 = bbox
    
    # Se calcula la diferencia exacta para que la herida encaje en el nuevo múltiplo
    diff_w = (math.ceil((x2 - x1) / patch_size) * patch_size) - (x2 - x1)
    diff_h = (math.ceil((y2 - y1) / patch_size) * patch_size) - (y2 - y1)
    
    expand_left = diff_w // 2
    expand_top = diff_h // 2
    
    new_x1 = x1 - expand_left
    new_y1 = y1 - expand_top
    new_x2 = x2 + (diff_w - expand_left)
    new_y2 = y2 + (diff_h - expand_top)
    
    # Si la caja choca con el límite físico de la foto, guardamos la cantidad de 
    # píxeles faltantes para aplicar el efecto espejo más adelante.
    pad_left = abs(new_x1) if new_x1 < 0 else 0
    pad_top = abs(new_y1) if new_y1 < 0 else 0
    pad_right = new_x2 - img_w if new_x2 > img_w else 0
    pad_bottom = new_y2 - img_h if new_y2 > img_h else 0
    
    # Truncamos las coordenadas al límite de la imagen para que PIL no introduzca Zero-Padding
    new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
    new_x2, new_y2 = min(img_w, new_x2), min(img_h, new_y2)
        
    cropped_img = original_img.crop((new_x1, new_y1, new_x2, new_y2))
    tensor_img = transforms.ToTensor()(cropped_img)
    
    # --- SOLUCIÓN ARQUITECTÓNICA A LA PARADOJA DEL ZERO-PADDING ---
    # En lugar de usar píxeles negros, usamos Reflection Padding nativo de PyTorch.
    # Esto copia la textura biológica de la piel hacia afuera, manteniendo los gradientes continuos.
    if any([pad_left, pad_right, pad_top, pad_bottom]):
        padder = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        tensor_img = padder(tensor_img.unsqueeze(0)).squeeze(0)
        
    grid_h = tensor_img.shape[1] // patch_size
    grid_w = tensor_img.shape[2] // patch_size
    
    patches = tensor_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).contiguous().view(tensor_img.shape[0], -1, patch_size, patch_size).permute(1, 0, 2, 3) 
    
    spatial_metadata = {
        'grid_shape': (grid_h, grid_w),
        'patch_size': patch_size,
        'user_bbox': bbox,
        'bbox_expanded': (new_x1, new_y1, new_x2, new_y2),
        'padding_applied': (pad_left, pad_right, pad_top, pad_bottom)
    }
    return patches, spatial_metadata

def ejecutar_reprocesamiento():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) 
    
    messagebox.showinfo("Reprocesador Dataset", "Selecciona la carpeta que contiene los archivos .pt originales (Ej. la carpeta 224px).")
    input_folder = filedialog.askdirectory(title="Carpeta de origen (.pt de 224px)")
    if not input_folder: return
    
    pt_files = [f for f in os.listdir(input_folder) if f.endswith('.pt')]
    if not pt_files:
        messagebox.showerror("Error", "No se encontraron archivos .pt en la carpeta seleccionada.")
        return

    target_size = simpledialog.askinteger("Resolución Objetivo", "Ingresa el nuevo tamaño del parche en píxeles (ej. 56, 112):", minvalue=16, maxvalue=224)
    if not target_size: return

    base_dir = os.path.dirname(input_folder)
    output_folder = os.path.join(base_dir, f"{target_size}px")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n--- INICIANDO REPROCESAMIENTO MASIVO A {target_size}x{target_size} px ---")
    
    procesados = 0
    for file_name in pt_files:
        input_path = os.path.join(input_folder, file_name)
        data = torch.load(input_path)
        
        try:
            img_path = data['original_file']
            user_bbox = data['spatial_metadata']['user_bbox']
            vector_Y = data['Y']
            class_names = data['class_names']
            
            # --- SE VUELVE A ABRIR LA FOTOGRAFÍA ORIGINAL DE ALTA RESOLUCIÓN ---
            img = Image.open(img_path).convert('RGB')
            
            bolsa_X, new_spatial_meta = smart_expansion_headless(img, user_bbox, target_size)
            
            out_path = os.path.join(output_folder, file_name)
            torch.save({
                'X': bolsa_X,
                'Y': vector_Y,              
                'class_names': class_names,
                'spatial_metadata': new_spatial_meta, 
                'original_file': img_path 
            }, out_path)
            
            procesados += 1
            print(f"[{procesados}/{len(pt_files)}] Procesado: {file_name} -> Grid: {new_spatial_meta['grid_shape']} (N={bolsa_X.shape[0]})")
            
        except KeyError as e:
            print(f"[!] Omitiendo {file_name}: Faltan metadatos originales ({e}). Se requiere procesar con el nuevo generador_bolsas.py")
        except FileNotFoundError:
            print(f"[!] Omitiendo {file_name}: No se encuentra la foto original ({data.get('original_file')}).")

    messagebox.showinfo("Éxito", f"Reprocesamiento completado.\nSe generaron {procesados} bolsas en:\n{output_folder}")
    print("\nPROCESO TERMINADO.")

if __name__ == "__main__":
    ejecutar_reprocesamiento()