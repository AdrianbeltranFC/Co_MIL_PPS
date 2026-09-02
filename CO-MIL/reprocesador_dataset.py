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
from typing import Optional
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

def _resolver_ruta_imagen(ruta_guardada: str, dataset_root: Optional[str]) -> Optional[str]:
    """La ruta a la foto original se guarda ABSOLUTA en el momento de anotar.
    Si el proyecto se movió o renombró desde entonces (pasó de verdad: rutas
    con 'Desktop/Adrián Emiliano...' de una ubicación vieja del proyecto), esa
    ruta ya no existe aunque la foto sí. Se intenta primero la ruta guardada
    tal cual, y si no existe, se busca el mismo nombre de archivo dentro de
    `dataset_root` (la carpeta de imágenes actual)."""
    if os.path.exists(ruta_guardada):
        return ruta_guardada
    if dataset_root:
        candidato = os.path.join(dataset_root, os.path.basename(ruta_guardada))
        if os.path.exists(candidato):
            return candidato
    return None


def reprocesar_carpeta(input_folder: str, target_size: int, output_folder: str = None,
                        dataset_root: Optional[str] = None) -> str:
    """Lógica pura de reprocesamiento (sin GUI): reextrae todas las bolsas .pt de
    `input_folder` a `target_size` px desde la imagen original en alta resolución.

    Preserva 'roi_labels' (y 'label_source', y el anotador original) en las
    bolsas reprocesadas -- sin esto, dataset.py no podría reconstruir el vector
    Y dinámicamente (ver catalogo_tejidos.py) para las bolsas de otra
    resolución, y perderían la protección contra el catálogo de clases
    fragmentado que sí tienen las bolsas de 224px.

    `dataset_root`: carpeta donde viven hoy las fotos .jpg originales, usada
    como respaldo si la ruta absoluta guardada en la bolsa (de cuando se
    anotó) ya no existe. Si no se indica, se asume dos niveles arriba de
    `input_folder` (ej. input_folder=".../Dataset_Experto_100/Bolsas_MIL_Procesadas/224px"
    -> dataset_root=".../Dataset_Experto_100").
    """
    pt_files = [f for f in os.listdir(input_folder) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError(f"No se encontraron archivos .pt en {input_folder}")

    if dataset_root is None:
        dataset_root = os.path.dirname(os.path.dirname(input_folder))

    if output_folder is None:
        base_dir = os.path.dirname(input_folder)
        output_folder = os.path.join(base_dir, f"{target_size}px")
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n--- INICIANDO REPROCESAMIENTO MASIVO A {target_size}x{target_size} px ---")
    print(f"    (respaldo de fotos originales: {dataset_root})")

    procesados = 0
    reubicados = 0
    for file_name in pt_files:
        input_path = os.path.join(input_folder, file_name)
        data = torch.load(input_path, weights_only=False)

        try:
            img_path_guardada = data['original_file']
            user_bbox = data['spatial_metadata']['user_bbox']
            vector_Y = data['Y']
            class_names = data['class_names']

            img_path = _resolver_ruta_imagen(img_path_guardada, dataset_root)
            if img_path is None:
                raise FileNotFoundError(img_path_guardada)
            if img_path != img_path_guardada:
                reubicados += 1

            # --- SE VUELVE A ABRIR LA FOTOGRAFÍA ORIGINAL DE ALTA RESOLUCIÓN ---
            img = Image.open(img_path).convert('RGB')

            bolsa_X, new_spatial_meta = smart_expansion_headless(img, user_bbox, target_size)

            # Preservar proveniencia (anotador, índice de ROI) del metadato original
            meta_original = data.get('spatial_metadata', {}) or {}
            if meta_original.get('annotator'):
                new_spatial_meta['annotator'] = meta_original['annotator']
            if 'roi_index' in meta_original:
                new_spatial_meta['roi_index'] = meta_original['roi_index']
            if 'roi_total' in meta_original:
                new_spatial_meta['roi_total'] = meta_original['roi_total']

            out_path = os.path.join(output_folder, file_name)
            torch.save({
                'X': bolsa_X,
                'Y': vector_Y,
                'class_names': class_names,
                'spatial_metadata': new_spatial_meta,
                'original_file': img_path,
                'roi_labels': data.get('roi_labels', []),
                'label_source': data.get('label_source'),
            }, out_path)

            procesados += 1
            print(f"[{procesados}/{len(pt_files)}] Procesado: {file_name} -> Grid: {new_spatial_meta['grid_shape']} (N={bolsa_X.shape[0]})")

        except KeyError as e:
            print(f"[!] Omitiendo {file_name}: Faltan metadatos originales ({e}). Se requiere procesar con el nuevo generador_bolsas.py")
        except FileNotFoundError as e:
            print(f"[!] Omitiendo {file_name}: No se encuentra la foto original ni en la ruta guardada "
                  f"({e}) ni en {dataset_root}.")

    print(f"\nPROCESO TERMINADO. {procesados}/{len(pt_files)} bolsas reprocesadas en: {output_folder}")
    if reubicados:
        print(f"    ({reubicados} de ellas usaron la foto encontrada en {dataset_root} porque la ruta "
              f"guardada originalmente ya no existía)")
    return output_folder


def ejecutar_reprocesamiento():
    """Versión interactiva (GUI): pide carpeta de origen y tamaño con diálogos."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    messagebox.showinfo("Reprocesador Dataset", "Selecciona la carpeta que contiene los archivos .pt originales (Ej. la carpeta 224px).")
    input_folder = filedialog.askdirectory(title="Carpeta de origen (.pt de 224px)")
    if not input_folder:
        return

    target_size = simpledialog.askinteger("Resolución Objetivo", "Ingresa el nuevo tamaño del parche en píxeles (ej. 56, 112):", minvalue=16, maxvalue=224)
    if not target_size:
        return

    try:
        output_folder = reprocesar_carpeta(input_folder, target_size)
        messagebox.showinfo("Éxito", f"Reprocesamiento completado.\nBolsas guardadas en:\n{output_folder}")
    except FileNotFoundError as e:
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reprocesador masivo de bolsas Co-MIL a otra resolución de parche")
    parser.add_argument("--entrada", default=None, help="Carpeta con las bolsas .pt originales (si se omite, se abre selector gráfico)")
    parser.add_argument("--tamano", type=int, default=None, help="Nuevo tamaño de parche en px (si se omite, se pide con diálogo)")
    parser.add_argument("--salida", default=None, help="Carpeta de salida (default: <tamano>px junto a la carpeta de entrada)")
    args = parser.parse_args()

    if args.entrada and args.tamano:
        reprocesar_carpeta(args.entrada, args.tamano, args.salida)
    else:
        ejecutar_reprocesamiento()