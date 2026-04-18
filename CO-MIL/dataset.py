"""
=========================================================================================
MÓDULO DE INGESTA DE DATOS (DATASET PARA Co-MIL)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (NO SE EJECUTA DIRECTAMENTE).
Es importado por los scripts de entrenamiento y validación.

Objetivo:
Gestiona la carga de las bolsas tensoriales serializadas (.pt) hacia la memoria RAM.
Implementa una arquitectura compatible con la librería externa 'torchmil'.
Para evitar crasheos de dimensionalidad con 'torchmil.data.collate_fn', el método
__getitem__ devuelve estrictamente tensores biológicos, aislando los metadatos
espaciales en un método independiente (get_metadata).
=========================================================================================
"""

import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple, Dict, Any

class CoMILDataset(Dataset):
    """
    Dataset personalizado para el Aprendizaje Multinstancia y Multietiqueta (MIML).
    """
    def __init__(self, pt_folder: str, target_size: int = 224):
        super().__init__()
        self.file_paths = glob.glob(os.path.join(pt_folder, "*.pt"))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No se encontraron tensores (.pt) en {pt_folder}")
            
        # Motor de Redimensionamiento Dinámico:
        # Previene el colapso del backbone (MobileNetV2) estirando los parches de 
        # alta densidad (ej. 56px) a la resolución operativa exigida (224px).
        self.target_size = target_size
        self.resize = transforms.Resize((self.target_size, self.target_size), antialias=True)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna ESTRICTAMENTE la Bolsa (X) y el Vector de Diagnóstico (Y).
        Esta limpieza es obligatoria para la compatibilidad con torchmil.data.collate_fn.
        """
        data = torch.load(self.file_paths[idx])
        bolsa_X = data['X']  # Matriz biológica: [N, 3, H, W]
        vector_Y = data['Y'] # Vector MIML dinámico
        
        # Validación Geométrica: Si el parche difiere de 224px, se aplica interpolación.
        if bolsa_X.shape[-1] != self.target_size:
            bolsa_X = self.resize(bolsa_X)
            
        return bolsa_X, vector_Y

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Método asíncrono para recuperar la topología original de la úlcera.
        Permite a los scripts visuales reconstruir mapas de calor sin contaminar 
        el flujo de tensores del entrenamiento.
        """
        data = torch.load(self.file_paths[idx])
        meta = data.get('spatial_metadata', {})
        # Preserva el mapeo de los tejidos anotados dinámicamente por el experto
        meta['class_names'] = data.get('class_names', ["Granulación", "Fibrina", "Callo"])
        return meta