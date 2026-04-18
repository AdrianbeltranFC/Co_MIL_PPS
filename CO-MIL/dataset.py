import os
import glob
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

class CoMILDataset(Dataset):
    """
    Dataset personalizado para cargar bolsas de instancias generadas en la etapa
    de preprocesamiento de úlceras de pie diabético.
    """
    def __init__(self, pt_folder: str):
        super().__init__()
        # Mapea todos los archivos .pt que contienen los diccionarios serializados
        self.file_paths = glob.glob(os.path.join(pt_folder, "*.pt"))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No se encontraron tensores en {pt_folder}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna la bolsa X y el vector de etiquetas MIML Y.
        """
        data: Dict[str, torch.Tensor] = torch.load(self.file_paths[idx])
        bolsa_X = data['X']  # Tensor de forma [N, 3, 224, 224]
        vector_Y = data['Y'] # Tensor de forma [3] (Granulación, Fibrina, Callo)
        
        return bolsa_X, vector_Y

def collate_fn_comil(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Función de agrupamiento (collate) para manejar bolsas de tamaño dinámico.
    Identifica la bolsa con mayor N en el mini-lote y rellena el resto con ceros,
    generando una máscara booleana para ignorar los parches fantasma.
    """
    bolsas_X = [item[0] for item in batch]
    vectores_Y = [item[1] for item in batch]

    # Encontrar la N máxima en el batch actual
    max_n = max(x.size(0) for x in bolsas_X)
    _, C, H, W = bolsas_X[0].shape
    batch_size = len(batch)

    # Inicializar tensores vacíos (rellenados con ceros)
    padded_X = torch.zeros((batch_size, max_n, C, H, W), dtype=torch.float32)
    # Máscara booleana: True para instancias reales, False para padding
    mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    
    Y_batch = torch.stack(vectores_Y)

    for i, x in enumerate(bolsas_X):
        n = x.size(0)
        padded_X[i, :n] = x
        mask[i, :n] = True  # Activamos las posiciones con tejido biológico real

    return padded_X, Y_batch, mask