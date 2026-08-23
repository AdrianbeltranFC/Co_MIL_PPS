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

Vectorización dinámica del vector Y:
El catálogo de clases puede haber cambiado de tamaño entre distintas sesiones de
anotación (ver catalogo_tejidos.py). Para no depender de qué catálogo estaba
vigente el día en que se guardó cada bolsa, el vector Y NO se lee directamente
del archivo .pt: se reconstruye en cada __getitem__ a partir de las etiquetas
crudas ('roi_labels') contra el catálogo canónico vigente. Así, sin importar
cuántas veces crezca o se renombre el catálogo, todas las bolsas se interpretan
de forma consistente entre sí.

Partición train/val/test:
Si se indica manifest_path + split, el dataset se filtra a solo las bolsas cuya
IMAGEN de origen (no ROI) quedó asignada a ese split en splits_manifest.json
(ver particionar_dataset.py). Esto evita que ROIs de la misma foto queden
repartidas entre entrenamiento y evaluación.

Aumento de datos:
Con augment=True se aplica, por parche, un aumento geométrico y de color leve
(flip, rotación pequeña, jitter de color) — solo debe activarse para el split
de entrenamiento.
=========================================================================================
"""

import json
import os
import sys
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, Any, List, Optional

_DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(_DIRECTORIO_ACTUAL)

import catalogo_tejidos

class CoMILDataset(Dataset):
    """
    Dataset personalizado para el Aprendizaje Multinstancia y Multietiqueta (MIML).
    """
    def __init__(
        self,
        pt_folder: str,
        target_size: int = 224,
        app_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        split: Optional[str] = None,
        augment: bool = False,
    ):
        super().__init__()
        todas_las_bolsas = glob.glob(os.path.join(pt_folder, "*.pt"))
        if len(todas_las_bolsas) == 0:
            raise FileNotFoundError(f"No se encontraron tensores (.pt) en {pt_folder}")

        if manifest_path and split:
            self.file_paths = self._filtrar_por_split(todas_las_bolsas, manifest_path, split)
            if len(self.file_paths) == 0:
                raise ValueError(
                    f"El split '{split}' no tiene ninguna bolsa asignada en {manifest_path}. "
                    "¿Corriste particionar_dataset.py sobre esta misma carpeta?"
                )
        else:
            self.file_paths = todas_las_bolsas

        # Motor de Redimensionamiento Dinámico:
        # Previene el colapso del backbone (MobileNetV2) estirando los parches de
        # alta densidad (ej. 56px) a la resolución operativa exigida (224px).
        self.target_size = target_size
        self.resize = transforms.Resize((self.target_size, self.target_size), antialias=True)

        self.augment = augment
        self.transform_augmento = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ])

        # Catálogo canónico vigente: por defecto, el que usa la GUI de anotación
        # (custom_classes.json / renombres_clases.json en CO-MIL/).
        self.app_dir = app_dir or _DIRECTORIO_ACTUAL
        self.class_catalog: List[str] = catalogo_tejidos.cargar_catalogo_vigente(self.app_dir)
        self.renombres: Dict[str, str] = catalogo_tejidos.cargar_renombres(self.app_dir)

    @staticmethod
    def _filtrar_por_split(archivos: List[str], manifest_path: str, split: str) -> List[str]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifiesto = json.load(f)
        imagenes = manifiesto.get("imagenes", {})

        seleccionados = []
        sin_particion = set()
        for archivo in archivos:
            clave = catalogo_tejidos.clave_imagen_desde_archivo(archivo)
            info = imagenes.get(clave)
            if info is None:
                sin_particion.add(clave)
                continue
            if info.get("split") == split:
                seleccionados.append(archivo)

        if sin_particion:
            print(
                f"[!] Aviso: {len(sin_particion)} imagen(es) sin partición asignada en el "
                f"manifiesto, se excluyen de este dataset: {sorted(sin_particion)}"
            )
        return seleccionados

    def __len__(self) -> int:
        return len(self.file_paths)

    def _cargar_bolsa(self, idx: int) -> Dict[str, Any]:
        return torch.load(self.file_paths[idx], weights_only=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna la Bolsa (X) y el Vector de Diagnóstico (Y) como diccionario.
        torchmil.data.collate_fn (v1.0.2) espera exactamente este formato: una
        lista de dicts con las mismas claves, que además usa para generar la
        máscara de padding automáticamente a partir de la clave 'X'.
        """
        data = self._cargar_bolsa(idx)
        bolsa_X = data['X']  # Matriz biológica: [N, 3, H, W]

        etiquetas_crudas = data.get('roi_labels')
        if etiquetas_crudas is not None:
            # Camino normal: se reconstruye Y contra el catálogo vigente.
            vector_Y = catalogo_tejidos.vectorizar_etiquetas(etiquetas_crudas, self.class_catalog, self.renombres)
        else:
            # Bolsas muy antiguas sin 'roi_labels' guardado: no hay forma de
            # reconstruir Y de forma confiable, se usa el vector congelado como
            # último recurso (puede no alinear con el catálogo vigente).
            vector_Y = data['Y']

        # Validación Geométrica: Si el parche difiere de 224px, se aplica interpolación.
        if bolsa_X.shape[-1] != self.target_size:
            bolsa_X = self.resize(bolsa_X)

        if self.augment:
            # Cada parche se aumenta de forma independiente (flip/rotación/color
            # propios por instancia), no toda la bolsa con la misma transformación.
            bolsa_X = torch.stack([self.transform_augmento(parche) for parche in bolsa_X])

        return {"X": bolsa_X, "Y": vector_Y}

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Método asíncrono para recuperar la topología original de la úlcera.
        Permite a los scripts visuales reconstruir mapas de calor sin contaminar
        el flujo de tensores del entrenamiento.
        """
        data = self._cargar_bolsa(idx)
        meta = data.get('spatial_metadata', {})
        # Siempre el catálogo canónico vigente, no el guardado en el archivo:
        # así el número de clases del modelo es determinista sin importar el
        # orden en que glob.glob() devuelva los archivos.
        meta['class_names'] = list(self.class_catalog)
        return meta

    def etiquetas_no_reconocidas(self, idx: int) -> List[str]:
        """Utilidad de diagnóstico: etiquetas crudas de esta bolsa que no se
        pudieron resolver contra el catálogo vigente (posible typo/basura)."""
        data = self._cargar_bolsa(idx)
        crudas = data.get('roi_labels', [])
        return catalogo_tejidos.etiquetas_no_reconocidas(crudas, self.class_catalog, self.renombres)