"""
=========================================================================================
INGESTA DE DATOS PARA EL REDISEÑO ESTILO CAM (Co-MIL)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (NO SE EJECUTA DIRECTAMENTE).

Por qué existe (23-ago-2026): dataset.py / CoMILDataset devuelve N parches recortados
independientes, pensados para InstanceEncoder (codifica cada parche por separado). El
rediseño estilo CAM necesita en cambio la imagen COMPLETA del ROI, para pasarla una sola
vez por el backbone y usar su mapa de features nativo como grilla de instancias (ver
_proceso_claude/scripts/prototipo_cam_feasibility.py, que validó la premisa antes de
escribir este módulo).

CoMILDatasetCAM reconstruye esa imagen completa a partir de los mismos parches ya
guardados (misma lógica de reensamblado que visualizar_resultados.py), así que no
depende de volver a tocar generador_bolsas.py ni las imágenes .jpg originales. Reutiliza
catalogo_tejidos para la vectorización de etiquetas (idéntica a CoMILDataset, para que
ambos enfoques sean comparables sobre exactamente las mismas etiquetas) y el método
estático de CoMILDataset para filtrar por split, en vez de duplicar esa lógica.

Esta clase NO reemplaza a CoMILDataset -- conviven para poder comparar el enfoque
"MIL por parches" (original) contra el estilo CAM sobre el mismo dataset.
=========================================================================================
"""

import glob
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

_DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(_DIRECTORIO_ACTUAL)

import catalogo_tejidos
from dataset import CoMILDataset


class CoMILDatasetCAM(Dataset):
    """Devuelve, por bolsa, la imagen COMPLETA del ROI reensamblada a partir de sus
    parches (no una lista de parches) más el vector Y -- ver docstring del módulo."""

    def __init__(
        self,
        pt_folder: str,
        app_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        split: Optional[str] = None,
    ):
        super().__init__()
        todas_las_bolsas = glob.glob(os.path.join(pt_folder, "*.pt"))
        if len(todas_las_bolsas) == 0:
            raise FileNotFoundError(f"No se encontraron tensores (.pt) en {pt_folder}")

        if manifest_path and split:
            # Reutiliza el filtro de CoMILDataset (agrupa por IMAGEN, no por ROI) en
            # vez de duplicar esa lógica -- el criterio de partición debe ser
            # idéntico entre ambos enfoques para que la comparación sea justa.
            self.file_paths = CoMILDataset._filtrar_por_split(todas_las_bolsas, manifest_path, split)
            if len(self.file_paths) == 0:
                raise ValueError(
                    f"El split '{split}' no tiene ninguna bolsa asignada en {manifest_path}."
                )
        else:
            self.file_paths = todas_las_bolsas

        self.app_dir = app_dir or _DIRECTORIO_ACTUAL
        self.class_catalog: List[str] = catalogo_tejidos.cargar_catalogo_vigente(self.app_dir)
        self.renombres: Dict[str, str] = catalogo_tejidos.cargar_renombres(self.app_dir)

    def __len__(self) -> int:
        return len(self.file_paths)

    def _cargar_bolsa(self, idx: int) -> Dict[str, Any]:
        return torch.load(self.file_paths[idx], weights_only=False)

    @staticmethod
    def _reconstruir_roi(bolsa_X: torch.Tensor, grid_shape, patch_size: int) -> torch.Tensor:
        """Reensambla la imagen real del ROI a partir de los parches guardados,
        tal como se guardaron en la malla original (misma lógica que
        visualizar_resultados.dibujar_panel_heatmaps)."""
        grid_h, grid_w = grid_shape
        lienzo = torch.zeros(3, grid_h * patch_size, grid_w * patch_size, dtype=bolsa_X.dtype)
        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                if idx < bolsa_X.shape[0]:
                    lienzo[:, r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = bolsa_X[idx]
                    idx += 1
        return lienzo

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._cargar_bolsa(idx)
        meta = data["spatial_metadata"]
        imagen = self._reconstruir_roi(data["X"], meta["grid_shape"], meta["patch_size"])

        etiquetas_crudas = data.get("roi_labels")
        if etiquetas_crudas is not None:
            vector_Y = catalogo_tejidos.vectorizar_etiquetas(etiquetas_crudas, self.class_catalog, self.renombres)
        else:
            vector_Y = data["Y"]

        return {"imagen": imagen, "Y": vector_Y}

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        data = self._cargar_bolsa(idx)
        meta = dict(data.get("spatial_metadata", {}))
        meta["class_names"] = list(self.class_catalog)
        return meta
