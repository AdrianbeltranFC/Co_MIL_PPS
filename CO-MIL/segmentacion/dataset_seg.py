"""
=========================================================================================
INGESTA DE DATOS PARA SEGMENTACIÓN SEMÁNTICA DE TEJIDOS  (DFUTissue)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (no se ejecuta directamente).

Contexto: tras el pivote del 27-ago-2026 (ver bitácora Parte IV), la LOCALIZACIÓN de
tejidos se ataca como segmentación semántica supervisada -- no como atención MIL. Este
módulo carga DFUTissue (imagen + máscara por píxel) para entrenar/evaluar ese brazo
supervisado sobre datos públicos mientras se anota el dataset propio.

DFUTissue (descargado con descargar_datos.py):
  datasets_publicos/DFUTissue/Labeled/Padded/{Images,Annotations}/{TrainVal,Test}/*.png
  Imágenes 256x256 RGB. Anotaciones 256x256 con valores enteros por píxel:
      0 = fondo   1 = Fibrina   2 = Granulación   3 = Callo
  Partición oficial: Labeled/labeled_train_names.txt (78), labeled_val_names.txt (16),
  test_names.txt (16).  Se usa tal cual para que los números sean comparables con el
  paper (Dhar et al., 2024).
=========================================================================================
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASES = ["Fondo", "Fibrina", "Granulación", "Callo"]
NUM_CLASES = len(CLASES)
# color RGB por clase, para visualización (coincide con palette_colorCode.txt de DFUTissue)
PALETA = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

# Normalización ImageNet (el encoder MobileNetV2 viene preentrenado en ImageNet).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _raiz_dfutissue() -> str:
    raiz_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(raiz_repo, "datasets_publicos", "DFUTissue", "Labeled", "Padded")


def _leer_nombres(ruta_txt: str) -> List[str]:
    with open(ruta_txt, "r", encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]


class DFUTissueSeg(Dataset):
    """Segmentación de tejidos en DFUTissue.

    split: 'train' | 'val' | 'test'  (usa la partición oficial del dataset).
    augment: solo debe activarse en 'train'.
    """

    def __init__(self, split: str = "train", augment: bool = False, target_size: int = 256,
                 aug_fuerte: bool = False):
        super().__init__()
        assert split in ("train", "val", "test")
        base = _raiz_dfutissue()
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"No existe {base}. Corre primero: python CO-MIL/segmentacion/descargar_datos.py"
            )
        raiz_labeled = os.path.dirname(base)  # .../Labeled
        archivos_split = {
            "train": "labeled_train_names.txt",
            "val": "labeled_val_names.txt",
            "test": "test_names.txt",
        }
        nombres = _leer_nombres(os.path.join(raiz_labeled, archivos_split[split]))

        carpeta = "Test" if split == "test" else "TrainVal"
        self.dir_img = os.path.join(base, "Images", carpeta)
        self.dir_ann = os.path.join(base, "Annotations", carpeta)
        self.nombres = [n for n in nombres if os.path.exists(os.path.join(self.dir_img, n + ".png"))]
        if len(self.nombres) != len(nombres):
            faltan = set(nombres) - set(self.nombres)
            print(f"[!] {len(faltan)} imagen(es) del split '{split}' no encontradas: {sorted(faltan)[:5]}")

        self.split = split
        self.augment = augment
        self.aug_fuerte = aug_fuerte
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.nombres)

    def _cargar(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        nombre = self.nombres[idx]
        img = Image.open(os.path.join(self.dir_img, nombre + ".png")).convert("RGB")
        ann = Image.open(os.path.join(self.dir_ann, nombre + ".png"))
        if self.target_size and img.size != (self.target_size, self.target_size):
            img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            ann = ann.resize((self.target_size, self.target_size), Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)
        ann = np.asarray(ann)
        if ann.ndim == 3:  # DFUTissue guarda la anotación como RGB con R=G=B=clase
            ann = ann[..., 0]
        ann = ann.astype(np.int64)
        ann[ann >= NUM_CLASES] = 0  # robustez ante valores inesperados
        return img, ann

    def _aumentar(self, img: np.ndarray, ann: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Geométricas: se aplican igual a imagen y máscara.
        if random.random() < 0.5:
            img, ann = img[:, ::-1], ann[:, ::-1]
        if random.random() < 0.5:
            img, ann = img[::-1], ann[::-1]
        k = random.randint(0, 3)
        if k:
            img, ann = np.rot90(img, k), np.rot90(ann, k)
        img = np.ascontiguousarray(img)
        ann = np.ascontiguousarray(ann)
        # Fotométricas: solo a la imagen (brillo/contraste leves).
        rango = 0.5 if self.aug_fuerte else 0.3
        if random.random() < 0.5:
            f = 1.0 + (random.random() - 0.5) * rango
            img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            m = img.mean()
            c = 1.0 + (random.random() - 0.5) * rango
            img = np.clip((img.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)

        if self.aug_fuerte:
            # Zoom / recorte aleatorio (escala 0.8-1.2): reencuadra imagen y máscara juntas.
            if random.random() < 0.6:
                img, ann = _escalar_recortar(img, ann)
            # Desplazamiento de matiz por canal (robustez ante tono de piel / iluminación).
            if random.random() < 0.5:
                desp = (np.random.rand(3) - 0.5) * 30
                img = np.clip(img.astype(np.float32) + desp, 0, 255).astype(np.uint8)
            # Ruido gaussiano leve (variabilidad de cámara).
            if random.random() < 0.3:
                img = np.clip(img.astype(np.float32) + np.random.randn(*img.shape) * 6,
                              0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(ann)

    def __getitem__(self, idx: int):
        img, ann = self._cargar(idx)
        if self.augment:
            img, ann = self._aumentar(img, ann)
        x = img.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(ann))
        return x, y


def _escalar_recortar(img: np.ndarray, ann: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zoom aleatorio: reescala imagen+máscara por un factor 0.8-1.2 y recorta/rellena
    al tamaño original. Da robustez ante distancia de cámara sin cambiar la resolución."""
    from PIL import Image as _Im
    h, w = ann.shape
    f = 0.8 + random.random() * 0.4
    nh, nw = max(8, int(h * f)), max(8, int(w * f))
    im = np.asarray(_Im.fromarray(img).resize((nw, nh), _Im.BILINEAR))
    an = np.asarray(_Im.fromarray(ann.astype(np.uint8)).resize((nw, nh), _Im.NEAREST)).astype(np.int64)
    out_i = np.zeros((h, w, 3), np.uint8)
    out_a = np.zeros((h, w), np.int64)
    y0 = random.randint(0, max(0, nh - h)); x0 = random.randint(0, max(0, nw - w))
    dy = random.randint(0, max(0, h - nh)); dx = random.randint(0, max(0, w - nw))
    ch, cw = min(h, nh), min(w, nw)
    out_i[dy:dy + ch, dx:dx + cw] = im[y0:y0 + ch, x0:x0 + cw]
    out_a[dy:dy + ch, dx:dx + cw] = an[y0:y0 + ch, x0:x0 + cw]
    return out_i, out_a


def pesos_sobremuestreo(ds: "DFUTissueSeg", clases_raras=(1,), factor: float = 3.0) -> list:
    """Peso por imagen para un WeightedRandomSampler: las imágenes que contienen
    alguna de `clases_raras` (por defecto Fibrina) se muestrean `factor` veces más.
    Compensa el fuerte desbalance (la fibrina ocupa <1 % de los píxeles)."""
    pesos = []
    for i in range(len(ds)):
        _, ann = ds._cargar(i)
        presentes = set(int(c) for c in np.unique(ann))
        pesos.append(factor if presentes & set(clases_raras) else 1.0)
    return pesos


def prevalencia(split_o_dataset) -> dict:
    """Nº de imágenes en las que aparece cada clase, para diagnóstico rápido."""
    ds = split_o_dataset if isinstance(split_o_dataset, Dataset) else DFUTissueSeg(split_o_dataset)
    conteo = {c: 0 for c in CLASES}
    for i in range(len(ds)):
        _, ann = ds._cargar(i)
        for c in np.unique(ann):
            conteo[CLASES[int(c)]] += 1
    return conteo


def colorear(mascara: np.ndarray) -> np.ndarray:
    """Índices de clase -> imagen RGB, para guardar visualizaciones."""
    return PALETA[np.asarray(mascara).astype(int).clip(0, NUM_CLASES - 1)]


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        ds = DFUTissueSeg(s)
        print(f"{s:6s}: {len(ds):3d} imágenes | prevalencia por clase: {prevalencia(ds)}")
    x, y = DFUTissueSeg("train", augment=True)[0]
    print("tensor imagen:", x.shape, x.dtype, "| máscara:", y.shape, y.dtype, "| clases en la máscara:", torch.unique(y).tolist())
