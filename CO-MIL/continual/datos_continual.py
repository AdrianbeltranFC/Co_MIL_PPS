"""
=========================================================================================
DATOS PARA EL PROTOTIPO DE APRENDIZAJE CONTINUO  (Domain-Incremental sobre DFUTissue)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (lo usa correr_continual.py).

Objetivo 3 del plan de tesis ("la Co de Co-MIL"). Escenario Domain-Incremental
(van de Ven et al., Nat Mach Intell 2022): el espacio de etiquetas es fijo
{Fibrina, Granulación, Callo} presente/ausente por imagen; entre "lotes" cambia la
distribución de entrada (cámara / sitio / anotador); la cabeza es compartida.

SIMULACIÓN DE LOTES. Sin dataset propio con procedencia real todavía, los T lotes se
simulan sobre DFUTissue. Se probaron dos vías (notas de investigación 2-sep):
  - k-means sobre características congeladas -> en DFUTissue NO funciona: las
    imágenes son demasiado homogéneas (silueta ~0.04, un cluster con 1 imagen).
  - **shift fotométrico fijo por "sitio"** (balance de blancos, brillo, gamma,
    ruido de sensor) -> covariate shift controlado y lotes balanceados. Es la vía
    que usa el prototipo. Cada imagen se asigna a un lote al azar (reparto
    uniforme) y se le aplica la transformación de ese lote ANTES del extractor.
Cuando llegue el dataset propio, se sustituye por los lotes de procedencia real
(especialista / Adrián / enfermería) y el resto del pipeline no cambia.

Las características de los parches se **precalculan una sola vez** con MobileNetV2
congelado (rejilla 8x8 = 64 instancias por imagen) y se cachean; el estudio continuo
corre luego en CPU en segundos.
=========================================================================================
"""

import os
import sys

import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SEG = os.path.join(os.path.dirname(_DIR), "segmentacion")
for _p in (_DIR, _SEG):
    if _p not in sys.path:
        sys.path.append(_p)

from dataset_seg import DFUTissueSeg  # noqa: E402

RAIZ_REPO = os.path.dirname(os.path.dirname(_DIR))
DIR_CACHE = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "continual")

CLASES_TEJIDO = ["Fibrina", "Granulación", "Callo"]   # índices 1, 2, 3 en la máscara
NUM_CLASES = len(CLASES_TEJIDO)
MIN_PIXELES = 64
DIM_FEATURES = 1280
N_INSTANCIAS = 64            # rejilla nativa 8x8 de MobileNetV2 sobre 256 px

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------------------
# Transformaciones fotométricas por "sitio"  (sobre imagen uint8 RGB)
# ---------------------------------------------------------------------------
def _sitio_0(img, rng):
    return img                                              # cámara de referencia


def _sitio_1(img, rng):
    """Cámara distinta: balance cálido, más claro, compresión JPEG fuerte."""
    x = img.astype(np.float32)
    x = x * np.array([1.15, 1.03, 0.88], np.float32)
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** 0.82
    return _jpeg(np.clip(x, 0, 255).astype(np.uint8), calidad=30)


def _sitio_2(img, rng):
    """Teléfono peor: balance frío, oscuro, DESENFOCADO y con menos resolución
    efectiva (reescalado) + ruido de sensor. Es el shift que de verdad borra la
    textura fina que distingue fibrina/callo."""
    from PIL import Image as _Im
    x = img.astype(np.float32)
    x = x * np.array([0.88, 0.98, 1.16], np.float32)
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** 1.20
    x = np.clip(x + rng.normal(0, 9.0, size=x.shape), 0, 255).astype(np.uint8)
    h, w = x.shape[:2]
    im = _Im.fromarray(x).resize((w // 3, h // 3), _Im.BILINEAR).resize((w, h), _Im.BILINEAR)
    im = im.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(1.4))
    return np.asarray(im, dtype=np.uint8)


def _jpeg(arr, calidad=30):
    import io as _io

    from PIL import Image as _Im
    buf = _io.BytesIO()
    _Im.fromarray(arr).save(buf, format="JPEG", quality=calidad)
    return np.asarray(_Im.open(buf).convert("RGB"), dtype=np.uint8)


TRANSFORMS_SITIO = [_sitio_0, _sitio_1, _sitio_2]


def _ruta_cache(T: int) -> str:
    return os.path.join(DIR_CACHE, f"features_dfutissue_fotometrico_T{T}.pt")


# ---------------------------------------------------------------------------
# Caché de características (una sola vez por T)
# ---------------------------------------------------------------------------
def _extractor_congelado(dispositivo):
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    m = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features.to(dispositivo).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def cachear_features(T: int = 3, dispositivo=None, semilla: int = 0, forzar=False) -> str:
    """Asigna cada imagen de DFUTissue a uno de T sitios (reparto uniforme, semilla
    fija), le aplica la transformación fotométrica de ese sitio, la pasa por
    MobileNetV2 congelado y guarda: bolsa [64,1280], etiqueta [3] y sitio. Idempotente."""
    ruta = _ruta_cache(T)
    if os.path.exists(ruta) and not forzar:
        return ruta
    assert T <= len(TRANSFORMS_SITIO), f"solo hay {len(TRANSFORMS_SITIO)} sitios definidos"
    dispositivo = dispositivo or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DIR_CACHE, exist_ok=True)
    extractor = _extractor_congelado(dispositivo)
    rng = np.random.default_rng(semilla)

    # cargar todas las imágenes (crudas) y etiquetas
    crudas, anns, nombres = [], [], []
    for split in ("train", "val", "test"):
        ds = DFUTissueSeg(split, augment=False)
        for i in range(len(ds)):
            img, ann = ds._cargar(i)
            crudas.append(img); anns.append(ann); nombres.append(ds.nombres[i])

    n = len(crudas)
    sitio = np.array([i % T for i in range(n)])
    sitio = rng.permutation(sitio)                          # reparto uniforme, aleatorio

    bolsas, etiquetas = [], []
    for img, ann, s in zip(crudas, anns, sitio):
        img_t = TRANSFORMS_SITIO[s](img, rng)
        x = (img_t.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        with torch.no_grad():
            fmap = extractor(x.unsqueeze(0).to(dispositivo))        # [1,1280,8,8]
        bolsas.append(fmap.squeeze(0).flatten(1).t().contiguous().cpu())   # [64,1280]
        pres = torch.zeros(NUM_CLASES)
        for c in (1, 2, 3):
            if int((ann == c).sum()) >= MIN_PIXELES:
                pres[c - 1] = 1.0
        etiquetas.append(pres)

    datos = {"nombres": nombres, "X": torch.stack(bolsas), "Y": torch.stack(etiquetas),
             "sitio": torch.tensor(sitio), "T": T}
    torch.save(datos, ruta)
    print(f"[cache] {n} bolsas · T={T} sitios · X={tuple(datos['X'].shape)} · "
          f"positivos/clase={datos['Y'].sum(0).tolist()} -> {os.path.relpath(ruta, RAIZ_REPO)}")
    return ruta


def cargar(T: int = 3):
    ruta = _ruta_cache(T)
    if not os.path.exists(ruta):
        cachear_features(T)
    d = torch.load(_ruta_cache(T))
    return d["nombres"], d["X"], d["Y"], d["sitio"].numpy()


# ---------------------------------------------------------------------------
# Particiones y diagnóstico
# ---------------------------------------------------------------------------
def resumen_dominios(sitio: np.ndarray, Y: torch.Tensor) -> str:
    Y = Y.numpy()
    filas = [f"{'sitio':>6} {'n':>4} | " + " ".join(f"{c:>12}" for c in CLASES_TEJIDO)]
    for d in sorted(set(sitio.tolist())):
        m = sitio == d
        filas.append(f"{d:>6} {int(m.sum()):>4} | "
                     + " ".join(f"{p:>12.2f}" for p in Y[m].mean(0)))
    return "\n".join(filas)


def split_por_dominio(sitio: np.ndarray, frac_test: float = 0.25, semilla: int = 0) -> dict:
    """Por sitio: partición train/test reproducible (índices globales sobre las bolsas)."""
    rng = np.random.default_rng(semilla)
    salida = {}
    for d in sorted(set(sitio.tolist())):
        idx = np.where(sitio == d)[0]
        rng.shuffle(idx)
        n_te = max(3, int(round(len(idx) * frac_test)))
        salida[int(d)] = (idx[n_te:].copy(), idx[:n_te].copy())
    return salida


# alias de compatibilidad (el runner los importa)
def asignar_dominios(*a, **k):   # el sitio ya viene del caché
    raise RuntimeError("El sitio se asigna en cachear_features(); usa cargar(T).")
