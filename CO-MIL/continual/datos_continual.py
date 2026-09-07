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
# Segundo dominio REAL: el lote mexicano con anotación débil (bolsas MIL)
# ---------------------------------------------------------------------------
# En el vector Y de 12 clases del catálogo mexicano: 0 = Granulación, 1 = Fibrina,
# 3 = Tejido Calloso.  Se reordena a [Fibrina, Granulación, Callo] para casar con
# CLASES_TEJIDO de este módulo.
_MEX_A_DFU = [1, 0, 3]
RUTA_CACHE_MEX = os.path.join(DIR_CACHE, "features_mexicano.pt")


def _reconstruir_roi(bolsa_X, grid_shape, patch_size):
    gh, gw = grid_shape
    lienzo = torch.zeros(3, gh * patch_size, gw * patch_size, dtype=bolsa_X.dtype)
    k = 0
    for r in range(gh):
        for c in range(gw):
            if k < bolsa_X.shape[0]:
                lienzo[:, r * patch_size:(r + 1) * patch_size,
                       c * patch_size:(c + 1) * patch_size] = bolsa_X[k]
                k += 1
    return lienzo


def cachear_features_mexicano(dispositivo=None, forzar=False, colornorm=False) -> str:
    """Cada ROI del lote experto (bolsa MIL, parches 224 px) se reensambla, se
    reescala a 256 y se pasa por el MISMO extractor congelado -> bolsa [64,1280],
    igual representación que DFUTissue. Etiqueta débil de 3 clases derivada de la
    etiqueta de imagen que puso el clínico.
    `colornorm`: primero desplaza media y desv. por canal de cada foto a las de
    DFUTissue (transferencia de color global) -- para separar el efecto cámara/color."""
    import glob
    ruta_out = RUTA_CACHE_MEX.replace(".pt", "_norm.pt") if colornorm else RUTA_CACHE_MEX
    if os.path.exists(ruta_out) and not forzar:
        return ruta_out
    dispositivo = dispositivo or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DIR_CACHE, exist_ok=True)
    extractor = _extractor_congelado(dispositivo)
    ref_m, ref_s = _stats_color_dfu() if colornorm else (None, None)
    patron = os.path.join(RAIZ_REPO, "APP_generador_bolsas", "**",
                          "Bolsas_MIL_Procesadas", "224px", "*__roi_*.pt")
    archivos = sorted(glob.glob(patron, recursive=True))

    from PIL import Image as _Im
    bolsas, etiquetas, nombres = [], [], []
    for f in archivos:
        d = torch.load(f, weights_only=False)
        meta = d["spatial_metadata"]
        img = _reconstruir_roi(d["X"], meta["grid_shape"], meta["patch_size"]).clamp(0, 1)
        arr = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        arr = np.asarray(_Im.fromarray(arr).resize((256, 256), _Im.BILINEAR))
        xf = arr.astype(np.float32) / 255.0
        if colornorm:
            m, sdv = xf.reshape(-1, 3).mean(0), xf.reshape(-1, 3).std(0) + 1e-6
            xf = np.clip((xf - m) / sdv * ref_s + ref_m, 0, 1).astype(np.float32)
        x = (xf - _MEAN) / _STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        with torch.no_grad():
            fmap = extractor(x.unsqueeze(0).to(dispositivo))
        bolsas.append(fmap.squeeze(0).flatten(1).t().contiguous().cpu())
        y12 = d["Y"].float()
        etiquetas.append(torch.tensor([float(y12[i]) for i in _MEX_A_DFU]))
        nombres.append(os.path.basename(f).replace("_bag.pt", ""))

    datos = {"nombres": nombres, "X": torch.stack(bolsas), "Y": torch.stack(etiquetas)}
    torch.save(datos, ruta_out)
    print(f"[cache-mex{'-norm' if colornorm else ''}] {len(nombres)} ROIs · "
          f"X={tuple(datos['X'].shape)} · positivos/clase (Fib,Gra,Cal)="
          f"{datos['Y'].sum(0).tolist()} -> {os.path.relpath(ruta_out, RAIZ_REPO)}")
    return ruta_out


def cargar_mexicano(colornorm=False):
    ruta = RUTA_CACHE_MEX.replace(".pt", "_norm.pt") if colornorm else RUTA_CACHE_MEX
    if not os.path.exists(ruta):
        cachear_features_mexicano(colornorm=colornorm)
    d = torch.load(ruta)
    return d["nombres"], d["X"], d["Y"]


# ---------------------------------------------------------------------------
# Idea del autor: ¿el salto de dominio es por cámara/color (arreglable con un
# filtro) o por la población en sí?  Dos formas de probarlo:
#   colornorm  -> normalizar el color de las fotos mexicanas al de DFUTissue.
#   aug        -> entrenar DFUTissue con aumentación fuerte de cámara/color.
# ---------------------------------------------------------------------------
def _stats_color_dfu():
    """Media y desv. por canal (RGB, escala 0-1) sobre las imágenes crudas de DFUTissue."""
    ms, ss, n = np.zeros(3), np.zeros(3), 0
    for split in ("train", "val", "test"):
        ds = DFUTissueSeg(split, augment=False)
        for i in range(len(ds)):
            img, _ = ds._cargar(i)
            x = img.astype(np.float64) / 255.0
            ms += x.reshape(-1, 3).mean(0); ss += x.reshape(-1, 3).std(0); n += 1
    return ms / n, ss / n


def _photometrico_aleatorio(img_u8, rng):
    """Transformación fotométrica fuerte y aleatoria (balance de blancos, gamma,
    brillo, desenfoque, JPEG). Para aumentación de entrenamiento."""
    import io as _io

    from PIL import Image as _Im
    x = img_u8.astype(np.float32)
    x = x * (0.8 + 0.4 * rng.random(3)).astype(np.float32)          # balance de blancos
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** (0.7 + 0.7 * rng.random())   # gamma
    x = x * (0.75 + 0.5 * rng.random())                             # brillo
    x = np.clip(x + rng.normal(0, 4 + 6 * rng.random(), x.shape), 0, 255).astype(np.uint8)
    im = _Im.fromarray(x)
    if rng.random() < 0.6:
        im = im.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"])
                       .GaussianBlur(0.4 + 1.4 * rng.random()))
    buf = _io.BytesIO(); im.save(buf, format="JPEG", quality=int(25 + 65 * rng.random()))
    return np.asarray(_Im.open(buf).convert("RGB"), dtype=np.uint8)


def cachear_features_dfu_aug(n_copias=4, dispositivo=None, semilla=0, forzar=False) -> str:
    """DFUTissue con `n_copias` versiones fotométricas aleatorias por imagen
    (aumentación de cámara/color). Comparte etiquetas con el original."""
    ruta = os.path.join(DIR_CACHE, f"features_dfutissue_aug{n_copias}.pt")
    if os.path.exists(ruta) and not forzar:
        return ruta
    dispositivo = dispositivo or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DIR_CACHE, exist_ok=True)
    extractor = _extractor_congelado(dispositivo)
    rng = np.random.default_rng(semilla)
    bolsas, etiquetas = [], []
    for split in ("train", "val", "test"):
        ds = DFUTissueSeg(split, augment=False)
        for i in range(len(ds)):
            img, ann = ds._cargar(i)
            pres = torch.zeros(NUM_CLASES)
            for c in (1, 2, 3):
                if int((ann == c).sum()) >= MIN_PIXELES:
                    pres[c - 1] = 1.0
            for _ in range(n_copias):
                aug = _photometrico_aleatorio(img, rng)
                x = (aug.astype(np.float32) / 255.0 - _MEAN) / _STD
                x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
                with torch.no_grad():
                    fmap = extractor(x.unsqueeze(0).to(dispositivo))
                bolsas.append(fmap.squeeze(0).flatten(1).t().contiguous().cpu())
                etiquetas.append(pres.clone())
    torch.save({"X": torch.stack(bolsas), "Y": torch.stack(etiquetas)}, ruta)
    print(f"[cache-aug] {len(bolsas)} bolsas ({n_copias}x) -> {os.path.relpath(ruta, RAIZ_REPO)}")
    return ruta


def cargar_dfu_aug(n_copias=4):
    ruta = os.path.join(DIR_CACHE, f"features_dfutissue_aug{n_copias}.pt")
    if not os.path.exists(ruta):
        cachear_features_dfu_aug(n_copias)
    d = torch.load(ruta)
    return d["X"], d["Y"]


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


def split_global(n: int, frac_test: float = 0.25, semilla: int = 0):
    """Partición train/test única sobre todas las imágenes (para el modo class-incremental,
    donde no hay dominios: lo que se incrementa son las clases, no la distribución)."""
    rng = np.random.default_rng(semilla)
    idx = rng.permutation(n)
    n_te = max(6, int(round(n * frac_test)))
    return idx[n_te:].copy(), idx[:n_te].copy()


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
