"""
=========================================================================================
REFINAMIENTO TIPO DENSECRF DE MAPAS DE ATENCIÓN (Co-MIL)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (NO SE EJECUTA DIRECTAMENTE).

Objetivo:
Afinar los mapas de atención gruesos (un valor por parche de 224px) hacia máscaras
alineadas con los bordes reales del tejido, usando el color/contraste de la imagen
original como guía — el mismo formalismo de campo medio (mean-field) de DenseCRF
(Krähenbühl & Koltun, 2011): potencial unario + término de suavidad (posición) +
término bilateral (posición + color), iterando la actualización de campo medio.

Nota de implementación — motor intercambiable:
La librería de referencia (pydensecrf) requiere compilar extensiones en C++/Cython.
Si está instalada (junto con un compilador de Visual Studio con el componente
"Desktop development with C++"), este módulo la usa automáticamente -- es más rápida
(látice permutoédrico) y es la implementación que cita la literatura de WSSS, así que
para las figuras finales del paper es la opción preferida. Si no está disponible (como
en este equipo, donde no hay compilador configurado), se usa una reimplementación en
NumPy puro del mismo algoritmo de campo medio (unario + suavidad + bilateral por
ventana deslizante vectorizada) -- más lenta, sin dependencias nativas, matemáticamente
equivalente. El resto del pipeline (visualizar_resultados.py, etc.) no necesita saber
cuál de las dos se está usando: siempre llama a refinar_mapa_atencion().
=========================================================================================
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

try:
    import pydensecrf.densecrf as _dcrf
    from pydensecrf.utils import unary_from_softmax as _unary_from_softmax
    MOTOR_DISPONIBLE = "pydensecrf"
except ImportError:
    _dcrf = None
    _unary_from_softmax = None
    MOTOR_DISPONIBLE = "numpy"


def _filtro_bilateral_conjunto(mapa: np.ndarray, guia_rgb: np.ndarray, radio: int,
                                sigma_espacial: float, sigma_color: float) -> np.ndarray:
    """
    Filtra `mapa` (H, W) usando `guia_rgb` (H, W, 3) en [0,1] como guía de color:
    cada píxel se recalcula como el promedio ponderado de su vecindario (radio
    `radio` px), con más peso para vecinos cercanos en posición Y parecidos en
    color en la imagen guía. Es el término bilateral de DenseCRF, vectorizado con
    ventana deslizante (sin bucles por píxel).
    """
    H, W = mapa.shape
    k = 2 * radio + 1

    mapa_pad = np.pad(mapa, radio, mode="reflect").astype(np.float32)
    guia_pad = np.pad(guia_rgb, ((radio, radio), (radio, radio), (0, 0)), mode="reflect").astype(np.float32)

    ventanas_mapa = sliding_window_view(mapa_pad, (k, k))                  # (H, W, k, k)
    ventanas_guia = sliding_window_view(guia_pad, (k, k, 3))[:, :, 0]      # (H, W, k, k, 3)

    centro_guia = guia_rgb.astype(np.float32)[:, :, None, None, :]        # (H, W, 1, 1, 3)
    dif_color2 = ((ventanas_guia - centro_guia) ** 2).sum(-1)             # (H, W, k, k)
    peso_color = np.exp(-dif_color2 / (2 * sigma_color ** 2))

    yy, xx = np.mgrid[-radio:radio + 1, -radio:radio + 1]
    peso_espacial = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_espacial ** 2)).astype(np.float32)

    pesos = peso_color * peso_espacial[None, None, :, :]
    pesos_norm = pesos / np.clip(pesos.sum(axis=(-1, -2), keepdims=True), 1e-8, None)

    return (ventanas_mapa * pesos_norm).sum(axis=(-1, -2))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _reducir_progresivo(imagen: np.ndarray, tam_objetivo: tuple) -> np.ndarray:
    """Reduce `imagen` (H,W) o (H,W,3) a `tam_objetivo` (ancho, alto) partiendo
    a la mitad en cada paso (INTER_AREA) hasta acercarse, y solo al final hace
    el ajuste fino exacto. Evita el aliasing periódico que deja un único
    downscale agresivo (factor >4-6x) sobre imágenes con estructura regular."""
    actual = imagen
    while actual.shape[1] > 2 * tam_objetivo[0] and actual.shape[0] > 2 * tam_objetivo[1]:
        nuevo_ancho, nuevo_alto = actual.shape[1] // 2, actual.shape[0] // 2
        actual = cv2.resize(actual, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    return cv2.resize(actual, tam_objetivo, interpolation=cv2.INTER_AREA)


def _refinar_mapa_atencion_pydensecrf(
    imagen_rgb: np.ndarray,
    mapa_atencion: np.ndarray,
    n_iter: int,
    peso_suavidad: float,
    peso_bilateral: float,
    sigma_espacial_suavidad: float,
    sigma_espacial_bilateral: float,
    sigma_color_bilateral: float,
) -> np.ndarray:
    """Mismo refinamiento usando la librería de referencia (rápida, corre a
    resolución completa sin necesidad de la aproximación por ventana)."""
    H, W = mapa_atencion.shape
    mapa = np.clip(mapa_atencion, 1e-4, 1 - 1e-4).astype(np.float32)
    probs = np.stack([1.0 - mapa, mapa], axis=0)  # (2, H, W): fondo, tejido

    d = _dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(_unary_from_softmax(probs))
    d.addPairwiseGaussian(sxy=sigma_espacial_suavidad, compat=peso_suavidad)
    imagen_uint8 = np.ascontiguousarray((np.clip(imagen_rgb, 0, 1) * 255).astype(np.uint8))
    d.addPairwiseBilateral(
        sxy=sigma_espacial_bilateral,
        srgb=sigma_color_bilateral * 255,
        rgbim=imagen_uint8,
        compat=peso_bilateral,
    )
    Q = np.array(d.inference(n_iter)).reshape((2, H, W))
    return Q[1]


def refinar_mapa_atencion(
    imagen_rgb: np.ndarray,
    mapa_atencion: np.ndarray,
    n_iter: int = 15,
    peso_suavidad: float = 2.0,
    peso_bilateral: float = 4.0,
    sigma_espacial_suavidad: float = 3.0,
    sigma_espacial_bilateral: float = 4.0,
    sigma_color_bilateral: float = 0.12,
    radio_bilateral: int = 12,
    resolucion_trabajo: int = 220,
    motor: str = "auto",
) -> np.ndarray:
    """
    Refina un mapa de atención grueso (valores en [0,1], mismo tamaño en píxeles
    que `imagen_rgb`, típicamente obtenido por upsample "nearest" desde la malla
    de parches) para que sus bordes se alineen con los bordes reales del tejido.

    Arguments:
        imagen_rgb: array (H, W, 3) en [0,1] — la reconstrucción de la bolsa.
        mapa_atencion: array (H, W) en [0,1] — el heatmap ya escalado a esta
            resolución (ej. con cv2.resize + INTER_NEAREST, igual que hoy).
        n_iter: iteraciones de actualización de campo medio.
        peso_suavidad / peso_bilateral: peso relativo de cada término del kernel.
        sigma_*: alcance de cada kernel (posición y color). Con motor="numpy",
            sigma_espacial_bilateral debe ser consistente con radio_bilateral
            (ver más abajo); con pydensecrf no aplica esa restricción.
        radio_bilateral: solo usado por el motor NumPy -- radio de la ventana
            del filtro bilateral, en píxeles de trabajo.
        resolucion_trabajo: solo usado por el motor NumPy -- lado mayor, en
            píxeles, al que se reescala para calcular el refinamiento (por
            velocidad). pydensecrf corre a la resolución original completa.
        motor: "auto" (usa pydensecrf si está instalado, si no NumPy),
            "pydensecrf" o "numpy" para forzar uno en particular.

    Returns:
        Mapa refinado (H, W) en [0,1], mismo tamaño que `imagen_rgb`.
    """
    if motor == "auto":
        motor = MOTOR_DISPONIBLE
    if motor == "pydensecrf":
        if _dcrf is None:
            raise ImportError(
                "motor='pydensecrf' pero la librería no está instalada en este entorno "
                "(pip install pydensecrf -- requiere un compilador de C++). Usa motor='numpy' "
                "o 'auto'."
            )
        return _refinar_mapa_atencion_pydensecrf(
            imagen_rgb, mapa_atencion, n_iter, peso_suavidad, peso_bilateral,
            sigma_espacial_suavidad, sigma_espacial_bilateral, sigma_color_bilateral,
        )

    H, W = mapa_atencion.shape
    escala = resolucion_trabajo / max(H, W)
    escala = min(escala, 1.0)  # nunca subir la resolución, solo bajarla

    if radio_bilateral < 2.5 * sigma_espacial_bilateral:
        # El filtro bilateral trunca la campana gaussiana espacial en
        # +-radio_bilateral. Si el radio es mucho más chico que el sigma
        # pedido, la ventana termina comportándose como una caja rígida (peso
        # casi plano dentro de la ventana) en vez de una campana suave, y el
        # resultado queda con artefactos en bloques/rejilla en vez de suave.
        # Regla práctica: radio >= ~3*sigma para capturar la campana completa.
        raise ValueError(
            f"radio_bilateral={radio_bilateral} es demasiado chico para "
            f"sigma_espacial_bilateral={sigma_espacial_bilateral} (usa al menos "
            f"~{2.5 * sigma_espacial_bilateral:.0f}), o baja el sigma."
        )

    if escala < 1.0:
        tam_trab = (max(1, int(round(W * escala))), max(1, int(round(H * escala))))
        # Reducción progresiva (a la mitad en cada paso, tipo pirámide de
        # imagen) en vez de un solo salto grande -- práctica estándar para
        # evitar aliasing cuando el factor de reducción es grande (>4-6x).
        img_trab = _reducir_progresivo(imagen_rgb.astype(np.float32), tam_trab)
        mapa_trab = _reducir_progresivo(mapa_atencion.astype(np.float32), tam_trab)
    else:
        img_trab = imagen_rgb
        mapa_trab = mapa_atencion

    mapa_trab = np.clip(mapa_trab, 1e-4, 1 - 1e-4).astype(np.float32)
    logit_unario = np.log(mapa_trab / (1 - mapa_trab))

    # Amortiguamiento: mezclar el Q propuesto en cada paso con el Q anterior en
    # vez de reemplazarlo de golpe. Sin esto, con pesos de kernel moderados/altos
    # la iteración overshootea en los bordes de cada parche, y el error se
    # retroalimenta iteración tras iteración hasta volverse un patrón fractal en
    # vez de converger a un mapa suave (se observó justo eso al probar con más
    # iteraciones: el "ruido" en los bordes crecía en vez de desaparecer).
    amortiguamiento = 0.5
    Q = mapa_trab.copy()
    for _ in range(n_iter):
        msg_suavidad = gaussian_filter(Q, sigma=sigma_espacial_suavidad) - Q
        msg_bilateral = _filtro_bilateral_conjunto(
            Q, img_trab, radio_bilateral, sigma_espacial_bilateral, sigma_color_bilateral
        ) - Q
        Q_propuesto = _sigmoid(logit_unario + peso_suavidad * msg_suavidad + peso_bilateral * msg_bilateral)
        Q = amortiguamiento * Q_propuesto + (1 - amortiguamiento) * Q

    if escala < 1.0:
        Q = cv2.resize(Q.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        Q = np.clip(Q, 0.0, 1.0)

    return Q
