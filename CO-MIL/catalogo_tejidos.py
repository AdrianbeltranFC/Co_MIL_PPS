"""
=========================================================================================
CATÁLOGO DE TEJIDOS Y RESOLUCIÓN DE ETIQUETAS (Co-MIL)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (NO SE EJECUTA DIRECTAMENTE).

Fuente única de verdad para el catálogo de clases de tejido, compartida entre la
herramienta de anotación (generador_bolsas.py) y la ingesta de datos para
entrenamiento (dataset.py).

Por qué existe: el catálogo de clases puede crecer entre sesiones de anotación
(se agregan tejidos raros) o renombrarse (ajuste de terminología clínica). Las
bolsas ya guardadas conservan las etiquetas tal como se marcaron en su momento
(campo 'roi_labels'). Este módulo resuelve esas etiquetas crudas contra el
catálogo vigente en tiempo de USO (no de guardado), para que el vector Y de
cada bolsa se pueda reconstruir de forma consistente sin importar cuántas veces
haya cambiado el catálogo entre sesiones.
=========================================================================================
"""

import glob
import json
import os
import unicodedata
from typing import Dict, List, Optional

import torch

# Clases base, siempre presentes en este orden. Es seguro agregar clases nuevas
# al final (tanto aquí como en el catálogo vigente vía la GUI): a las bolsas ya
# guardadas simplemente les corresponde un 0 en la columna nueva. Lo que NO es
# seguro es reordenar o quitar clases existentes sin actualizar SINONIMOS_MANUALES.
CLASES_BASE: List[str] = [
    "Tejido de Granulación",
    "Fibrina (Esfacelo)",
    "Tejido Necrótico (Escara)",
    "Tejido Calloso (Hiperqueratosis)",
    "Exudado",
    "Epitelización",
    "Hueso Expuesto",
    "Tendón / Músculo Expuesto",
    "Piel Perilesional Sana",
    "Maceración / Eritema",
]

# Nombre crudo (normalizado) -> nombre canónico vigente. Cubre renombrados que
# ocurrieron ANTES de que existiera el registro automático de renombres (ver
# registrar_renombre). Los renombrados hechos desde la GUI a partir de ahora se
# registran solos en renombres_clases.json y no requieren tocar este diccionario.
SINONIMOS_MANUALES: Dict[str, str] = {
    "granulacion": "Tejido de Granulación",
}

NOMBRE_REGISTRO_CATALOGO = "custom_classes.json"
NOMBRE_REGISTRO_RENOMBRES = "renombres_clases.json"


def limpiar_texto(etiqueta: str) -> str:
    """Quita espacios redundantes al inicio/fin y entre palabras."""
    return " ".join(str(etiqueta).strip().split())


def normalizar_clave(etiqueta: str) -> str:
    """Normaliza una etiqueta ignorando mayúsculas y acentos, para comparar
    nombres de clase sin que un typo de acentuación cuente como clase distinta."""
    limpio = limpiar_texto(etiqueta)
    plano = unicodedata.normalize("NFKD", limpio).encode("ascii", "ignore").decode("ascii")
    return plano.casefold()


def deduplicar(etiquetas: List[str]) -> List[str]:
    vistas = set()
    resultado = []
    for etiqueta in etiquetas:
        limpio = limpiar_texto(etiqueta)
        if not limpio:
            continue
        clave = normalizar_clave(limpio)
        if clave in vistas:
            continue
        vistas.add(clave)
        resultado.append(limpio)
    return resultado


def cargar_catalogo_vigente(app_dir: str) -> List[str]:
    """Lee custom_classes.json (mismo formato que usa generador_bolsas.py) y
    devuelve el catálogo vigente completo (clases base + clases agregadas).
    Si el archivo no existe todavía, devuelve solo las clases base."""
    ruta = os.path.join(app_dir, NOMBRE_REGISTRO_CATALOGO)
    if not os.path.exists(ruta):
        return list(CLASES_BASE)
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return list(CLASES_BASE)

    if isinstance(data, dict) and isinstance(data.get("class_catalog"), list):
        etiquetas = data["class_catalog"]
    elif isinstance(data, dict):
        etiquetas = list(CLASES_BASE) + list(data.get("custom_classes", []))
    elif isinstance(data, list):
        etiquetas = data
    else:
        etiquetas = list(CLASES_BASE)

    return deduplicar(list(CLASES_BASE) + list(etiquetas))


def cargar_renombres(app_dir: str) -> Dict[str, str]:
    """Combina los sinónimos manuales (renombrados históricos, anteriores al
    registro automático) con renombres_clases.json (lo que la GUI va
    registrando sola cada vez que alguien usa 'Renombrar etiquetas del
    catálogo'). Las claves quedan normalizadas para comparar sin acentos."""
    combinados = dict(SINONIMOS_MANUALES)
    ruta = os.path.join(app_dir, NOMBRE_REGISTRO_RENOMBRES)
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                registrados = json.load(f)
            for nombre_anterior, nombre_nuevo in registrados.items():
                combinados[normalizar_clave(nombre_anterior)] = nombre_nuevo
        except Exception:
            pass
    return combinados


def registrar_renombre(app_dir: str, nombre_anterior: str, nombre_nuevo: str) -> None:
    """Se llama desde la GUI cada vez que se renombra una clase base, para que
    las bolsas ya guardadas con el nombre anterior se sigan resolviendo bien
    contra el catálogo vigente en el futuro."""
    ruta = os.path.join(app_dir, NOMBRE_REGISTRO_RENOMBRES)
    registrados: Dict[str, str] = {}
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                registrados = json.load(f)
        except Exception:
            registrados = {}
    registrados[nombre_anterior] = nombre_nuevo
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(registrados, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def resolver_nombre_canonico(
    etiqueta_cruda: str, catalogo_vigente: List[str], renombres: Dict[str, str]
) -> Optional[str]:
    """Traduce una etiqueta tal como fue guardada en una bolsa (en cualquier
    momento de la vida del catálogo) hacia su nombre canónico vigente.
    Devuelve None si la etiqueta no se reconoce (typo, basura, o clase que ya
    no existe en el catálogo actual)."""
    clave = normalizar_clave(etiqueta_cruda)
    for clase in catalogo_vigente:
        if normalizar_clave(clase) == clave:
            return clase
    if clave in renombres:
        nuevo = renombres[clave]
        for clase in catalogo_vigente:
            if normalizar_clave(clase) == normalizar_clave(nuevo):
                return clase
        return nuevo
    return None


def vectorizar_etiquetas(
    etiquetas_crudas: List[str],
    catalogo_vigente: List[str],
    renombres: Dict[str, str],
) -> torch.Tensor:
    """Construye el vector Y multietiqueta contra el catálogo vigente, a
    partir de las etiquetas crudas de una bolsa. Etiquetas no reconocidas se
    ignoran silenciosamente (mismo criterio que el resto del pipeline)."""
    vector = torch.zeros(len(catalogo_vigente), dtype=torch.float32)
    indice_por_clave = {normalizar_clave(c): i for i, c in enumerate(catalogo_vigente)}
    for cruda in etiquetas_crudas:
        resuelto = resolver_nombre_canonico(cruda, catalogo_vigente, renombres)
        if resuelto is None:
            continue
        idx = indice_por_clave.get(normalizar_clave(resuelto))
        if idx is not None:
            vector[idx] = 1.0
    return vector


def etiquetas_no_reconocidas(
    etiquetas_crudas: List[str], catalogo_vigente: List[str], renombres: Dict[str, str]
) -> List[str]:
    """Utilidad de diagnóstico: cuáles etiquetas crudas de una bolsa no se
    pudieron resolver contra el catálogo vigente (posible typo o basura)."""
    return [e for e in etiquetas_crudas if resolver_nombre_canonico(e, catalogo_vigente, renombres) is None]


def auditar_etiquetas_no_reconocidas(
    pt_folder: str, catalogo_vigente: List[str], renombres: Dict[str, str]
) -> Dict[str, List[str]]:
    """Escanea TODAS las bolsas .pt de una carpeta y devuelve, por archivo, las
    etiquetas crudas que no se pudieron resolver contra el catálogo vigente
    (diccionario vacío = todo bien).

    Por qué existe: antes de la Etapa 0 (2026-08-21), el catálogo de clases se
    fragmentaba en silencio -- una bolsa con una etiqueta cruda que ya no
    coincidía con el catálogo simplemente se ignoraba dentro de
    vectorizar_etiquetas(), y esa clase parecía tener menos ejemplos de los
    que en realidad tiene (o incluso 0). El fix de catalogo_tejidos.py resolvió
    el problema, pero como el catálogo puede seguir creciendo/renombrándose en
    sesiones futuras de anotación, esta auditoría debe correr SIEMPRE al
    inicio de entrenar_comil.py y evaluar_comil.py (no a mano, no solo cuando
    alguien sospecha algo raro) para que una fragmentación nueva se detecte
    de inmediato en vez de disfrazarse de "esta clase es rara"."""
    problemas: Dict[str, List[str]] = {}
    for ruta in glob.glob(os.path.join(pt_folder, "*.pt")):
        data = torch.load(ruta, map_location="cpu", weights_only=False)
        crudas = data.get("roi_labels")
        if not crudas:
            continue
        no_reconocidas = etiquetas_no_reconocidas(crudas, catalogo_vigente, renombres)
        if no_reconocidas:
            problemas[os.path.basename(ruta)] = no_reconocidas
    return problemas


def contar_positivos_por_clase(
    pt_folder: str, catalogo_vigente: List[str], renombres: Dict[str, str]
) -> Dict[str, int]:
    """Cuenta en cuántas bolsas de `pt_folder` aparece cada clase como
    positiva, usando la misma resolución dinámica de etiquetas que dataset.py.

    Por qué existe: un split (train/val/test) puede tener muy pocos positivos
    de una clase rara por pura variación estadística del reparto, y eso es
    indistinguible -- mirando solo ese split -- de una clase que sigue sin
    resolverse bien contra el catálogo. Esta función da el conteo sobre TODO
    el dataset, para que evaluar_comil.py pueda mostrar ambos números uno al
    lado del otro y la diferencia entre "es rara de verdad" y "hubo mala
    suerte en el split" quede visible sin tener que investigar a mano."""
    conteo = {c: 0 for c in catalogo_vigente}
    for ruta in glob.glob(os.path.join(pt_folder, "*.pt")):
        data = torch.load(ruta, map_location="cpu", weights_only=False)
        crudas = data.get("roi_labels")
        if not crudas:
            continue
        vector = vectorizar_etiquetas(crudas, catalogo_vigente, renombres)
        for i, clase in enumerate(catalogo_vigente):
            if vector[i] == 1:
                conteo[clase] += 1
    return conteo


def clave_imagen_desde_archivo(nombre_archivo: str) -> str:
    """A partir del nombre de un archivo de bolsa (ej. 'UPD_017__roi_03_bag.pt'
    o 'UPD_017_bag.pt'), devuelve la clave de la imagen original ('UPD_017').
    Es la unidad correcta para agrupar bolsas al armar particiones train/val/
    test: varias bolsas (ROIs) de la misma foto NUNCA deben quedar repartidas
    en conjuntos distintos, o se filtra información entre ellos."""
    base = os.path.basename(nombre_archivo)
    if base.endswith("_bag.pt"):
        base = base[: -len("_bag.pt")]
    return base.split("__roi_")[0]
