"""
=========================================================================================
SEGUIMIENTO DE EXPERIMENTOS (Co-MIL)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (NO SE EJECUTA DIRECTAMENTE).

Objetivo:
Cada corrida de entrenamiento (entrenar_comil.py) queda en su propia carpeta con fecha,
junto con un metadata.json legible que responde sin ambigüedad: ¿de cuándo es este
modelo?, ¿con qué partición de datos se entrenó?, ¿qué resolución de parche usa?,
¿cuántas bolsas y de qué split? evaluar_comil.py y visualizar_resultados.py escriben
sus resultados DENTRO de esa misma carpeta, para que nunca queden mezclados resultados
de corridas distintas en el mismo lugar (justo el problema que motivó este módulo: había
carpetas de heatmaps de un modelo viejo y uno nuevo sin nada que las distinguiera).
=========================================================================================
"""

import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

NOMBRE_METADATA = "metadata.json"
NOMBRE_PUNTERO_ULTIMO = "ultimo_experimento.txt"
PREFIJO_EXPERIMENTO = "exp_"


def resumen_patch_size(pt_folder: str) -> Dict[str, object]:
    """Escanea una carpeta de bolsas .pt y reporta qué tamaño(s) de parche están en
    uso. En un dataset sano debería haber un único valor; si aparece más de uno, es
    una señal real de que se están mezclando resoluciones distintas (por ejemplo, si
    algún día se usa reprocesador_dataset.py para generar parches más finos junto a
    los de 224px en la misma carpeta)."""
    import torch  # import perezoso: este módulo no debe exigir torch solo para leer metadata.json

    archivos = glob.glob(os.path.join(pt_folder, "*.pt"))
    tamanos = set()
    n_parches = []
    for f in archivos:
        data = torch.load(f, map_location="cpu", weights_only=False)
        meta = data.get("spatial_metadata", {}) or {}
        tamanos.add(meta.get("patch_size"))
        n_parches.append(data["X"].shape[0])

    return {
        "total_bolsas": len(archivos),
        "patch_sizes_encontrados": sorted(t for t in tamanos if t is not None),
        "mezcla_de_resoluciones": len(tamanos) > 1,
        "parches_por_bolsa_min": min(n_parches) if n_parches else None,
        "parches_por_bolsa_max": max(n_parches) if n_parches else None,
    }


def crear_carpeta_experimento(raiz_pesos: str, etiqueta: Optional[str] = None) -> str:
    """Crea (y devuelve la ruta a) una carpeta nueva para esta corrida de
    entrenamiento, nombrada con fecha y hora para que el orden cronológico sea
    obvio con solo mirar el listado de carpetas. `etiqueta` es un sufijo corto
    opcional y legible (ej. "split-corregido") para no depender solo de la hora."""
    ahora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"{PREFIJO_EXPERIMENTO}{ahora}"
    if etiqueta:
        nombre += f"_{etiqueta}"
    ruta = os.path.join(raiz_pesos, nombre)
    os.makedirs(ruta, exist_ok=True)
    return ruta


def guardar_metadata(ruta_experimento: str, metadata: Dict[str, object]) -> None:
    metadata = dict(metadata)
    metadata.setdefault("guardado_en", datetime.now().isoformat())
    with open(os.path.join(ruta_experimento, NOMBRE_METADATA), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def cargar_metadata(ruta_experimento: str) -> Dict[str, object]:
    ruta = os.path.join(ruta_experimento, NOMBRE_METADATA)
    if not os.path.exists(ruta):
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def marcar_como_mas_reciente(raiz_pesos: str, ruta_experimento: str) -> None:
    """Escribe un puntero de texto plano al experimento más reciente, para que
    evaluar_comil.py / visualizar_resultados.py puedan usarlo por default sin que el
    usuario tenga que pasar la ruta a mano cada vez (pero siempre pueden apuntar a
    una carpeta de experimento distinta explícitamente para reproducir resultados
    viejos)."""
    with open(os.path.join(raiz_pesos, NOMBRE_PUNTERO_ULTIMO), "w", encoding="utf-8") as f:
        f.write(os.path.basename(ruta_experimento))


def experimento_mas_reciente(raiz_pesos: str) -> Optional[str]:
    """Devuelve la ruta al experimento más reciente: primero intenta el puntero
    explícito (más confiable), y si no existe, cae a ordenar las carpetas exp_*
    por nombre (que ya son cronológicas por construcción)."""
    puntero = os.path.join(raiz_pesos, NOMBRE_PUNTERO_ULTIMO)
    if os.path.exists(puntero):
        with open(puntero, "r", encoding="utf-8") as f:
            nombre = f.read().strip()
        ruta = os.path.join(raiz_pesos, nombre)
        if os.path.isdir(ruta):
            return ruta

    candidatos = sorted(
        d for d in glob.glob(os.path.join(raiz_pesos, f"{PREFIJO_EXPERIMENTO}*")) if os.path.isdir(d)
    )
    return candidatos[-1] if candidatos else None


def listar_experimentos(raiz_pesos: str) -> List[str]:
    return sorted(
        d for d in glob.glob(os.path.join(raiz_pesos, f"{PREFIJO_EXPERIMENTO}*")) if os.path.isdir(d)
    )
