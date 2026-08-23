"""
=========================================================================================
PARTICIÓN REPRODUCIBLE DEL DATASET (Co-MIL)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (PREPARACIÓN DE DATOS).
CÓMO EJECUTAR: python CO-MIL/particionar_dataset.py --bolsas "ruta/a/224px"

Objetivo:
Arma (o actualiza) una partición train/val/test a nivel de IMAGEN sobre las bolsas ya
generadas. Nunca a nivel de bolsa/ROI: varias bolsas de la misma foto son casi idénticas
entre sí, y repartirlas en conjuntos distintos filtraría información (fuga de datos).

Diseño adaptativo:
Pensado para poder correrse hoy con un subconjunto de imágenes anotadas, y de nuevo más
adelante con más imágenes, sin invalidar comparaciones anteriores. Las imágenes que ya
tienen partición asignada la CONSERVAN; solo se reparten las imágenes nuevas, buscando
mantener las proporciones objetivo (train/val/test). El resultado se guarda en un
manifiesto versionado (splits_manifest.json) que cualquier script de entrenamiento o
evaluación, o un revisor externo, puede leer directamente.

También asigna un número de "fold" (0..N-1) a cada imagen nueva, útil para una eventual
validación cruzada a nivel imagen sin tener que rehacer la partición.
=========================================================================================
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Set

_DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(_DIRECTORIO_ACTUAL)

import torch
import catalogo_tejidos


def agrupar_por_imagen(bolsas_dir: str) -> Dict[str, List[str]]:
    archivos = glob.glob(os.path.join(bolsas_dir, "*.pt"))
    grupos: Dict[str, List[str]] = defaultdict(list)
    for f in archivos:
        clave = catalogo_tejidos.clave_imagen_desde_archivo(f)
        grupos[clave].append(f)
    return grupos


def clases_por_imagen(
    grupos: Dict[str, List[str]], catalogo_vigente: List[str], renombres: Dict[str, str]
) -> Dict[str, Set[str]]:
    resultado: Dict[str, Set[str]] = {}
    for imagen, archivos in grupos.items():
        clases: Set[str] = set()
        for f in archivos:
            data = torch.load(f, map_location="cpu", weights_only=False)
            crudas = data.get("roi_labels", [])
            for cruda in crudas:
                resuelto = catalogo_tejidos.resolver_nombre_canonico(cruda, catalogo_vigente, renombres)
                if resuelto:
                    clases.add(resuelto)
        resultado[imagen] = clases
    return resultado


def cargar_manifiesto(ruta: str) -> dict:
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "seed": None, "proporciones": {}, "n_folds": 1, "imagenes": {}}


def guardar_manifiesto(ruta: str, manifiesto: dict) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)


def construir_particion(
    imagenes_clases: Dict[str, Set[str]],
    manifiesto: dict,
    proporciones: Dict[str, float],
    n_folds: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)

    ya_asignadas: Dict[str, dict] = manifiesto.get("imagenes", {})
    nuevas = [img for img in imagenes_clases if img not in ya_asignadas]

    # Frecuencia global de cada clase (sobre todas las imagenes conocidas hoy),
    # usada para priorizar la asignacion de imagenes con clases raras primero
    # (mismo espiritu que una estratificacion iterativa multietiqueta).
    frecuencia: Dict[str, int] = defaultdict(int)
    for clases in imagenes_clases.values():
        for c in clases:
            frecuencia[c] += 1

    def rareza(imagen: str) -> float:
        clases = imagenes_clases[imagen]
        if not clases:
            return 0.0
        return sum(1.0 / frecuencia[c] for c in clases)

    nuevas.sort(key=rareza, reverse=True)

    conteos: Dict[str, int] = defaultdict(int)
    for info in ya_asignadas.values():
        conteos[info["split"]] += 1
    total_previo = sum(conteos.values())

    splits = list(proporciones.keys())
    resultado_nuevas: Dict[str, dict] = {}

    for i, imagen in enumerate(nuevas):
        total_actual = total_previo + i + 1

        def deficit(s: str) -> float:
            # conteos[s] ya incluye tanto las imagenes previas del manifiesto
            # como las nuevas asignadas en iteraciones anteriores de este
            # mismo bucle (se actualiza al final de cada vuelta) -- no hay
            # que volver a contarlas via resultado_nuevas o se duplican.
            objetivo = proporciones[s] * total_actual
            return objetivo - conteos[s]

        elegido = max(splits, key=deficit)
        fold = rng.randrange(n_folds) if n_folds > 1 else 0
        resultado_nuevas[imagen] = {
            "split": elegido,
            "fold": fold,
            "clases": sorted(imagenes_clases[imagen]),
        }
        conteos[elegido] += 1

    manifiesto["imagenes"] = {**ya_asignadas, **resultado_nuevas}
    manifiesto["seed"] = seed
    manifiesto["proporciones"] = proporciones
    manifiesto["n_folds"] = n_folds
    manifiesto["total_imagenes"] = len(manifiesto["imagenes"])
    return manifiesto, resultado_nuevas


def garantizar_cobertura_minima(manifiesto: dict, splits_evaluables=("val", "test")) -> List[tuple]:
    """
    Pasada de reparación posterior al reparto por proporciones.

    El reparto proporcional (arriba) optimiza el TOTAL de imágenes por split, pero con
    tan pocas imágenes por clase eso no garantiza que cada clase quede representada en
    val/test: una clase con pocos ejemplos puede caer entera en 'train' por azar. Eso
    hace que sensibilidad/especificidad/AUC-ROC de esa clase salgan "N/D" en la
    evaluación aunque sí exista anotada.

    Esta función corrige eso: para cada clase con al menos 2 imágenes distintas en todo
    el dataset conocido, garantiza que 'val' y 'test' tengan al menos una, moviendo una
    imagen desde el split con más ejemplares de esa clase (normalmente 'train', y solo
    si eso no lo deja en cero). Clases con una sola imagen en todo el dataset no se
    pueden repartir (no hay de dónde tomar prestado) y quedan documentadas tal cual.

    Importante: una misma imagen suele contener varias clases (varias ROIs con tejidos
    distintos), así que el mismo movimiento puede resolver la cobertura de más de una
    clase a la vez, y una imagen ya movida para la clase A no debe volver a moverse a un
    split distinto al resolver la clase B (se perdería el arreglo de A). Por eso se
    mantiene un único estado vivo (`estado`) que se consulta y actualiza en cada paso,
    en vez de recalcular por-clase con datos que quedan desactualizados a medio proceso.

    Devuelve la lista de movimientos aplicados, para que quede auditable.
    """
    imagenes = manifiesto["imagenes"]
    todos_los_splits = ("train", "val", "test")

    estado: Dict[str, str] = {img: info["split"] for img, info in imagenes.items()}
    clases_de: Dict[str, List[str]] = {img: info.get("clases", []) for img, info in imagenes.items()}

    imagenes_por_clase: Dict[str, List[str]] = defaultdict(list)
    for img, clases in clases_de.items():
        for clase in clases:
            imagenes_por_clase[clase].append(img)

    movimientos = []

    for clase, imgs in imagenes_por_clase.items():
        if len(imgs) < 2:
            continue  # no hay de donde tomar prestado

        for split_objetivo in splits_evaluables:
            if any(estado[img] == split_objetivo for img in imgs):
                continue  # ya tiene al menos una imagen con esta clase (estado vivo)

            conteo_por_split: Dict[str, List[str]] = defaultdict(list)
            for img in imgs:
                conteo_por_split[estado[img]].append(img)

            donante = max(
                (s for s in todos_los_splits if s != split_objetivo),
                key=lambda s: len(conteo_por_split.get(s, [])),
                default=None,
            )
            candidatos = conteo_por_split.get(donante, [])
            if not candidatos:
                continue
            if donante == "train" and len(candidatos) < 2:
                # no dejar a 'train' en cero para esta clase
                continue

            imagen_a_mover = candidatos[0]
            estado[imagen_a_mover] = split_objetivo
            movimientos.append((clase, imagen_a_mover, donante, split_objetivo))

    for img, nuevo_split in estado.items():
        imagenes[img]["split"] = nuevo_split

    return movimientos


def resumen_por_split(manifiesto: dict) -> Dict[str, int]:
    conteos: Dict[str, int] = defaultdict(int)
    for info in manifiesto["imagenes"].values():
        conteos[info["split"]] += 1
    return dict(conteos)


def huecos_de_cobertura(manifiesto: dict, splits_evaluables=("val", "test")) -> Dict[str, List[str]]:
    """Diagnóstico final: qué clases, después de la reparación, siguen sin ninguna
    imagen en val y/o test. Con datasets tan chicos, dos clases pueden competir por
    la MISMA imagen como único puente hacia splits distintos — no siempre es posible
    resolver todo en una sola pasada. Esto lo deja explícito en vez de esconderlo."""
    imagenes_por_clase: Dict[str, set] = defaultdict(set)
    for info in manifiesto["imagenes"].values():
        for clase in info.get("clases", []):
            imagenes_por_clase[clase].add(info["split"])

    huecos: Dict[str, List[str]] = {}
    for clase, splits_presentes in imagenes_por_clase.items():
        faltantes = [s for s in splits_evaluables if s not in splits_presentes]
        if faltantes:
            huecos[clase] = faltantes
    return huecos


def main():
    parser = argparse.ArgumentParser(description="Particion reproducible train/val/test a nivel imagen para Co-MIL")
    parser.add_argument("--bolsas", required=True, help="Carpeta con las bolsas .pt (ej. Bolsas_MIL_Procesadas/224px)")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5, help="Numero de folds para una eventual validacion cruzada")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--app-dir", default=None, help="Carpeta con custom_classes.json/renombres_clases.json (default: CO-MIL/)")
    args = parser.parse_args()

    proporciones = {"train": args.train, "val": args.val, "test": args.test}
    suma = sum(proporciones.values())
    if abs(suma - 1.0) > 1e-6:
        print(f"[!] Las proporciones deben sumar 1.0 (suma actual: {suma:.4f})")
        return

    app_dir = args.app_dir or _DIRECTORIO_ACTUAL
    catalogo_vigente = catalogo_tejidos.cargar_catalogo_vigente(app_dir)
    renombres = catalogo_tejidos.cargar_renombres(app_dir)

    grupos = agrupar_por_imagen(args.bolsas)
    if not grupos:
        print(f"[!] No se encontraron bolsas .pt en {args.bolsas}")
        return
    print(f"[+] {len(grupos)} imagenes unicas encontradas en {args.bolsas} ({sum(len(v) for v in grupos.values())} bolsas)")

    imagenes_clases = clases_por_imagen(grupos, catalogo_vigente, renombres)

    ruta_manifiesto = os.path.join(os.path.dirname(os.path.normpath(args.bolsas)), "splits_manifest.json")
    manifiesto = cargar_manifiesto(ruta_manifiesto)
    n_previas = len(manifiesto.get("imagenes", {}))

    manifiesto, nuevas = construir_particion(imagenes_clases, manifiesto, proporciones, args.folds, args.seed)

    movimientos = garantizar_cobertura_minima(manifiesto)
    guardar_manifiesto(ruta_manifiesto, manifiesto)

    print(f"[+] {len(nuevas)} imagenes nuevas asignadas a un split ({n_previas} ya tenian split y se conservaron tal cual).")
    if movimientos:
        print(f"[+] {len(movimientos)} ajuste(s) de cobertura minima (clases que habian quedado sin representacion en val/test):")
        for clase, imagen, donante, destino in movimientos:
            print(f"    - '{clase}': se movio {imagen} de {donante} -> {destino}")
    else:
        print("[+] Sin ajustes de cobertura minima necesarios.")
    print(f"[+] Distribucion actual: {resumen_por_split(manifiesto)}")

    clases_presentes = {c for info in manifiesto["imagenes"].values() for c in info.get("clases", [])}
    sin_ningun_ejemplo = [c for c in catalogo_vigente if c not in clases_presentes]
    if sin_ningun_ejemplo:
        print(f"[!] Clases del catalogo SIN NINGUN ejemplo anotado todavia en este dataset: {sin_ningun_ejemplo}")

    huecos = huecos_de_cobertura(manifiesto)
    if huecos:
        print("[!] Clases que SIGUEN sin ningun ejemplo en alguno de val/test (con datasets chicos, "
              "no siempre se puede resolver en una sola pasada -- se espera que se resuelva solo "
              "conforme crezca el numero de imagenes anotadas por clase):")
        for clase, faltantes in huecos.items():
            print(f"    - '{clase}': falta en {faltantes}")
    else:
        print("[+] Cobertura completa: todas las clases anotadas tienen al menos un ejemplo en cada split evaluable.")

    print(f"[+] Manifiesto guardado en: {ruta_manifiesto}")


if __name__ == "__main__":
    main()
