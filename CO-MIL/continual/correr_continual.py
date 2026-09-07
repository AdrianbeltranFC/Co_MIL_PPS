"""
=========================================================================================
PROTOTIPO DE APRENDIZAJE CONTINUO  ---  bucle + métricas de olvido + boxplots
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:
  python CO-MIL/continual/correr_continual.py --modo clase
  python CO-MIL/continual/correr_continual.py --modo domain --semillas 42,1,7,2,3

Objetivo 3 del plan de tesis ("la Co de Co-MIL"). Dos escenarios (van de Ven 2022):

  --modo domain  (Domain-Incremental): las clases son fijas {Fibrina, Granulación,
     Callo}; lo que cambia entre lotes es la distribución de entrada (cámara/sitio).
     Los T=3 sitios se simulan con shift fotométrico (ver datos_continual.py).
     -> Resultado 6-sep: el olvido es LEVE (backbone congelado + cabeza chica +
        clases compartidas = poco que sobrescribir).

  --modo clase   (Class-Incremental): la taxonomía CRECE. En el paso k solo se
     supervisa la clase k; el modelo debe aprenderla sin olvidar las anteriores.
     Con backbone congelado, aquí el olvido SÍ pesa (las capas de atención
     compartidas se desplazan al optimizar solo la clase nueva). Es lo que hacen
     los dos únicos precedentes de MIL + continual (MICIL, Ebrahimi).

Estrategias: naive (cota inf.) · joint (cota sup.) · experience replay · latent
replay. Métricas desde R[i,j] = F1 en (el lote / la clase) j tras el paso i:
ACC, BWT (Lopez-Paz & Ranzato 2017), Forgetting (Chaudhry et al. 2018).
=========================================================================================
"""

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.append(_DIR)

from datos_continual import (CLASES_TEJIDO, NUM_CLASES, RAIZ_REPO, cachear_features, cargar,
                             resumen_dominios, split_global, split_por_dominio)
from estrategias import construir_estrategia, entrenar_cabeza, evaluar_cabeza, nueva_cabeza

DIR_SALIDA = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "continual")
COL_ESTR = {"naive": "#B4791A", "joint": "#5C6468", "er": "#1C6B63", "latente": "#7B3FA0"}


def metricas_R(R: np.ndarray) -> dict:
    T = R.shape[0]
    acc = float(np.mean(R[T - 1]))
    if T > 1:
        bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))
        olvido = float(np.mean([max(R[i, j] for i in range(j, T - 1)) - R[T - 1, j]
                                for j in range(T - 1)]))
    else:
        bwt = olvido = 0.0
    return {"ACC": acc, "BWT": bwt, "Forgetting": olvido}


def _pos_weight(Y_tr):
    n_pos = Y_tr.sum(0).clamp(min=1.0)
    return (len(Y_tr) - n_pos) / n_pos


# ---------------------------------------------------------------------------
# Modo Domain-Incremental
# ---------------------------------------------------------------------------
def correr_domain(nombre_estr, orden, splits, X, Y, semilla, epocas, buffer, disp, cabeza="kbranch"):
    modelo = nueva_cabeza(semilla, disp, tipo=cabeza)
    estr = construir_estrategia(nombre_estr, X, Y, buffer_por_clase=buffer, semilla=semilla)
    T = len(orden)
    R = np.zeros((T, T))
    todas = list(range(NUM_CLASES))
    mask_full = torch.ones(X.shape[:2], dtype=torch.bool)
    for i, dom in enumerate(orden):
        idx_tr = splits[dom][0]
        H, M, Yt, C = estr.datos_entrenamiento(idx_tr, todas)
        entrenar_cabeza(modelo, H, M, Yt, C, epocas=epocas, dispositivo=disp,
                        pos_weight=_pos_weight(Yt))
        estr.actualizar_buffer(idx_tr, todas)
        for j, dom_ev in enumerate(orden):
            idx_te = splits[dom_ev][1]
            R[i, j] = evaluar_cabeza(modelo, X[idx_te], mask_full[idx_te], Y[idx_te], disp)["f1_macro"]
    return R


# ---------------------------------------------------------------------------
# Modo Class-Incremental
# ---------------------------------------------------------------------------
def correr_clase(nombre_estr, orden, idx_tr, idx_te, X, Y, semilla, epocas, buffer, disp, cabeza="kbranch"):
    modelo = nueva_cabeza(semilla, disp, tipo=cabeza)
    estr = construir_estrategia(nombre_estr, X, Y, buffer_por_clase=buffer, semilla=semilla)
    T = len(orden)
    R = np.zeros((T, T))
    mask_full = torch.ones(X.shape[:2], dtype=torch.bool)
    for i in range(T):
        cols_paso = [orden[i]]
        H, M, Yt, C = estr.datos_entrenamiento(idx_tr, cols_paso)
        entrenar_cabeza(modelo, H, M, Yt, C, epocas=epocas, dispositivo=disp,
                        pos_weight=_pos_weight(Yt))
        estr.actualizar_buffer(idx_tr, cols_paso)
        met = evaluar_cabeza(modelo, X[idx_te], mask_full[idx_te], Y[idx_te], disp)
        for j in range(T):
            v = met["f1_por_clase"][orden[j]]
            R[i, j] = 0.0 if v != v else v
    return R


# ---------------------------------------------------------------------------
# Boxplots
# ---------------------------------------------------------------------------
def graficar(agregado, modo, ruta_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    estr = list(agregado.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, met in zip(axes, ("ACC", "BWT", "Forgetting")):
        bp = ax.boxplot([agregado[e][met] for e in estr], tick_labels=estr,
                        patch_artist=True, widths=0.6)
        for parche, e in zip(bp["boxes"], estr):
            parche.set_facecolor(COL_ESTR.get(e, "#999")); parche.set_alpha(0.6)
        for m in bp["medians"]:
            m.set_color("#22282B")
        ax.set_title(met); ax.grid(alpha=0.3, axis="y")
        if met != "ACC":
            ax.axhline(0, color="#B9C0C2", lw=1)
    nombre = {"domain": "Domain-Incremental", "clase": "Class-Incremental"}[modo]
    fig.suptitle(f"Aprendizaje continuo ({nombre}, DFUTissue) --- "
                 "distribución sobre órdenes x semillas", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modo", default="clase", choices=["domain", "clase"])
    p.add_argument("--cabeza", default="kbranch", choices=["kbranch", "compartida"],
                   help="kbranch = arquitectura CoMIL (K ramas); compartida = ablación de cabeza única")
    p.add_argument("--estrategias", default="naive,joint,er,latente")
    p.add_argument("--semillas", default="42,1,7")
    p.add_argument("--dominios", type=int, default=3, help="solo modo domain")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--buffer", type=int, default=15, help="exemplars por clase (er, latente)")
    p.add_argument("--semilla_datos", type=int, default=0)
    p.add_argument("--ordenes", default="todos", help="'todos' o 'identidad'")
    p.add_argument("--forzar_cache", action="store_true")
    args = p.parse_args()

    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estrategias = [s.strip() for s in args.estrategias.split(",")]
    semillas = [int(s) for s in args.semillas.split(",")]

    T = args.dominios if args.modo == "domain" else NUM_CLASES
    cachear_features(T=args.dominios, dispositivo=disp, semilla=args.semilla_datos,
                     forzar=args.forzar_cache)
    nombres, X, Y, sitio = cargar(T=args.dominios)

    if args.modo == "domain":
        print(resumen_dominios(sitio, Y))
        splits = split_por_dominio(sitio, semilla=args.semilla_datos)
        elementos = sorted(int(d) for d in set(sitio.tolist()))
    else:
        idx_tr, idx_te = split_global(len(nombres), semilla=args.semilla_datos)
        print(f"class-incremental | train {len(idx_tr)} / test {len(idx_te)} | "
              f"orden de clases = {CLASES_TEJIDO}")
        elementos = list(range(NUM_CLASES))

    ordenes = ([tuple(elementos)] if args.ordenes == "identidad"
               else [tuple(int(x) for x in o) for o in itertools.permutations(elementos)])

    ruta = os.path.join(DIR_SALIDA, f"{args.modo}_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta, exist_ok=True)
    print(f"\n=== continual [{args.modo}] | {disp.type.upper()} | {estrategias} | "
          f"{len(ordenes)} órdenes x {len(semillas)} semillas | buffer={args.buffer}/clase ===\n")

    runs = []
    t0 = time.time()
    for ne in estrategias:
        for orden in ordenes:
            for s in semillas:
                if args.modo == "domain":
                    R = correr_domain(ne, orden, splits, X, Y, s, args.epocas, args.buffer, disp, args.cabeza)
                else:
                    R = correr_clase(ne, orden, idx_tr, idx_te, X, Y, s, args.epocas, args.buffer, disp, args.cabeza)
                m = metricas_R(R)
                runs.append({"estrategia": ne, "orden": list(orden), "semilla": s,
                             "R": R.tolist(), **m})
                print(f"  {ne:>8}  orden={orden}  s={s}  "
                      f"ACC={m['ACC']:.3f}  BWT={m['BWT']:+.3f}  Forgetting={m['Forgetting']:+.3f}")

    agregado = {e: {k: [r[k] for r in runs if r["estrategia"] == e]
                    for k in ("ACC", "BWT", "Forgetting")} for e in estrategias}
    with open(os.path.join(ruta, "resultados_continual.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "runs": runs}, f, ensure_ascii=False, indent=2)
    graficar(agregado, args.modo, os.path.join(ruta, "boxplots_continual.png"))

    print(f"\n=== RESUMEN [{args.modo}] (media ± desv. sobre {len(ordenes)}x{len(semillas)} corridas) ===")
    print(f"{'estrategia':>10} | {'ACC':>15} {'BWT':>16} {'Forgetting':>16}")
    for e in estrategias:
        a = agregado[e]
        f = lambda k, sg="": f"{np.mean(a[k]):{sg}.3f}±{np.std(a[k]):.3f}"
        print(f"{e:>10} | {f('ACC'):>15} {f('BWT', '+'):>16} {f('Forgetting', '+'):>16}")
    print(f"\n[{time.time() - t0:.0f}s]  [+] {os.path.relpath(ruta, RAIZ_REPO)}")


if __name__ == "__main__":
    main()
