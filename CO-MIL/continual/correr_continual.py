"""
=========================================================================================
PROTOTIPO DE APRENDIZAJE CONTINUO  ---  bucle Domain-Incremental + métricas + boxplots
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:
  python CO-MIL/continual/correr_continual.py
  python CO-MIL/continual/correr_continual.py --estrategias naive,joint,er,latente --semillas 42,1

Qué hace (objetivo 3 del plan de tesis, "la Co de Co-MIL"; prototipo mínimo):
  1. Cachea las características congeladas de MobileNetV2 para DFUTissue (una vez).
  2. Simula T=3 lotes por agrupamiento de características (covariate shift real).
  3. Para cada estrategia x orden de lotes x semilla: entrena la cabeza MIL lote a
     lote y rellena la matriz de accuracy  R[i,j] = F1-macro en el lote j tras
     entrenar hasta el lote i.
  4. De R saca ACC, BWT (retro-transferencia; negativo = olvido) y Forgetting
     Measure; agrega sobre (órdenes x semillas) y dibuja un boxplot por métrica.

Cotas que deben aparecer: naive (inferior, olvida), joint (superior, oráculo);
las de repetición (er, latente) deben quedar en medio. Esto es también el
esqueleto del objetivo 4 (comparar los 3 escenarios).

Prototipo mínimo: faltan por añadir LwF y NCM (una ancla por familia), el barrido
de tamaño de búfer, la ablación de olvido por capa (atención vs. clasificador,
Li et al. CVPR 2025) y el test de Wilcoxon.
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

from datos_continual import (CLASES_TEJIDO, RAIZ_REPO, cachear_features, cargar,
                             resumen_dominios, split_por_dominio)
from estrategias import construir_estrategia, entrenar_cabeza, evaluar_cabeza, nueva_cabeza

DIR_SALIDA = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "continual")


# ---------------------------------------------------------------------------
# Métricas desde la matriz R  (Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Una corrida continua completa (una estrategia, un orden, una semilla)
# ---------------------------------------------------------------------------
def correr_una(nombre_estr, orden, splits, X, Y, semilla, epocas, buffer, dispositivo):
    modelo = nueva_cabeza(semilla, dispositivo)
    estr = construir_estrategia(nombre_estr, X, Y, buffer_por_clase=buffer, semilla=semilla)
    T = len(orden)
    R = np.zeros((T, T))
    mask_full = torch.ones(X.shape[:2], dtype=torch.bool)
    for i, dom in enumerate(orden):
        idx_tr = splits[dom][0]
        H_tr, M_tr, Y_tr = estr.datos_entrenamiento(idx_tr)
        n_pos = Y_tr.sum(0).clamp(min=1.0)
        pos_weight = (len(Y_tr) - n_pos) / n_pos
        entrenar_cabeza(modelo, H_tr, M_tr, Y_tr, epocas=epocas, dispositivo=dispositivo,
                        pos_weight=pos_weight)
        estr.actualizar_buffer(idx_tr)
        for j, dom_ev in enumerate(orden):
            idx_te = splits[dom_ev][1]
            R[i, j] = evaluar_cabeza(modelo, X[idx_te], mask_full[idx_te], Y[idx_te],
                                     dispositivo)["f1_macro"]
    return R


# ---------------------------------------------------------------------------
# Boxplots
# ---------------------------------------------------------------------------
def graficar(agregado, ruta_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metricas = ["ACC", "BWT", "Forgetting"]
    estrategias = list(agregado.keys())
    col = {"naive": "#B4791A", "joint": "#5C6468", "er": "#1C6B63", "latente": "#7B3FA0"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, met in zip(axes, metricas):
        datos = [agregado[e][met] for e in estrategias]
        bp = ax.boxplot(datos, tick_labels=estrategias, patch_artist=True, widths=0.6)
        for parche, e in zip(bp["boxes"], estrategias):
            parche.set_facecolor(col.get(e, "#999")); parche.set_alpha(0.6)
        for mediana in bp["medians"]:
            mediana.set_color("#22282B")
        ax.set_title(met); ax.grid(alpha=0.3, axis="y")
        if met != "ACC":
            ax.axhline(0, color="#B9C0C2", lw=1)
    fig.suptitle("Aprendizaje continuo (Domain-IL sobre DFUTissue) --- "
                 "distribución sobre órdenes de lote x semillas", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--estrategias", default="naive,joint,er,latente")
    p.add_argument("--semillas", default="42")
    p.add_argument("--dominios", type=int, default=3)
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--buffer", type=int, default=10, help="exemplars por clase (er, latente)")
    p.add_argument("--semilla_dominios", type=int, default=0)
    p.add_argument("--ordenes", default="todos", help="'todos' (T! permutaciones) o 'identidad'")
    p.add_argument("--forzar_cache", action="store_true")
    args = p.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estrategias = [s.strip() for s in args.estrategias.split(",")]
    semillas = [int(s) for s in args.semillas.split(",")]

    cachear_features(T=args.dominios, dispositivo=dispositivo, semilla=args.semilla_dominios,
                     forzar=args.forzar_cache)
    nombres, X, Y, sitio = cargar(T=args.dominios)
    print(resumen_dominios(sitio, Y))
    splits = split_por_dominio(sitio, semilla=args.semilla_dominios)

    dominios = sorted(int(d) for d in set(sitio.tolist()))
    if args.ordenes == "identidad":
        ordenes = [tuple(dominios)]
    else:
        ordenes = [tuple(int(x) for x in o) for o in itertools.permutations(dominios)]

    ruta_exp = os.path.join(DIR_SALIDA, "corrida_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta_exp, exist_ok=True)
    print(f"\n=== continual | {dispositivo.type.upper()} | estrategias={estrategias} | "
          f"{len(ordenes)} órdenes x {len(semillas)} semillas | buffer={args.buffer}/clase ===")
    print(f"    salida: {os.path.relpath(ruta_exp, RAIZ_REPO)}\n")

    runs = []
    t0 = time.time()
    for nombre_estr in estrategias:
        for orden in ordenes:
            for semilla in semillas:
                R = correr_una(nombre_estr, orden, splits, X, Y, semilla,
                               args.epocas, args.buffer, dispositivo)
                m = metricas_R(R)
                runs.append({"estrategia": nombre_estr, "orden": list(orden),
                             "semilla": semilla, "R": R.tolist(), **m})
                print(f"  {nombre_estr:>8}  orden={orden}  semilla={semilla}  "
                      f"ACC={m['ACC']:.3f}  BWT={m['BWT']:+.3f}  Forgetting={m['Forgetting']:.3f}")

    # --- agregado ---
    agregado = {}
    for e in estrategias:
        sub = [r for r in runs if r["estrategia"] == e]
        agregado[e] = {k: [r[k] for r in sub] for k in ("ACC", "BWT", "Forgetting")}

    with open(os.path.join(ruta_exp, "resultados_continual.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "sitio": sitio.tolist(),
                   "resumen_dominios": resumen_dominios(sitio, Y), "runs": runs}, f,
                  ensure_ascii=False, indent=2)
    graficar(agregado, os.path.join(ruta_exp, "boxplots_continual.png"))

    print(f"\n=== RESUMEN (media ± desv. sobre {len(ordenes)}x{len(semillas)} corridas) ===")
    print(f"{'estrategia':>10} | {'ACC':>14} {'BWT':>15} {'Forgetting':>15}")
    for e in estrategias:
        a = agregado[e]
        def mm(k, signo=""):
            return f"{np.mean(a[k]):{signo}.3f}±{np.std(a[k]):.3f}"
        print(f"{e:>10} | {mm('ACC'):>14} {mm('BWT', '+'):>15} {mm('Forgetting'):>15}")
    print(f"\n[{time.time() - t0:.0f}s]  [+] {os.path.relpath(ruta_exp, RAIZ_REPO)}")


if __name__ == "__main__":
    main()
