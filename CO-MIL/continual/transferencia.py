"""
=========================================================================================
¿CUÁNTO CAE EL MODELO AL CAMBIAR DE POBLACIÓN?  (DFUTissue -> lote mexicano)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/continual/transferencia.py

Motivación (la real del tutor): la app del artículo previo, entrenada en un
conjunto de datos, FALLA con fotos de otra población (otros tonos de piel, otra
cámara, otro fondo). Los experimentos continuos con shift fotométrico simulado NO
reproducían eso porque el "cambio de dominio" era falso (filtros sobre las mismas
110 fotos). Este script usa un segundo dominio REAL: las ~185 regiones del lote
experto mexicano con etiqueta débil de tejido.

Dos pasos:
  1. EL SALTO. Entrenar la cabeza MIL solo en DFUTissue; medir F1 (clasificación
     de imagen, 3 tejidos) en el test de DFUTissue y en el lote mexicano NUNCA
     visto. La diferencia es el salto de dominio.
  2. ¿LO ARREGLA EL APRENDIZAJE CONTINUO? Entrenar en secuencia DFUTissue ->
     mexicano con cada estrategia; medir cuánto se recupera en mexicano y cuánto
     se pierde (si algo) en DFUTissue.

Todo con el extractor MobileNetV2 congelado y las características cacheadas.
=========================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.append(_DIR)

from datos_continual import (CLASES_TEJIDO, NUM_CLASES, RAIZ_REPO, cachear_features,
                             cachear_features_mexicano, cargar, cargar_mexicano)
from estrategias import construir_estrategia, entrenar_cabeza, evaluar_cabeza, nueva_cabeza

DIR_SALIDA = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "continual")
TODAS = list(range(NUM_CLASES))


def _f1(modelo, X, idx, Y, disp):
    m = torch.ones(len(idx), X.shape[1], dtype=torch.bool)
    return evaluar_cabeza(modelo, X[idx], m, Y[idx], disp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", default="42,1,7,2,3")
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--buffer", type=int, default=20)
    p.add_argument("--frac_test_mex", type=float, default=0.4)
    p.add_argument("--forzar_cache", action="store_true")
    args = p.parse_args()
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semillas = [int(s) for s in args.semillas.split(",")]

    # --- datos ---
    cachear_features(T=1, dispositivo=disp, forzar=args.forzar_cache)     # DFUTissue limpio
    cachear_features_mexicano(dispositivo=disp, forzar=args.forzar_cache)
    _, Xd, Yd, _ = cargar(T=1)
    _, Xm, Ym = cargar_mexicano()
    # DFUTissue: partición oficial (0..77 train, 78..93 val, 94..109 test)
    dfu_tr, dfu_te = np.arange(78), np.arange(94, len(Xd))
    X = torch.cat([Xd, Xm], 0)
    Y = torch.cat([Yd, Ym], 0)
    off = len(Xd)
    print(f"DFUTissue: {len(Xd)} bolsas (train {len(dfu_tr)}, test {len(dfu_te)}) | "
          f"mexicano: {len(Xm)} ROIs")
    print(f"prevalencia mexicano (Fib,Gra,Cal): {Ym.sum(0).tolist()}\n")

    filas_salto, runs = [], []
    for s in semillas:
        rng = np.random.default_rng(s)
        idx_m = off + rng.permutation(len(Xm))
        n_te = int(round(len(Xm) * args.frac_test_mex))
        mex_te, mex_tr = idx_m[:n_te], idx_m[n_te:]

        # ---- Paso 1: el salto ----
        modelo = nueva_cabeza(s, disp)
        estr = construir_estrategia("naive", X, Y)
        H, M, Yt, C = estr.datos_entrenamiento(dfu_tr, TODAS)
        n_pos = Yt.sum(0).clamp(min=1.0)
        entrenar_cabeza(modelo, H, M, Yt, C, epocas=args.epocas, dispositivo=disp,
                        pos_weight=(len(Yt) - n_pos) / n_pos)
        f_dfu = _f1(modelo, X, dfu_te, Y, disp)["f1_macro"]
        f_mex = _f1(modelo, X, mex_te, Y, disp)["f1_macro"]
        filas_salto.append((s, f_dfu, f_mex))
        print(f"[semilla {s}]  entrenado en DFUTissue  ->  F1 DFUTissue {f_dfu:.3f}  |  "
              f"F1 mexicano {f_mex:.3f}   (salto {f_dfu - f_mex:+.3f})")

        # ---- Paso 2: continual DFUTissue -> mexicano ----
        for ne in ("naive", "joint", "er", "latente"):
            modelo = nueva_cabeza(s, disp)
            estr = construir_estrategia(ne, X, Y, buffer_por_clase=args.buffer, semilla=s)
            R = np.zeros((2, 2))
            for i, idx_lote in enumerate((dfu_tr, mex_tr)):
                H, M, Yt, C = estr.datos_entrenamiento(idx_lote, TODAS)
                n_pos = Yt.sum(0).clamp(min=1.0)
                entrenar_cabeza(modelo, H, M, Yt, C, epocas=args.epocas, dispositivo=disp,
                                pos_weight=(len(Yt) - n_pos) / n_pos)
                estr.actualizar_buffer(idx_lote, TODAS)
                R[i, 0] = _f1(modelo, X, dfu_te, Y, disp)["f1_macro"]
                R[i, 1] = _f1(modelo, X, mex_te, Y, disp)["f1_macro"]
            runs.append({"estrategia": ne, "semilla": s, "R": R.tolist(),
                         "olvido_dfu": float(R[0, 0] - R[1, 0]),
                         "ganancia_mex": float(R[1, 1] - R[0, 1]),
                         "final_dfu": float(R[1, 0]), "final_mex": float(R[1, 1])})

    # --- resumen ---
    sd = np.array([[a, b] for _, a, b in filas_salto])
    print(f"\n=== EL SALTO (media sobre {len(semillas)} semillas) ===")
    print(f"  F1 DFUTissue (test)         {sd[:, 0].mean():.3f} ± {sd[:, 0].std():.3f}")
    print(f"  F1 mexicano (nunca visto)   {sd[:, 1].mean():.3f} ± {sd[:, 1].std():.3f}")
    print(f"  salto                       {(sd[:, 0] - sd[:, 1]).mean():+.3f}")

    print(f"\n=== DFUTissue -> mexicano en secuencia (media ± desv.) ===")
    print(f"{'estrategia':>10} | {'final DFUTissue':>16} {'final mexicano':>16} "
          f"{'olvido DFU':>12} {'ganancia MEX':>13}")
    for ne in ("naive", "joint", "er", "latente"):
        r = [x for x in runs if x["estrategia"] == ne]
        g = lambda k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r]))
        fd, fm, od, gm = g("final_dfu"), g("final_mex"), g("olvido_dfu"), g("ganancia_mex")
        print(f"{ne:>10} | {fd[0]:.3f}±{fd[1]:.3f}     {fm[0]:.3f}±{fm[1]:.3f}     "
              f"{od[0]:+.3f}      {gm[0]:+.3f}")

    ruta = os.path.join(DIR_SALIDA, "transferencia_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta, exist_ok=True)
    with open(os.path.join(ruta, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump({"salto": filas_salto, "runs": runs, "config": vars(args)}, f,
                  ensure_ascii=False, indent=2)
    _graficar(filas_salto, runs, os.path.join(ruta, "transferencia.png"))
    print(f"\n[+] {os.path.relpath(ruta, RAIZ_REPO)}")


def _graficar(salto, runs, ruta_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sd = np.array([[a, b] for _, a, b in salto])
    estr = ["naive", "joint", "er", "latente"]
    col = {"naive": "#B4791A", "joint": "#5C6468", "er": "#1C6B63", "latente": "#7B3FA0"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(["DFUTissue\n(test)", "mexicano\n(nunca visto)"],
              [sd[:, 0].mean(), sd[:, 1].mean()],
              yerr=[sd[:, 0].std(), sd[:, 1].std()], color=["#3F72A8", "#BB3B2E"], alpha=.8)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("F1 (clasificación de tejido)")
    ax[0].set_title("Paso 1 — el salto de dominio\n(entrenado solo en DFUTissue)")
    ax[0].grid(alpha=.3, axis="y")
    x = np.arange(len(estr)); w = 0.38
    fdu = [np.mean([r["final_dfu"] for r in runs if r["estrategia"] == e]) for e in estr]
    fme = [np.mean([r["final_mex"] for r in runs if r["estrategia"] == e]) for e in estr]
    ax[1].bar(x - w / 2, fdu, w, label="DFUTissue", color="#3F72A8", alpha=.8)
    ax[1].bar(x + w / 2, fme, w, label="mexicano", color="#BB3B2E", alpha=.8)
    ax[1].axhline(sd[:, 1].mean(), color="#888", ls=":", label="mexicano sin continual")
    ax[1].set_xticks(x); ax[1].set_xticklabels(estr); ax[1].set_ylim(0, 1)
    ax[1].set_title("Paso 2 — tras entrenar también en mexicano"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(ruta_png, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
