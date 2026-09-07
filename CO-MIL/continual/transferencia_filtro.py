"""
=========================================================================================
¿EL SALTO DE DOMINIO ES POR CÁMARA/COLOR O POR LA POBLACIÓN?
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/continual/transferencia_filtro.py

Idea del autor. El modelo entrenado en DFUTissue cae ~44 puntos de F1 en fotos
mexicanas nunca vistas (ver transferencia.py). ¿Cuánto de esa caída es cámara/
color (arreglable con un filtro) y cuánto es la población en sí (tono de piel,
tipo de herida)?

Se mide el salto en cuatro condiciones:
  base       : entrenar DFUTissue tal cual.
  aug        : entrenar DFUTissue con aumentación fuerte de cámara/color
               (4 copias fotométricas aleatorias por imagen).
  colornorm  : normalizar el color de las fotos mexicanas al de DFUTissue antes
               de que el modelo las vea (transferencia de color global).
  aug+norm   : las dos cosas.

Si el filtro cierra buena parte del hueco -> era cámara/color.
Si apenas lo mueve -> es la población, y hacen falta datos mexicanos de verdad.
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

from datos_continual import (NUM_CLASES, RAIZ_REPO, cachear_features, cachear_features_dfu_aug,
                             cachear_features_mexicano, cargar, cargar_dfu_aug, cargar_mexicano)
from estrategias import CabezaMIL, entrenar_cabeza, evaluar_cabeza, nueva_cabeza

DIR_SALIDA = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "continual")
TODAS = list(range(NUM_CLASES))


def _entrenar_dfu(Xtr, Ytr, semilla, epocas, disp):
    torch.manual_seed(semilla)
    modelo = CabezaMIL().to(disp)
    M = torch.ones(len(Xtr), Xtr.shape[1], dtype=torch.bool)
    C = torch.ones(len(Xtr), NUM_CLASES, dtype=torch.bool)
    n_pos = Ytr.sum(0).clamp(min=1.0)
    entrenar_cabeza(modelo, Xtr, M, Ytr, C, epocas=epocas, dispositivo=disp,
                    pos_weight=(len(Ytr) - n_pos) / n_pos)
    return modelo


def _f1(modelo, X, Y, disp):
    m = torch.ones(len(X), X.shape[1], dtype=torch.bool)
    return evaluar_cabeza(modelo, X, m, Y, disp)["f1_macro"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", default="42,1,7,2,3")
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--copias_aug", type=int, default=4)
    p.add_argument("--forzar_cache", action="store_true")
    args = p.parse_args()
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semillas = [int(s) for s in args.semillas.split(",")]

    cachear_features(T=1, dispositivo=disp, forzar=args.forzar_cache)
    cachear_features_dfu_aug(args.copias_aug, dispositivo=disp, forzar=args.forzar_cache)
    cachear_features_mexicano(dispositivo=disp, forzar=args.forzar_cache)
    cachear_features_mexicano(dispositivo=disp, forzar=args.forzar_cache, colornorm=True)

    _, Xd, Yd, _ = cargar(T=1)
    Xda, Yda = cargar_dfu_aug(args.copias_aug)
    _, Xm, Ym = cargar_mexicano(colornorm=False)
    _, Xmn, Ymn = cargar_mexicano(colornorm=True)
    dfu_tr, dfu_te = np.arange(78), np.arange(94, len(Xd))
    aug_tr = np.arange(78 * args.copias_aug)          # las 78 de train, x copias

    condiciones = {
        "base":      (Xd[dfu_tr], Yd[dfu_tr], Xm, Ym),
        "aug":       (Xda[aug_tr], Yda[aug_tr], Xm, Ym),
        "colornorm": (Xd[dfu_tr], Yd[dfu_tr], Xmn, Ymn),
        "aug+norm":  (Xda[aug_tr], Yda[aug_tr], Xmn, Ymn),
    }

    res = {c: {"dfu": [], "mex": []} for c in condiciones}
    for s in semillas:
        for nombre, (Xtr, Ytr, Xmex, Ymex) in condiciones.items():
            modelo = _entrenar_dfu(Xtr, Ytr, s, args.epocas, disp)
            fd = _f1(modelo, Xd[dfu_te], Yd[dfu_te], disp)
            fm = _f1(modelo, Xmex, Ymex, disp)
            res[nombre]["dfu"].append(fd); res[nombre]["mex"].append(fm)
            print(f"[semilla {s}] {nombre:>10}  F1 DFUTissue {fd:.3f}  |  F1 mexicano {fm:.3f}  "
                  f"(salto {fd - fm:+.3f})")

    print(f"\n=== EL SALTO POR CONDICIÓN (media ± desv. sobre {len(semillas)} semillas) ===")
    print(f"{'condición':>10} | {'F1 DFUTissue':>14} {'F1 mexicano':>14} {'salto':>10}")
    base_salto = np.mean(np.array(res["base"]["dfu"]) - np.array(res["base"]["mex"]))
    for c in condiciones:
        d, m = np.array(res[c]["dfu"]), np.array(res[c]["mex"])
        salto = np.mean(d - m)
        cierre = "" if c == "base" else f"  (cierra {base_salto - salto:+.3f})"
        print(f"{c:>10} | {d.mean():.3f}±{d.std():.3f}   {m.mean():.3f}±{m.std():.3f}   "
              f"{salto:+.3f}{cierre}")

    ruta = os.path.join(DIR_SALIDA, "filtro_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta, exist_ok=True)
    with open(os.path.join(ruta, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump({"res": res, "config": vars(args)}, f, ensure_ascii=False, indent=2)
    _graficar(res, list(condiciones), os.path.join(ruta, "filtro.png"))
    print(f"\n[+] {os.path.relpath(ruta, RAIZ_REPO)}")


def _graficar(res, cond, ruta_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(cond)); w = 0.38
    dfu = [np.mean(res[c]["dfu"]) for c in cond]
    mex = [np.mean(res[c]["mex"]) for c in cond]
    dfu_e = [np.std(res[c]["dfu"]) for c in cond]
    mex_e = [np.std(res[c]["mex"]) for c in cond]
    ax.bar(x - w / 2, dfu, w, yerr=dfu_e, label="DFUTissue (test)", color="#3F72A8", alpha=.85)
    ax.bar(x + w / 2, mex, w, yerr=mex_e, label="mexicano (nunca visto)", color="#BB3B2E", alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(cond); ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (clasificación de tejido)")
    ax.set_title("¿El salto de dominio es cámara/color o población?")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(ruta_png, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
