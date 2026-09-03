"""
=========================================================================================
CURVA DE EFICIENCIA DE ANOTACIÓN  (supervisión densa vs. débil vs. mixta)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/segmentacion/entrenar_eficiencia.py
               python CO-MIL/segmentacion/entrenar_eficiencia.py --mascaras 0,10,40,78 --epocas 120

Qué hace (tareas T3 + T4 del plan post-27-ago, ver bitácora Parte IV):
  Pregunta central de la tesis: ¿cuánta capacidad de localizar tejidos se recupera con
  anotación BARATA (solo etiqueta de imagen: qué tejidos hay) frente a anotación CARA
  (máscara densa por píxel)?

  Un solo modelo (FPN + MobileNetV2) entrenado con una mezcla:
    - N imágenes con máscara densa  -> pérdida de segmentación (Dice + entropía cruzada)
    - (78 - N) imágenes con solo etiqueta de imagen -> pérdida débil: se agrupa el mapa
      de logits por clase con log-sum-exp y se compara con la etiqueta multi-clase
      (BCE). Es el mecanismo estándar de segmentación débilmente supervisada.

  Se barre N in {0, 5, 10, 20, 40, 78}:
    N = 0   -> brazo puramente DÉBIL (tarea T3)
    N = 78  -> brazo puramente SUPERVISADO (= baseline T2)
    intermedios -> brazo MIXTO; la curva Dice-vs-N es la frontera de eficiencia (T4).

  Evaluación idéntica en todos los puntos (argmax de clases sobre los logits a 256px),
  para que la comparación aísle SOLO el efecto de cuánta anotación densa hay.

Notas:
  - Las etiquetas de imagen se derivan de las máscaras (qué clases tienen algún píxel):
    es gratis y consistente, y es exactamente lo que un clínico marcaría en segundos.
  - El subconjunto de N imágenes "con máscara" se elige de forma estratificada (cubrir
    las 3 clases de tejido) y con semilla fija, para que el barrido sea reproducible.
=========================================================================================
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.append(_DIR)

import segmentation_models_pytorch as smp
from dataset_seg import CLASES, NUM_CLASES, DFUTissueSeg
from entrenar_seg import construir_modelo, dice_iou_desde_confusion, evaluar, matriz_confusion

RAIZ_REPO = os.path.dirname(os.path.dirname(_DIR))
DIR_PESOS = os.path.join(RAIZ_REPO, "Pesos_Entrenados")
CLASES_TEJIDO = [1, 2, 3]  # 0 = fondo


# ---------------------------------------------------------------------------
# Dataset que expone: imagen, máscara, etiqueta de imagen, y si esta imagen
# "tiene máscara" en este experimento.
# ---------------------------------------------------------------------------
class DFUTissueMixto(Dataset):
    """`solo_supervisado=True` -> el dataset solo contiene las N imágenes con máscara
    (las demás ni se ven). `False` -> las 78, con una bandera de si tienen máscara."""

    def __init__(self, split, nombres_con_mascara, augment, aug_fuerte=False,
                 solo_supervisado=False):
        base = DFUTissueSeg(split, augment=augment, aug_fuerte=aug_fuerte)
        self.con_mascara = set(nombres_con_mascara)
        if solo_supervisado:
            self._idx = [i for i in range(len(base)) if base.nombres[i] in self.con_mascara]
        else:
            self._idx = list(range(len(base)))
        self.base = base

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, k):
        i = self._idx[k]
        x, y = self.base[i]
        nombre = self.base.nombres[i]
        etiqueta_img = torch.zeros(NUM_CLASES, dtype=torch.float32)
        for c in torch.unique(y):
            etiqueta_img[int(c)] = 1.0
        tiene_mascara = 1.0 if nombre in self.con_mascara else 0.0
        return x, y, etiqueta_img, torch.tensor(tiene_mascara)


def etiquetas_de_imagen_por_nombre(split: str) -> dict:
    ds = DFUTissueSeg(split)
    d = {}
    for i in range(len(ds)):
        _, ann = ds._cargar(i)
        d[ds.nombres[i]] = set(int(c) for c in np.unique(ann))
    return d


def elegir_subconjunto(n: int, etiquetas: dict, semilla: int = 42):
    """n nombres 'con máscara', estratificado para cubrir las 3 clases de tejido."""
    rng = random.Random(semilla)
    nombres = sorted(etiquetas.keys())
    rng.shuffle(nombres)
    if n >= len(nombres):
        return set(nombres)
    if n == 0:
        return set()
    elegidos = []
    # primero: al menos un ejemplo de cada clase de tejido
    for clase in CLASES_TEJIDO:
        for nom in nombres:
            if nom not in elegidos and clase in etiquetas[nom]:
                elegidos.append(nom)
                break
        if len(elegidos) >= n:
            return set(elegidos[:n])
    # rellenar
    for nom in nombres:
        if nom not in elegidos:
            elegidos.append(nom)
        if len(elegidos) >= n:
            break
    return set(elegidos[:n])


# ---------------------------------------------------------------------------
# Pérdidas
# ---------------------------------------------------------------------------
def lse_pool(logits: torch.Tensor, r: float = 5.0) -> torch.Tensor:
    """Agrupación log-sum-exp espacial por clase: [B,C,H,W] -> [B,C].
    Entre el promedio (sobre-activa) y el máximo (sub-activa); r controla lo picudo."""
    b, c, h, w = logits.shape
    z = logits.view(b, c, h * w)
    return (torch.logsumexp(r * z, dim=2) - np.log(h * w)) / r


def entrenar_un_punto(n_mascaras, nombres_mascara, epocas, batch, lr, paciencia,
                      semilla, dispositivo, modo="mixto", aug_fuerte=False, lambda_debil=1.0):
    """modo:
       'mixto'            -> N máscaras (pérdida de segmentación) + (78-N) etiquetas (pérdida débil).
       'solo_supervisado' -> N máscaras y nada más; las (78-N) restantes se IGNORAN.
       La diferencia entre las dos curvas = cuánto aporta de verdad la anotación débil
       (frente a «solo el modelo aprendiendo de N muestras»)."""
    torch.manual_seed(semilla)
    np.random.seed(semilla)

    ds_tr = DFUTissueMixto("train", nombres_mascara, augment=True, aug_fuerte=aug_fuerte,
                           solo_supervisado=(modo == "solo_supervisado"))
    if len(ds_tr) == 0:
        return {"n_mascaras": n_mascaras, "modo": modo, "epocas_corridas": 0,
                "mejor_epoca": 0, "minutos": 0.0, "val_dice_tejidos": 0.0,
                "test": evaluar(construir_modelo("FPN", "mobilenet_v2").to(dispositivo),
                                DataLoader(DFUTissueSeg("test"), batch_size=batch), dispositivo)}, None
    ds_va = DFUTissueSeg("val", augment=False)
    ds_te = DFUTissueSeg("test", augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=min(batch, max(1, len(ds_tr))), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False)

    modelo = construir_modelo("FPN", "mobilenet_v2").to(dispositivo)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    ce_loss = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=12, min_lr=1e-6)

    mejor_val, mejor_epoca, mejor_state = -1.0, -1, None
    t0 = time.time()
    for epoca in range(1, epocas + 1):
        modelo.train()
        for x, y, et_img, tiene_m in dl_tr:
            x, y = x.to(dispositivo), y.to(dispositivo)
            et_img, tiene_m = et_img.to(dispositivo), tiene_m.to(dispositivo)
            opt.zero_grad()
            logits = modelo(x)
            sup = tiene_m > 0.5
            deb = ~sup
            perdida = torch.zeros((), device=dispositivo)
            if sup.any():
                perdida = perdida + 0.5 * dice_loss(logits[sup], y[sup]) \
                          + 0.5 * ce_loss(logits[sup], y[sup])
            if deb.any():
                pooled = lse_pool(logits[deb])[:, CLASES_TEJIDO]
                perdida = perdida + lambda_debil * bce(pooled, et_img[deb][:, CLASES_TEJIDO])
            perdida.backward()
            opt.step()

        met_va = evaluar(modelo, dl_va, dispositivo)
        val_score = met_va["dice_medio_tejidos"]
        sched.step(val_score)
        if val_score > mejor_val:
            mejor_val, mejor_epoca = val_score, epoca
            mejor_state = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
        if epoca - mejor_epoca >= paciencia:
            break

    modelo.load_state_dict(mejor_state)
    met_te = evaluar(modelo, dl_te, dispositivo)
    return {
        "n_mascaras": n_mascaras,
        "modo": modo,
        "epocas_corridas": epoca,
        "mejor_epoca": mejor_epoca,
        "minutos": round((time.time() - t0) / 60, 1),
        "test": met_te,
        "val_dice_tejidos": mejor_val,
    }, mejor_state


def agregar_por_n(runs):
    """Agrupa por (modo, N) y saca media±std sobre semillas."""
    from collections import defaultdict
    grupos = defaultdict(list)
    for r in runs:
        grupos[(r.get("modo", "mixto"), r["n_mascaras"])].append(r)
    agregado = []
    for (modo, n) in sorted(grupos, key=lambda k: (k[0], k[1])):
        rr = grupos[(modo, n)]
        fila = {"modo": modo, "n_mascaras": n, "n_semillas": len(rr),
                "semillas": [r.get("semilla") for r in rr]}
        for clave in ["dice_medio_tejidos", "iou_medio_tejidos", "dice_medio_por_imagen"]:
            vals = [r["test"][clave] for r in rr]
            fila[clave + "_media"] = float(np.mean(vals))
            fila[clave + "_std"] = float(np.std(vals))
        for clase in ["Fibrina", "Granulación", "Callo"]:
            vals = [r["test"]["dice_por_clase"][clase] for r in rr]
            fila["dice_" + clase + "_media"] = float(np.mean(vals))
            fila["dice_" + clase + "_std"] = float(np.std(vals))
        agregado.append(fila)
    return agregado


def graficar_curva(runs, ruta_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ag = agregar_por_n(runs)
    modos = sorted({a["modo"] for a in ag})
    estilo = {"mixto": ("var", "#1C6B63", "-", "N máscaras + etiquetas débiles"),
              "solo_supervisado": ("var", "#B4791A", "--", "solo N máscaras (nada más)")}
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    if len(modos) == 1 and modos[0] == "mixto":
        # vista por tejido (una sola curva)
        colores = {"Fibrina": "#C0392B", "Granulación": "#2E9E5B", "Callo": "#3B7DD8"}
        sub = [a for a in ag if a["modo"] == "mixto"]
        ns = np.array([a["n_mascaras"] for a in sub])
        for clase in ["Fibrina", "Granulación", "Callo"]:
            m = np.array([a["dice_" + clase + "_media"] for a in sub])
            s = np.array([a["dice_" + clase + "_std"] for a in sub])
            ax.plot(ns, m, "o-", label=clase, color=colores[clase])
            ax.fill_between(ns, m - s, m + s, color=colores[clase], alpha=0.15)
        m = np.array([a["dice_medio_tejidos_media"] for a in sub])
        s = np.array([a["dice_medio_tejidos_std"] for a in sub])
        ax.plot(ns, m, "s--", label="media tejidos", color="#222", lw=2)
        ax.fill_between(ns, m - s, m + s, color="#222", alpha=0.12)
    else:
        # comparación: ¿aporta la anotación débil?  media de tejidos, un color por modo
        for modo in modos:
            sub = [a for a in ag if a["modo"] == modo]
            ns = np.array([a["n_mascaras"] for a in sub])
            m = np.array([a["dice_medio_tejidos_media"] for a in sub])
            s = np.array([a["dice_medio_tejidos_std"] for a in sub])
            _, col, ls, lbl = estilo.get(modo, ("", "#555", "-", modo))
            ax.plot(ns, m, "o", ls=ls, color=col, lw=2, label=lbl)
            ax.fill_between(ns, m - s, m + s, color=col, alpha=0.15)

    n_sem = max(a["n_semillas"] for a in ag)
    ax.set_xlabel("Nº de imágenes con máscara densa (de 78)")
    ax.set_ylabel("Dice medio en tejidos (test)")
    ax.set_title(f"Eficiencia de anotación — DFUTissue ({n_sem} semillas, banda = ±1σ)\n"
                 "la brecha entre curvas = cuánto aporta de verdad la anotación débil")
    ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mascaras", default="0,5,10,20,40,78",
                   help="lista de N separada por comas")
    p.add_argument("--semillas", default="42",
                   help="lista de semillas separada por comas (varias -> curva con bandas de error)")
    p.add_argument("--modos", default="mixto",
                   help="'mixto', 'solo_supervisado', o ambos separados por coma. Con ambos, la "
                        "brecha entre curvas responde: ¿aporta algo la anotación débil?")
    p.add_argument("--aug_fuerte", action="store_true")
    p.add_argument("--epocas", type=int, default=120)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--paciencia", type=int, default=28)
    p.add_argument("--guardar_modelos", action="store_true",
                   help="guardar un .pth por corrida (por defecto no, para no llenar el disco)")
    args = p.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valores_n = [int(x) for x in args.mascaras.split(",")]
    semillas = [int(x) for x in args.semillas.split(",")]
    modos = [m.strip() for m in args.modos.split(",")]
    etiquetas = etiquetas_de_imagen_por_nombre("train")

    ruta_exp = os.path.join(DIR_PESOS, "eficiencia_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta_exp, exist_ok=True)
    print(f"=== Curva de eficiencia | {dispositivo.type.upper()} | N = {valores_n} | semillas = {semillas} ===")
    print(f"    salida: {os.path.relpath(ruta_exp, RAIZ_REPO)}\n")

    runs = []
    for n in valores_n:
        for modo in modos:
            if modo == "solo_supervisado" and n == 0:
                continue  # sin imágenes no hay nada que entrenar
            for semilla in semillas:
                nombres_m = elegir_subconjunto(n, etiquetas, semilla)
                print(f"[N={n:2d} {modo} semilla={semilla}]  "
                      f"({len(nombres_m)} con máscara, "
                      f"{78 - len(nombres_m) if modo == 'mixto' else 0} con etiqueta débil)")
                res, state = entrenar_un_punto(n, nombres_m, args.epocas, args.batch, args.lr,
                                               args.paciencia, semilla, dispositivo,
                                               modo=modo, aug_fuerte=args.aug_fuerte)
                res["semilla"] = semilla
                d = res["test"]
                print(f"       -> Dice medio tejidos {d['dice_medio_tejidos']:.3f}  "
                      f"(Fib {d['dice_por_clase']['Fibrina']:.3f} | "
                      f"Gra {d['dice_por_clase']['Granulación']:.3f} | "
                      f"Cal {d['dice_por_clase']['Callo']:.3f})   [{res['minutos']} min]")
                if args.guardar_modelos and state is not None:
                    torch.save({"model_state_dict": state, "n_mascaras": n, "modo": modo,
                                "semilla": semilla},
                               os.path.join(ruta_exp, f"modelo_{modo}_N{n:02d}_s{semilla}.pth"))
                runs.append(res)
            with open(os.path.join(ruta_exp, "resultados.json"), "w", encoding="utf-8") as f:
                json.dump({"config": vars(args), "dispositivo": dispositivo.type,
                           "runs": runs, "agregado_por_n": agregar_por_n(runs)},
                          f, ensure_ascii=False, indent=2)
            graficar_curva(runs, os.path.join(ruta_exp, "curva_eficiencia.png"))

    print("\n=== RESUMEN (media ± desv. estándar sobre semillas) ===")
    print(f"{'modo':>17} {'N':>3} | {'Fibrina':>13} {'Granul.':>13} {'Callo':>13} {'Media':>13}")
    for a in agregar_por_n(runs):
        def mm(c):
            return f"{a['dice_' + c + '_media']:.3f}±{a['dice_' + c + '_std']:.3f}"
        print(f"{a['modo']:>17} {a['n_mascaras']:>3} | {mm('Fibrina'):>13} {mm('Granulación'):>13} "
              f"{mm('Callo'):>13} "
              f"{a['dice_medio_tejidos_media']:.3f}±{a['dice_medio_tejidos_std']:.3f}")
    print(f"\n[+] {os.path.relpath(ruta_exp, RAIZ_REPO)}")


if __name__ == "__main__":
    main()
