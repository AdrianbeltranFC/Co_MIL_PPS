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
    def __init__(self, split: str, nombres_con_mascara: set, augment: bool):
        self.base = DFUTissueSeg(split, augment=augment)
        self.con_mascara = nombres_con_mascara

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
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
                      semilla, dispositivo, lambda_debil=1.0):
    torch.manual_seed(semilla)
    np.random.seed(semilla)

    ds_tr = DFUTissueMixto("train", nombres_mascara, augment=True)
    ds_va = DFUTissueSeg("val", augment=False)
    ds_te = DFUTissueSeg("test", augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
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
        "epocas_corridas": epoca,
        "mejor_epoca": mejor_epoca,
        "minutos": round((time.time() - t0) / 60, 1),
        "test": met_te,
        "val_dice_tejidos": mejor_val,
    }, mejor_state


def graficar_curva(resultados, ruta_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ns = [r["n_mascaras"] for r in resultados]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colores = {"Fibrina": "#C0392B", "Granulación": "#2E9E5B", "Callo": "#3B7DD8"}
    for clase in ["Fibrina", "Granulación", "Callo"]:
        ax.plot(ns, [r["test"]["dice_por_clase"][clase] for r in resultados],
                "o-", label=clase, color=colores[clase])
    ax.plot(ns, [r["test"]["dice_medio_tejidos"] for r in resultados],
            "s--", label="media tejidos", color="#222", lw=2)
    ax.set_xlabel("Nº de imágenes con máscara densa (de 78)")
    ax.set_ylabel("Dice en test")
    ax.set_title("Frontera de eficiencia de anotación — DFUTissue\n"
                 "(N=0: solo etiqueta de imagen · N=78: supervisión densa completa)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mascaras", default="0,5,10,20,40,78",
                   help="lista de N separada por comas")
    p.add_argument("--epocas", type=int, default=120)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--paciencia", type=int, default=28)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valores_n = [int(x) for x in args.mascaras.split(",")]
    etiquetas = etiquetas_de_imagen_por_nombre("train")

    ruta_exp = os.path.join(DIR_PESOS, "eficiencia_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta_exp, exist_ok=True)
    print(f"=== Curva de eficiencia | dispositivo {dispositivo.type.upper()} | N = {valores_n} ===")
    print(f"    salida: {os.path.relpath(ruta_exp, RAIZ_REPO)}\n")

    resultados = []
    for n in valores_n:
        nombres_m = elegir_subconjunto(n, etiquetas, args.semilla)
        print(f"[N={n:2d}]  ({len(nombres_m)} imágenes con máscara, {78 - len(nombres_m)} solo etiqueta)")
        res, state = entrenar_un_punto(n, nombres_m, args.epocas, args.batch, args.lr,
                                       args.paciencia, args.semilla, dispositivo)
        d = res["test"]
        print(f"       -> Dice medio tejidos {d['dice_medio_tejidos']:.3f}  "
              f"(Fib {d['dice_por_clase']['Fibrina']:.3f} | "
              f"Gra {d['dice_por_clase']['Granulación']:.3f} | "
              f"Cal {d['dice_por_clase']['Callo']:.3f})   [{res['minutos']} min]")
        torch.save({"model_state_dict": state, "n_mascaras": n},
                   os.path.join(ruta_exp, f"modelo_N{n:02d}.pth"))
        resultados.append(res)
        # guardado incremental
        with open(os.path.join(ruta_exp, "resultados.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "dispositivo": dispositivo.type,
                       "resultados": resultados}, f, ensure_ascii=False, indent=2)
        graficar_curva(resultados, os.path.join(ruta_exp, "curva_eficiencia.png"))

    print("\n=== RESUMEN ===")
    print(f"{'N':>4} | {'Fibrina':>8} {'Granul.':>8} {'Callo':>8} {'Media':>8}")
    for r in resultados:
        d = r["test"]["dice_por_clase"]
        print(f"{r['n_mascaras']:>4} | {d['Fibrina']:>8.3f} {d['Granulación']:>8.3f} "
              f"{d['Callo']:>8.3f} {r['test']['dice_medio_tejidos']:>8.3f}")
    print(f"\n[+] {os.path.relpath(ruta_exp, RAIZ_REPO)}")


if __name__ == "__main__":
    main()
