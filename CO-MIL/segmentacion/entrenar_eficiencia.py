"""
=========================================================================================
CURVA DE EFICIENCIA DE ANOTACIÓN  (supervisión densa vs. débil vs. mixta)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/segmentacion/entrenar_eficiencia.py
               python CO-MIL/segmentacion/entrenar_eficiencia.py --mascaras 0,10,40,78 --epocas 120
               python CO-MIL/segmentacion/entrenar_eficiencia.py \
                   --mascaras 5,10,20,40,78 --semillas 42,1,7 \
                   --modos mixto,solo_supervisado --max_tejidos_debil none,1,2

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

Curvas comparables en una sola corrida (--modos, --max_tejidos_debil):
    - solo_supervisado : N máscaras y nada más (las 78-N restantes se ignoran).
    - mixto            : N máscaras + TODAS las (78-N) etiquetas débiles.
    - mixto  max1/max2 : N máscaras + solo las etiquetas débiles de imágenes con <=1
                         (o <=2) tejidos presentes.  Idea de Adrián (6-sep): una
                         etiqueta [1,1,1] no da señal discriminativa a la pérdida
                         LSE-pool+BCE; una [1,0,0] sí. Con menos tejidos por imagen la
                         señal débil es más informativa (aunque quedan menos imágenes).
    La BRECHA entre 'mixto' y 'solo_supervisado' = cuánto aporta de verdad la débil.
    La BRECHA entre 'mixto' y 'mixto max1/max2'   = si conviene filtrar la débil.

Notas:
  - Las etiquetas de imagen se derivan de las máscaras (qué clases tienen algún píxel):
    es gratis y consistente, y es exactamente lo que un clínico marcaría en segundos.
  - El subconjunto de N imágenes "con máscara" se elige de forma estratificada (cubrir
    las 3 clases de tejido) y con semilla fija, para que el barrido sea reproducible.
  - --dir_salida reutiliza una carpeta ya existente y REANUDA (salta las corridas ya
    hechas). Sirve para partir el estudio entre varias sesiones de Colab acumulando
    todo en la misma carpeta de Drive.
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
MIN_PIXELES_TEJIDO = 64    # una clase "está presente" si ocupa >= este nº de píxeles


# ---------------------------------------------------------------------------
# Identidad de cada curva (modo + filtro de la anotación débil)
# ---------------------------------------------------------------------------
def etiqueta_curva(modo: str, max_tejidos_debil) -> str:
    if modo == "solo_supervisado":
        return "solo_supervisado"
    if max_tejidos_debil is None:
        return "mixto"
    return f"mixto_max{max_tejidos_debil}"


_LBL_CURVA = {
    "solo_supervisado": "solo N máscaras (nada más)",
    "mixto": "N máscaras + todas las etiquetas débiles",
    "mixto_max1": "N máscaras + etiquetas débiles de ≤1 tejido",
    "mixto_max2": "N máscaras + etiquetas débiles de ≤2 tejidos",
}
_COL_CURVA = {
    "solo_supervisado": "#B4791A",
    "mixto": "#1C6B63",
    "mixto_max1": "#7B3FA0",
    "mixto_max2": "#3B7DD8",
}


# ---------------------------------------------------------------------------
# Dataset que expone: imagen, máscara, etiqueta de imagen, y si esta imagen
# "tiene máscara" en este experimento.
# ---------------------------------------------------------------------------
class DFUTissueMixto(Dataset):
    """`solo_supervisado=True` -> el dataset solo contiene las N imágenes con máscara
    (las demás ni se ven).
    `solo_supervisado=False, max_tejidos_debil=None` -> las 78, con una bandera de si
    tienen máscara.
    `solo_supervisado=False, max_tejidos_debil=k` -> las N con máscara + solo las
    imágenes SIN máscara que tienen <= k tejidos presentes (filtro del brazo débil)."""

    def __init__(self, split, nombres_con_mascara, augment, aug_fuerte=False,
                 solo_supervisado=False, max_tejidos_debil=None, conteo_tejidos=None):
        base = DFUTissueSeg(split, augment=augment, aug_fuerte=aug_fuerte)
        self.con_mascara = set(nombres_con_mascara)
        if solo_supervisado:
            self._idx = [i for i in range(len(base)) if base.nombres[i] in self.con_mascara]
        elif max_tejidos_debil is None:
            self._idx = list(range(len(base)))
        else:
            ct = conteo_tejidos or contar_tejidos_por_nombre(split)
            self._idx = [
                i for i in range(len(base))
                if base.nombres[i] in self.con_mascara
                or ct.get(base.nombres[i], 99) <= max_tejidos_debil
            ]
        self.base = base
        self.n_debil = sum(1 for i in self._idx if base.nombres[i] not in self.con_mascara)

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


def contar_tejidos_por_nombre(split: str, min_pixeles: int = MIN_PIXELES_TEJIDO) -> dict:
    """Nº de clases de tejido (1, 2, 3) con al menos `min_pixeles` píxeles en la máscara.
    Se usa un umbral en píxeles (no `np.unique`) para no contar como 'presente' una
    astilla de anotación de unos pocos píxeles."""
    ds = DFUTissueSeg(split)
    d = {}
    for i in range(len(ds)):
        _, ann = ds._cargar(i)
        d[ds.nombres[i]] = sum(1 for c in CLASES_TEJIDO
                               if int((ann == c).sum()) >= min_pixeles)
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
                      semilla, dispositivo, modo="mixto", aug_fuerte=False,
                      lambda_debil=1.0, max_tejidos_debil=None, conteo_tejidos=None):
    """modo:
       'mixto'            -> N máscaras (pérdida de segmentación) + (78-N) etiquetas de
                             imagen (pérdida débil).  Si max_tejidos_debil=k, solo se
                             usan como débiles las imágenes con <= k tejidos presentes.
       'solo_supervisado' -> N máscaras y nada más; las (78-N) restantes se IGNORAN.
       La diferencia entre las curvas = cuánto aporta de verdad la anotación débil
       (frente a «solo el modelo aprendiendo de N muestras»), y si conviene filtrarla."""
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    curva = etiqueta_curva(modo, max_tejidos_debil)

    ds_tr = DFUTissueMixto("train", nombres_mascara, augment=True, aug_fuerte=aug_fuerte,
                           solo_supervisado=(modo == "solo_supervisado"),
                           max_tejidos_debil=(None if modo == "solo_supervisado"
                                              else max_tejidos_debil),
                           conteo_tejidos=conteo_tejidos)
    if len(ds_tr) == 0:
        return {"n_mascaras": n_mascaras, "modo": modo, "curva": curva,
                "max_tejidos_debil": max_tejidos_debil, "n_debil": 0,
                "epocas_corridas": 0, "mejor_epoca": 0, "minutos": 0.0,
                "val_dice_tejidos": 0.0,
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
        "curva": curva,
        "max_tejidos_debil": max_tejidos_debil,
        "n_debil": ds_tr.n_debil,
        "epocas_corridas": epoca,
        "mejor_epoca": mejor_epoca,
        "minutos": round((time.time() - t0) / 60, 1),
        "test": met_te,
        "val_dice_tejidos": mejor_val,
    }, mejor_state


def agregar_por_n(runs):
    """Agrupa por (curva, N) y saca media±std sobre semillas."""
    from collections import defaultdict
    grupos = defaultdict(list)
    for r in runs:
        curva = r.get("curva") or etiqueta_curva(r.get("modo", "mixto"),
                                                 r.get("max_tejidos_debil"))
        grupos[(curva, r["n_mascaras"])].append(r)
    agregado = []
    for (curva, n) in sorted(grupos, key=lambda k: (k[0], k[1])):
        rr = grupos[(curva, n)]
        fila = {"curva": curva, "modo": rr[0].get("modo", "mixto"),
                "max_tejidos_debil": rr[0].get("max_tejidos_debil"),
                "n_mascaras": n, "n_semillas": len(rr),
                "n_debil_medio": float(np.mean([r.get("n_debil", 0) for r in rr])),
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
    curvas = sorted({a["curva"] for a in ag})
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    if set(curvas) == {"mixto"}:
        # vista por tejido (una sola curva)
        colores = {"Fibrina": "#C0392B", "Granulación": "#2E9E5B", "Callo": "#3B7DD8"}
        sub = sorted([a for a in ag if a["curva"] == "mixto"], key=lambda a: a["n_mascaras"])
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
        # comparación de curvas: media de tejidos, un color por curva
        for i, curva in enumerate(curvas):
            sub = sorted([a for a in ag if a["curva"] == curva], key=lambda a: a["n_mascaras"])
            ns = np.array([a["n_mascaras"] for a in sub])
            m = np.array([a["dice_medio_tejidos_media"] for a in sub])
            s = np.array([a["dice_medio_tejidos_std"] for a in sub])
            col = _COL_CURVA.get(curva, plt.cm.tab10(i % 10))
            ls = "--" if curva == "solo_supervisado" else "-"
            ax.plot(ns, m, "o", ls=ls, color=col, lw=2, label=_LBL_CURVA.get(curva, curva))
            ax.fill_between(ns, m - s, m + s, color=col, alpha=0.15)

    n_sem = max(a["n_semillas"] for a in ag)
    ax.set_xlabel("Nº de imágenes con máscara densa (de 78)")
    ax.set_ylabel("Dice medio en tejidos (test)")
    ax.set_title(f"Eficiencia de anotación — DFUTissue ({n_sem} semillas, banda = ±1σ)\n"
                 "brecha entre curvas = valor real de la anotación débil / del filtro")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _parse_filtro(tok: str):
    tok = tok.strip().lower()
    if tok in ("none", "", "all", "todas"):
        return None
    return int(tok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mascaras", default="0,5,10,20,40,78",
                   help="lista de N separada por comas")
    p.add_argument("--semillas", default="42",
                   help="lista de semillas separada por comas (varias -> curva con bandas de error)")
    p.add_argument("--modos", default="mixto",
                   help="'mixto', 'solo_supervisado', o ambos separados por coma. Con ambos, la "
                        "brecha entre curvas responde: ¿aporta algo la anotación débil?")
    p.add_argument("--max_tejidos_debil", default="none",
                   help="filtro del brazo DÉBIL: 'none' = todas las imágenes sin máscara; un "
                        "entero k = solo las imágenes sin máscara con <=k tejidos presentes. "
                        "Lista separada por comas -> una curva 'mixto' por valor (p. ej. "
                        "'none,1,2'). No afecta a 'solo_supervisado'.")
    p.add_argument("--aug_fuerte", action="store_true")
    p.add_argument("--epocas", type=int, default=120)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--paciencia", type=int, default=28)
    p.add_argument("--guardar_modelos", action="store_true",
                   help="guardar un .pth por corrida (por defecto no, para no llenar el disco)")
    p.add_argument("--dir_salida", default="",
                   help="carpeta de salida a REUTILIZAR (reanuda: salta las corridas ya hechas). "
                        "Para partir el estudio entre sesiones de Colab acumulando en la misma "
                        "carpeta de Drive. Por defecto crea una nueva eficiencia_<fecha>/.")
    args = p.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valores_n = [int(x) for x in args.mascaras.split(",")]
    semillas = [int(x) for x in args.semillas.split(",")]
    modos = [m.strip() for m in args.modos.split(",")]
    filtros_debil = [_parse_filtro(t) for t in args.max_tejidos_debil.split(",")]
    etiquetas = etiquetas_de_imagen_por_nombre("train")
    conteo_tej = contar_tejidos_por_nombre("train")

    ruta_exp = args.dir_salida or os.path.join(
        DIR_PESOS, "eficiencia_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta_exp, exist_ok=True)

    runs = []
    ruta_json = os.path.join(ruta_exp, "resultados.json")
    if os.path.exists(ruta_json):
        try:
            runs = json.load(open(ruta_json, encoding="utf-8")).get("runs", [])
            print(f"    reanudando: {len(runs)} corrida(s) previa(s) en {os.path.relpath(ruta_exp, RAIZ_REPO)}")
        except Exception:
            runs = []
    ya_hechas = {(r.get("curva"), r["n_mascaras"], r.get("semilla")) for r in runs}

    print(f"=== Curva de eficiencia | {dispositivo.type.upper()} | N = {valores_n} | "
          f"semillas = {semillas} | modos = {modos} | filtro débil = {filtros_debil} ===")
    print(f"    salida: {os.path.relpath(ruta_exp, RAIZ_REPO)}\n")

    for n in valores_n:
        for modo in modos:
            filtros = [None] if modo == "solo_supervisado" else filtros_debil
            for max_tej in filtros:
                if modo == "solo_supervisado" and n == 0:
                    continue  # sin imágenes no hay nada que entrenar
                curva = etiqueta_curva(modo, max_tej)
                for semilla in semillas:
                    if (curva, n, semilla) in ya_hechas:
                        print(f"[N={n:2d} {curva} semilla={semilla}]  ya hecho, se salta")
                        continue
                    nombres_m = elegir_subconjunto(n, etiquetas, semilla)
                    if modo == "mixto" and max_tej is not None:
                        n_deb = sum(1 for nom, c in conteo_tej.items()
                                    if nom not in nombres_m and c <= max_tej)
                        if n_deb == 0:
                            print(f"[N={n:2d} {curva} semilla={semilla}]  0 débiles tras el "
                                  f"filtro, se salta (idéntico a solo_supervisado)")
                            continue
                    print(f"[N={n:2d} {curva} semilla={semilla}]  ({len(nombres_m)} con máscara)")
                    res, state = entrenar_un_punto(
                        n, nombres_m, args.epocas, args.batch, args.lr, args.paciencia,
                        semilla, dispositivo, modo=modo, aug_fuerte=args.aug_fuerte,
                        max_tejidos_debil=max_tej, conteo_tejidos=conteo_tej)
                    res["semilla"] = semilla
                    d = res["test"]
                    print(f"       -> Dice medio tejidos {d['dice_medio_tejidos']:.3f}  "
                          f"(Fib {d['dice_por_clase']['Fibrina']:.3f} | "
                          f"Gra {d['dice_por_clase']['Granulación']:.3f} | "
                          f"Cal {d['dice_por_clase']['Callo']:.3f})   "
                          f"[{res.get('n_debil', 0)} débiles, {res['minutos']} min]")
                    if args.guardar_modelos and state is not None:
                        torch.save({"model_state_dict": state, "n_mascaras": n, "curva": curva,
                                    "modo": modo, "max_tejidos_debil": max_tej, "semilla": semilla},
                                   os.path.join(ruta_exp, f"modelo_{curva}_N{n:02d}_s{semilla}.pth"))
                    runs.append(res)
                    ya_hechas.add((curva, n, semilla))
                    with open(ruta_json, "w", encoding="utf-8") as f:
                        json.dump({"config": vars(args), "dispositivo": dispositivo.type,
                                   "runs": runs, "agregado_por_n": agregar_por_n(runs)},
                                  f, ensure_ascii=False, indent=2)
                    graficar_curva(runs, os.path.join(ruta_exp, "curva_eficiencia.png"))

    print("\n=== RESUMEN (media ± desv. estándar sobre semillas) ===")
    print(f"{'curva':>22} {'N':>3} {'nD':>4} | {'Fibrina':>13} {'Granul.':>13} {'Callo':>13} {'Media':>13}")
    for a in agregar_por_n(runs):
        def mm(c):
            return f"{a['dice_' + c + '_media']:.3f}±{a['dice_' + c + '_std']:.3f}"
        print(f"{a['curva']:>22} {a['n_mascaras']:>3} {a['n_debil_medio']:>4.0f} | "
              f"{mm('Fibrina'):>13} {mm('Granulación'):>13} {mm('Callo'):>13} "
              f"{a['dice_medio_tejidos_media']:.3f}±{a['dice_medio_tejidos_std']:.3f}")
    print(f"\n[+] {os.path.relpath(ruta_exp, RAIZ_REPO)}")


if __name__ == "__main__":
    main()
