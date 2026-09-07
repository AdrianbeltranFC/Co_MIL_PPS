"""
=========================================================================================
SEGMENTACIÓN CON CARACTERÍSTICAS DE FOUNDATION MODEL  (DINOv2 congelado + cabeza ligera)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/segmentacion/dinov2_seg.py
               python CO-MIL/segmentacion/dinov2_seg.py --entrada 448 --perdida tversky \
                   --sobremuestreo --copias_aug 4 --semillas 42,1,7

Experimento A (ambición de frontera, 7-sep). El límite del proyecto es el
extractor: MobileNetV2 congelado (ImageNet) no entiende tejido de herida. DINOv2
(Meta, auto-supervisado sobre 142 M de imágenes) transfiere mejor.

DINOv2 CONGELADO -> tokens de parche (rejilla entrada/14) -> decodificador
convolucional LIGERO (lo único que entrena) -> máscara de 4 clases a 256 px.

Comparación JUSTA con el caballo de batalla (FPN + MobileNetV2, receta fuerte,
6 semillas -> Dice medio 0.710): mismas perillas (Tversky, sobre-muestreo de
fibrina, aumentación) y varias semillas. Las características de DINOv2 se
precalculan una vez (congeladas) con `copias_aug` versiones geométricas, y luego
el decoder entrena rápido sobre esa caché.
=========================================================================================
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.append(_DIR)

from dataset_seg import CLASES, NUM_CLASES, DFUTissueSeg, colorear
from entrenar_seg import dice_iou_desde_confusion, matriz_confusion

RAIZ_REPO = os.path.dirname(os.path.dirname(_DIR))
DIR_PESOS = os.path.join(RAIZ_REPO, "Pesos_Entrenados")
DIR_CACHE = os.path.join(DIR_PESOS, "dinov2_cache")


# ---------------------------------------------------------------------------
# Extractor congelado
# ---------------------------------------------------------------------------
def _cargar_dinov2(nombre, entrada, dispositivo):
    import timm
    m = timm.create_model(nombre + ".lvd142m", pretrained=True, num_classes=0,
                          img_size=entrada, dynamic_img_size=True).to(dispositivo).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def _mapa_tokens(vit, x, entrada, dispositivo):
    if x.shape[-1] != entrada:
        x = F.interpolate(x, entrada, mode="bilinear", align_corners=False)
    f = vit.forward_features(x.to(dispositivo))
    pt = f[:, vit.num_prefix_tokens:, :]
    b, n, d = pt.shape
    g = int(round(n ** 0.5))
    return pt.transpose(1, 2).reshape(b, d, g, g).cpu()


def _aug_geo(img, ann, k):
    """k: 0 = original; 1-3 = flip h / flip v / rot90. Imagen [3,H,W] y máscara [H,W] tensores."""
    if k == 1:
        return torch.flip(img, [-1]), torch.flip(ann, [-1])
    if k == 2:
        return torch.flip(img, [-2]), torch.flip(ann, [-2])
    if k == 3:
        return torch.rot90(img, 1, [-2, -1]), torch.rot90(ann, 1, [-2, -1])
    return img, ann


def cachear(nombre, entrada, copias_aug, dispositivo, forzar=False):
    os.makedirs(DIR_CACHE, exist_ok=True)
    ruta = os.path.join(DIR_CACHE, f"{nombre}_e{entrada}_a{copias_aug}.pt")
    if os.path.exists(ruta) and not forzar:
        return ruta
    vit = _cargar_dinov2(nombre, entrada, dispositivo)
    datos = {}
    for split in ("train", "val", "test"):
        ds = DFUTissueSeg(split, augment=False)
        F_list, Y_list = [], []
        n_copias = copias_aug if split == "train" else 1
        for i in range(len(ds)):
            x, y = ds[i]
            for k in range(n_copias):
                xk, yk = _aug_geo(x, y, k)
                F_list.append(_mapa_tokens(vit, xk.unsqueeze(0), entrada, dispositivo)[0])
                Y_list.append(yk)
        datos[split] = (torch.stack(F_list), torch.stack(Y_list))
        print(f"    [{split}] {len(F_list)} mapas  F={tuple(datos[split][0].shape)}")
    torch.save(datos, ruta)
    return ruta


# ---------------------------------------------------------------------------
# Decoder ligero
# ---------------------------------------------------------------------------
class DecoderLigero(nn.Module):
    def __init__(self, dim, g, salida=256, n_clases=NUM_CLASES):
        super().__init__()
        self.proj = nn.Conv2d(dim, 192, 1)
        n_up = max(1, int(round(np.log2(salida / g))))
        cin, bloques, c = 192, [], 192
        for _ in range(n_up):
            c = max(32, cin // 2)
            bloques += [nn.Conv2d(cin, c, 3, padding=1), nn.GroupNorm(min(16, c), c), nn.GELU(),
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)]
            cin = c
        self.up = nn.Sequential(*bloques)
        self.head = nn.Conv2d(cin, n_clases, 1)
        self.salida = salida

    def forward(self, x):
        y = self.head(self.up(self.proj(x)))
        if y.shape[-1] != self.salida:
            y = F.interpolate(y, self.salida, mode="bilinear", align_corners=False)
        return y


# ---------------------------------------------------------------------------
# Métrica
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluar(dec, F_te, Y_te, dispositivo, batch=8):
    dec.eval()
    cm = np.zeros((NUM_CLASES, NUM_CLASES), dtype=np.int64)
    for k in range(0, len(F_te), batch):
        logits = dec(F_te[k:k + batch].to(dispositivo))
        pred = logits.argmax(1).cpu()
        yb = Y_te[k:k + batch]
        if pred.shape[-1] != yb.shape[-1]:
            pred = F.interpolate(pred.unsqueeze(1).float(), yb.shape[-1]).squeeze(1).long()
        cm += matriz_confusion(pred, yb, NUM_CLASES)
    dice, iou, presente = dice_iou_desde_confusion(cm)
    return {"dice_por_clase": {CLASES[c]: float(dice[c]) for c in range(NUM_CLASES)},
            "iou_por_clase": {CLASES[c]: float(iou[c]) for c in range(NUM_CLASES)},
            "dice_medio_tejidos": float(np.mean([dice[c] for c in (1, 2, 3)])),
            "iou_medio_tejidos": float(np.mean([iou[c] for c in (1, 2, 3)]))}


def _perdida(nombre):
    import segmentation_models_pytorch as smp
    ce = nn.CrossEntropyLoss()
    if nombre == "tversky":
        tv = smp.losses.TverskyLoss(mode="multiclass", from_logits=True, alpha=0.3, beta=0.7)
        return lambda lg, y: 0.6 * tv(lg, y) + 0.4 * ce(lg, y)
    dl = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    return lambda lg, y: 0.5 * dl(lg, y) + 0.5 * ce(lg, y)


def entrenar_decoder(F_tr, Y_tr, F_va, Y_va, dim, g, semilla, epocas, lr, paciencia,
                     perdida, sobremuestreo, dispositivo):
    torch.manual_seed(semilla); np.random.seed(semilla)
    dec = DecoderLigero(dim, g).to(dispositivo)
    fn = _perdida(perdida)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=12, min_lr=1e-6)

    if sobremuestreo:
        from torch.utils.data import WeightedRandomSampler
        w = torch.tensor([3.0 if (y == 1).sum() >= 64 else 1.0 for y in Y_tr])
        sampler = WeightedRandomSampler(w, len(Y_tr), replacement=True)
        dl = DataLoader(TensorDataset(F_tr, Y_tr), batch_size=8, sampler=sampler)
    else:
        dl = DataLoader(TensorDataset(F_tr, Y_tr), batch_size=8, shuffle=True)

    mejor, mejor_ep, estado = -1.0, -1, None
    for epoca in range(1, epocas + 1):
        dec.train()
        for fb, yb in dl:
            fb, yb = fb.to(dispositivo), yb.to(dispositivo)
            opt.zero_grad(); fn(dec(fb), yb).backward(); opt.step()
        v = evaluar(dec, F_va, Y_va, dispositivo)["dice_medio_tejidos"]
        sched.step(v)
        if v > mejor:
            mejor, mejor_ep = v, epoca
            estado = {k: t.cpu().clone() for k, t in dec.state_dict().items()}
        if epoca - mejor_ep >= paciencia:
            break
    dec.load_state_dict(estado)
    return dec, mejor_ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="vit_small_patch14_dinov2",
                   choices=["vit_small_patch14_dinov2", "vit_base_patch14_dinov2"])
    p.add_argument("--entrada", type=int, default=336, help="múltiplo de 14 (224/336/448)")
    p.add_argument("--copias_aug", type=int, default=4, help="versiones geométricas por imagen de train")
    p.add_argument("--perdida", default="tversky", choices=["dicece", "tversky"])
    p.add_argument("--sobremuestreo", action="store_true", default=True)
    p.add_argument("--epocas", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--paciencia", type=int, default=30)
    p.add_argument("--semillas", default="42,1,7")
    p.add_argument("--forzar_cache", action="store_true")
    args = p.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semillas = [int(s) for s in args.semillas.split(",")]
    print(f"=== DINOv2 seg | {args.backbone} entrada {args.entrada} | pérdida {args.perdida} "
          f"sobremuestreo {args.sobremuestreo} | {dispositivo.type.upper()} ===")

    t0 = time.time()
    ruta_cache = cachear(args.backbone, args.entrada, args.copias_aug, dispositivo, args.forzar_cache)
    datos = torch.load(ruta_cache)
    (F_tr, Y_tr), (F_va, Y_va), (F_te, Y_te) = datos["train"], datos["val"], datos["test"]
    dim, g = F_tr.shape[1], F_tr.shape[-1]
    print(f"    caché: {tuple(F_tr.shape)}  dim {dim}  rejilla {g}x{g}  ({time.time() - t0:.0f}s)")

    resultados = []
    for s in semillas:
        dec, mep = entrenar_decoder(F_tr, Y_tr, F_va, Y_va, dim, g, s, args.epocas, args.lr,
                                    args.paciencia, args.perdida, args.sobremuestreo, dispositivo)
        te = evaluar(dec, F_te, Y_te, dispositivo)
        resultados.append(te)
        d = te["dice_por_clase"]
        print(f"  semilla {s} (época {mep})  Dice medio {te['dice_medio_tejidos']:.3f}  "
              f"(Fib {d['Fibrina']:.2f} | Gra {d['Granulación']:.2f} | Cal {d['Callo']:.2f})")

    arr = np.array([[r["dice_por_clase"]["Fibrina"], r["dice_por_clase"]["Granulación"],
                     r["dice_por_clase"]["Callo"], r["dice_medio_tejidos"]] for r in resultados])
    print(f"\n=== DINOv2-small congelado + decoder ligero ({len(semillas)} semillas) ===")
    print(f"  Fibrina  {arr[:, 0].mean():.3f} ± {arr[:, 0].std():.3f}")
    print(f"  Granul.  {arr[:, 1].mean():.3f} ± {arr[:, 1].std():.3f}")
    print(f"  Callo    {arr[:, 2].mean():.3f} ± {arr[:, 2].std():.3f}")
    print(f"  MEDIA    {arr[:, 3].mean():.3f} ± {arr[:, 3].std():.3f}")
    print(f"  --- vs FPN + MobileNetV2 congelado, receta fuerte, 6 semillas: 0.710 ± 0.019 ---")

    ruta = os.path.join(DIR_PESOS, "dinov2_seg_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta, exist_ok=True)
    json.dump({"config": vars(args), "semillas": semillas,
               "media_tejidos": float(arr[:, 3].mean()), "std_tejidos": float(arr[:, 3].std()),
               "por_semilla": resultados, "referencia_fpn_mnv2": 0.710,
               "minutos": round((time.time() - t0) / 60, 1)},
              open(os.path.join(ruta, "metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[+] {os.path.relpath(ruta, RAIZ_REPO)}")


if __name__ == "__main__":
    main()
