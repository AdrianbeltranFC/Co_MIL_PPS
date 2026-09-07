"""
=========================================================================================
SEGMENTACIÓN CON CARACTERÍSTICAS DE FOUNDATION MODEL  (DINOv2 congelado + cabeza ligera)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/segmentacion/dinov2_seg.py
               python CO-MIL/segmentacion/dinov2_seg.py --backbone vit_base_patch14_dinov2 --epocas 300

Experimento A (ambición de frontera, 7-sep). El límite de todo el proyecto es el
extractor: MobileNetV2 congelado (ImageNet) no entiende tejido de herida. DINOv2
(Meta, auto-supervisado sobre 142 M de imágenes) transfiere mucho mejor.

Aquí: DINOv2 CONGELADO -> tokens de parche (rejilla 16x16, 384-D en el modelo
small) -> decodificador convolucional LIGERO (solo esto entrena) -> máscara de
4 clases a 256 px. Es la receta "features de foundation model + cabeza ligera",
la más eficiente en anotación que hay.

Comparar contra el baseline: FPN + MobileNetV2 congelado, Dice medio 0.71 (6
semillas). Mismo split oficial de DFUTissue, misma pérdida (Dice + entropía
cruzada), misma evaluación.
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
from torch.utils.data import DataLoader

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.append(_DIR)

from dataset_seg import CLASES, NUM_CLASES, DFUTissueSeg, colorear
from entrenar_seg import dice_iou_desde_confusion, matriz_confusion

RAIZ_REPO = os.path.dirname(os.path.dirname(_DIR))
DIR_PESOS = os.path.join(RAIZ_REPO, "Pesos_Entrenados")


class ExtractorDINOv2(nn.Module):
    """DINOv2 congelado -> mapa de tokens de parche [B, D, g, g]."""

    def __init__(self, nombre="vit_small_patch14_dinov2", entrada=224):
        super().__init__()
        import timm
        self.vit = timm.create_model(nombre + ".lvd142m", pretrained=True, num_classes=0,
                                     img_size=entrada, dynamic_img_size=True)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.dim = self.vit.embed_dim
        self.entrada = entrada
        self.g = entrada // self.vit.patch_embed.patch_size[0]

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-1] != self.entrada:
            x = F.interpolate(x, self.entrada, mode="bilinear", align_corners=False)
        f = self.vit.forward_features(x)
        pt = f[:, self.vit.num_prefix_tokens:, :]                 # [B, g*g, D]
        b, n, d = pt.shape
        return pt.transpose(1, 2).reshape(b, d, self.g, self.g)   # [B, D, g, g]


class DecoderLigero(nn.Module):
    """g x g -> 256 x 256 (x16) con 4 bloques conv + upsample. ~0.9 M parámetros."""

    def __init__(self, dim, n_clases=NUM_CLASES, canales=(192, 96, 48, 32)):
        super().__init__()
        self.proj = nn.Conv2d(dim, canales[0], 1)
        cin = canales[0]
        bloques = []
        for c in canales:
            bloques += [
                nn.Conv2d(cin, c, 3, padding=1), nn.GroupNorm(min(16, c), c), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
            cin = c
        self.up = nn.Sequential(*bloques)
        self.head = nn.Conv2d(cin, n_clases, 1)

    def forward(self, x):
        return self.head(self.up(self.proj(x)))


class SegDINOv2(nn.Module):
    def __init__(self, backbone="vit_small_patch14_dinov2", entrada=224):
        super().__init__()
        self.enc = ExtractorDINOv2(backbone, entrada)
        self.dec = DecoderLigero(self.enc.dim)

    def forward(self, x):
        return self.dec(self.enc(x))


@torch.no_grad()
def evaluar(modelo, loader, dispositivo):
    modelo.eval()
    cm = np.zeros((NUM_CLASES, NUM_CLASES), dtype=np.int64)
    por_imagen = []
    for x, y in loader:
        logits = modelo(x.to(dispositivo))
        if logits.shape[-1] != y.shape[-1]:
            logits = F.interpolate(logits, y.shape[-1], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu()
        for b in range(pred.shape[0]):
            cmi = matriz_confusion(pred[b:b + 1], y[b:b + 1], NUM_CLASES)
            cm += cmi
            d, _, pres = dice_iou_desde_confusion(cmi)
            ct = [c for c in (1, 2, 3) if pres[c] or cmi[:, c].sum() > 0]
            if ct:
                por_imagen.append(float(np.mean([d[c] for c in ct])))
    dice, iou, presente = dice_iou_desde_confusion(cm)
    return {
        "dice_por_clase": {CLASES[c]: float(dice[c]) for c in range(NUM_CLASES)},
        "iou_por_clase": {CLASES[c]: float(iou[c]) for c in range(NUM_CLASES)},
        "clase_presente_en_gt": {CLASES[c]: bool(presente[c]) for c in range(NUM_CLASES)},
        "dice_medio_tejidos": float(np.mean([dice[c] for c in (1, 2, 3)])),
        "iou_medio_tejidos": float(np.mean([iou[c] for c in (1, 2, 3)])),
        "dice_medio_por_imagen": float(np.mean(por_imagen)) if por_imagen else 0.0,
    }


def entrenar(backbone="vit_small_patch14_dinov2", epocas=250, batch=8, lr=1e-3,
             paciencia=45, semilla=42, aug_fuerte=False):
    torch.manual_seed(semilla); np.random.seed(semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Segmentación DINOv2 ({backbone}) congelado + decoder ligero | {dispositivo.type.upper()} ===")

    ds_tr = DFUTissueSeg("train", augment=True, aug_fuerte=aug_fuerte)
    ds_va = DFUTissueSeg("val", augment=False)
    ds_te = DFUTissueSeg("test", augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False)

    modelo = SegDINOv2(backbone).to(dispositivo)
    entrenables = [p for p in modelo.parameters() if p.requires_grad]
    print(f"    entrenables: {sum(p.numel() for p in entrenables) / 1e6:.2f} M  "
          f"(extractor congelado: {sum(p.numel() for p in modelo.enc.parameters()) / 1e6:.1f} M)")

    ce = nn.CrossEntropyLoss()
    try:
        import segmentation_models_pytorch as smp
        dl = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        fn = lambda lg, y: 0.5 * dl(lg, y) + 0.5 * ce(lg, y)
    except ImportError:
        fn = ce
    opt = torch.optim.AdamW(entrenables, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=15, min_lr=1e-6)

    ruta = os.path.join(DIR_PESOS, "dinov2_seg_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta, exist_ok=True)
    mejor, mejor_ep, t0 = -1.0, -1, time.time()
    for epoca in range(1, epocas + 1):
        modelo.dec.train()
        for x, y in dl_tr:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x)
            if logits.shape[-1] != y.shape[-1]:
                logits = F.interpolate(logits, y.shape[-1], mode="bilinear", align_corners=False)
            opt.zero_grad(); fn(logits, y).backward(); opt.step()
        v = evaluar(modelo, dl_va, dispositivo)["dice_medio_tejidos"]
        sched.step(v)
        if v > mejor:
            mejor, mejor_ep = v, epoca
            torch.save({"dec_state": modelo.dec.state_dict(), "backbone": backbone}, os.path.join(ruta, "mejor.pth"))
        if epoca % 10 == 0 or v == mejor:
            print(f"época {epoca:3d}/{epocas} | val Dice(tejidos) {v:.4f}" + ("  <- mejor" if v == mejor else ""))
        if epoca - mejor_ep >= paciencia:
            print("early stopping"); break

    modelo.dec.load_state_dict(torch.load(os.path.join(ruta, "mejor.pth"))["dec_state"])
    te = evaluar(modelo, dl_te, dispositivo)
    print(f"\n=== TEST (mejor época {mejor_ep}) ===")
    for c in CLASES:
        print(f"  {c:14s} Dice {te['dice_por_clase'][c]:.3f}  IoU {te['iou_por_clase'][c]:.3f}")
    print(f"  {'MEDIA tejidos':14s} Dice {te['dice_medio_tejidos']:.3f}  IoU {te['iou_medio_tejidos']:.3f}")
    print(f"\n  vs. FPN + MobileNetV2 congelado (receta fuerte, 6 semillas): Dice medio 0.710 ± 0.019")

    meta = {"tarea": "segmentacion_dinov2_congelado", "backbone": backbone,
            "entrenables_M": round(sum(p.numel() for p in entrenables) / 1e6, 3),
            "epocas_corridas": epoca, "mejor_epoca": mejor_ep, "semilla": semilla,
            "aug_fuerte": aug_fuerte, "minutos": round((time.time() - t0) / 60, 1),
            "test": te, "referencia_fpn_mnv2": 0.710}
    json.dump(meta, open(os.path.join(ruta, "metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    _muestras(modelo, ds_te, dispositivo, ruta)
    print(f"\n[+] {os.path.relpath(ruta, RAIZ_REPO)}")
    return ruta, te


def _muestras(modelo, ds, dispositivo, ruta, n=8):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    modelo.eval()
    n = min(n, len(ds))
    fig, ax = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    _M = np.array([0.485, 0.456, 0.406]); _S = np.array([0.229, 0.224, 0.225])
    with torch.no_grad():
        for i in range(n):
            x, y = ds[i]
            lg = modelo(x.unsqueeze(0).to(dispositivo))
            lg = F.interpolate(lg, y.shape[-1], mode="bilinear", align_corners=False)
            pred = lg.argmax(1)[0].cpu().numpy()
            img = (x.numpy().transpose(1, 2, 0) * _S + _M).clip(0, 1)
            for j, (dato, tit) in enumerate([(img, "imagen"),
                                             (colorear(y.numpy()) / 255, "anotación"),
                                             (colorear(pred) / 255, "predicción")]):
                a = ax[i, j] if n > 1 else ax[j]
                a.imshow(dato); a.axis("off")
                if i == 0:
                    a.set_title(tit, fontsize=10)
    fig.suptitle("DINOv2 congelado + decoder ligero — DFUTissue test", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ruta, "muestras_test.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="vit_small_patch14_dinov2",
                   choices=["vit_small_patch14_dinov2", "vit_base_patch14_dinov2"])
    p.add_argument("--epocas", type=int, default=250)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--paciencia", type=int, default=45)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--aug_fuerte", action="store_true")
    a = p.parse_args()
    entrenar(backbone=a.backbone, epocas=a.epocas, batch=a.batch, lr=a.lr,
             paciencia=a.paciencia, semilla=a.semilla, aug_fuerte=a.aug_fuerte)
