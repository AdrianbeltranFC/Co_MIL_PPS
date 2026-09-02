"""
=========================================================================================
BASELINE SUPERVISADO DE SEGMENTACIÓN DE TEJIDOS  (FPN + MobileNetV2, DFUTissue)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/segmentacion/entrenar_seg.py
               python CO-MIL/segmentacion/entrenar_seg.py --epocas 200 --arch Unet --encoder mobilenet_v2

Qué hace (tarea T2 del plan post-27-ago, ver bitácora Parte IV):
  Entrena una cabeza de segmentación semántica ligera sobre un extractor MobileNetV2
  preentrenado en ImageNet, con las 78 imágenes de entrenamiento de DFUTissue, y la
  evalúa sobre las 16 de prueba (partición oficial). Es el "brazo supervisado" / cota
  de referencia contra el que luego se comparan el brazo débil (MIL) y el mixto.
  Métrica: Dice e IoU por clase de tejido (Fibrina, Granulación, Callo), agregadas
  sobre todo el split de prueba y también promediadas por imagen.

Referencias de magnitud (para saber si el resultado es sano):
  - DFUTissue original (Dhar et al. 2024, Unet + MiT-b3, 500 épocas): Dice medio ~0.85.
  - Kabir et al. 2025 (FPN + VGG16, LOOCV, dataset distinto pero misma tarea): Dice ~0.82.
  Con MobileNetV2 (4.2M parámetros, pensado para embebido) y CPU se espera algo menor;
  el objetivo aquí es un baseline reproducible y desplegable, no ganar el benchmark.
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
from torch.utils.data import DataLoader

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.append(_DIR)

import segmentation_models_pytorch as smp
from dataset_seg import CLASES, NUM_CLASES, DFUTissueSeg, colorear

RAIZ_REPO = os.path.dirname(os.path.dirname(_DIR))
DIR_PESOS = os.path.join(RAIZ_REPO, "Pesos_Entrenados")


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def matriz_confusion(pred: torch.Tensor, gt: torch.Tensor, n: int) -> np.ndarray:
    """pred, gt: [B,H,W] enteros. Devuelve matriz n x n acumulada (gt filas, pred cols)."""
    k = (gt >= 0) & (gt < n)
    idx = n * gt[k].to(torch.int64) + pred[k].to(torch.int64)
    return torch.bincount(idx, minlength=n * n).reshape(n, n).cpu().numpy()


def dice_iou_desde_confusion(cm: np.ndarray):
    """Dice e IoU por clase a partir de la matriz de confusión pooled."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    presente = (cm.sum(1) > 0)  # clase con píxeles en el ground truth del split
    return dice, iou, presente


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def construir_modelo(arch: str, encoder: str):
    fn = {"FPN": smp.FPN, "Unet": smp.Unet, "UnetPlusPlus": smp.UnetPlusPlus,
          "DeepLabV3Plus": smp.DeepLabV3Plus}[arch]
    return fn(encoder_name=encoder, encoder_weights="imagenet", classes=NUM_CLASES, activation=None)


def evaluar(modelo, loader, dispositivo):
    modelo.eval()
    cm = np.zeros((NUM_CLASES, NUM_CLASES), dtype=np.int64)
    dice_por_imagen = []  # media sobre clases de tejido presentes en cada imagen
    with torch.no_grad():
        for x, y in loader:
            logits = modelo(x.to(dispositivo))
            pred = logits.argmax(1).cpu()
            for b in range(pred.shape[0]):
                cmi = matriz_confusion(pred[b:b + 1], y[b:b + 1], NUM_CLASES)
                cm += cmi
                d, _, pres = dice_iou_desde_confusion(cmi)
                clases_tejido = [c for c in (1, 2, 3) if pres[c] or cmi[:, c].sum() > 0]
                if clases_tejido:
                    dice_por_imagen.append(float(np.mean([d[c] for c in clases_tejido])))
    dice, iou, presente = dice_iou_desde_confusion(cm)
    return {
        "dice_por_clase": {CLASES[c]: float(dice[c]) for c in range(NUM_CLASES)},
        "iou_por_clase": {CLASES[c]: float(iou[c]) for c in range(NUM_CLASES)},
        "clase_presente_en_gt": {CLASES[c]: bool(presente[c]) for c in range(NUM_CLASES)},
        "dice_medio_tejidos": float(np.mean([dice[c] for c in (1, 2, 3)])),
        "iou_medio_tejidos": float(np.mean([iou[c] for c in (1, 2, 3)])),
        "dice_medio_por_imagen": float(np.mean(dice_por_imagen)) if dice_por_imagen else 0.0,
        "dice_std_por_imagen": float(np.std(dice_por_imagen)) if dice_por_imagen else 0.0,
    }


def entrenar(arch="FPN", encoder="mobilenet_v2", epocas=200, batch=8, lr=1e-4,
             paciencia=40, semilla=42):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Segmentación {arch}+{encoder} | dispositivo: {dispositivo.type.upper()} ===")

    ds_tr = DFUTissueSeg("train", augment=True)
    ds_va = DFUTissueSeg("val", augment=False)
    ds_te = DFUTissueSeg("test", augment=False)
    print(f"train {len(ds_tr)} | val {len(ds_va)} | test {len(ds_te)}")

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=0)

    modelo = construir_modelo(arch, encoder).to(dispositivo)

    # Pérdida: Dice (robusta al desbalance -- el fondo es ~90% de los píxeles) + CE.
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    ce_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=15, min_lr=1e-6)

    ruta_exp = os.path.join(DIR_PESOS, "seg_exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                            + f"_{arch}-{encoder}")
    os.makedirs(ruta_exp, exist_ok=True)

    mejor_val = -1.0
    mejor_epoca = -1
    historial = []
    t0 = time.time()

    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_acum = 0.0
        for x, y in dl_tr:
            x, y = x.to(dispositivo), y.to(dispositivo)
            opt.zero_grad()
            logits = modelo(x)
            loss = 0.5 * dice_loss(logits, y) + 0.5 * ce_loss(logits, y)
            loss.backward()
            opt.step()
            perdida_acum += loss.item()
        perdida_media = perdida_acum / len(dl_tr)

        met_va = evaluar(modelo, dl_va, dispositivo)
        val_score = met_va["dice_medio_tejidos"]
        sched.step(val_score)
        historial.append({"epoca": epoca, "loss_train": perdida_media,
                          "val_dice_tejidos": val_score})

        marca = ""
        if val_score > mejor_val:
            mejor_val, mejor_epoca = val_score, epoca
            torch.save({"model_state_dict": modelo.state_dict(), "arch": arch,
                        "encoder": encoder, "epoca": epoca, "clases": CLASES},
                       os.path.join(ruta_exp, "mejor_modelo.pth"))
            marca = "  <- mejor"
        print(f"época {epoca:03d}/{epocas} | loss {perdida_media:.4f} | "
              f"val Dice(tejidos) {val_score:.4f}{marca}")

        if epoca - mejor_epoca >= paciencia:
            print(f"early stopping (sin mejora en {paciencia} épocas)")
            break

    # --- Evaluación final sobre test con el mejor modelo ---
    ckpt = torch.load(os.path.join(ruta_exp, "mejor_modelo.pth"), map_location=dispositivo)
    modelo.load_state_dict(ckpt["model_state_dict"])
    met_te = evaluar(modelo, dl_te, dispositivo)
    met_va_final = evaluar(modelo, dl_va, dispositivo)

    print("\n=== RESULTADO EN TEST (mejor modelo, época %d) ===" % mejor_epoca)
    for c in CLASES:
        pres = met_te["clase_presente_en_gt"][c]
        print(f"  {c:12s}  Dice {met_te['dice_por_clase'][c]:.3f}  "
              f"IoU {met_te['iou_por_clase'][c]:.3f}   {'' if pres else '(sin positivos en test)'}")
    print(f"  {'MEDIA tejidos':12s}  Dice {met_te['dice_medio_tejidos']:.3f}  "
          f"IoU {met_te['iou_medio_tejidos']:.3f}")
    print(f"  Dice medio por imagen: {met_te['dice_medio_por_imagen']:.3f} "
          f"± {met_te['dice_std_por_imagen']:.3f}")

    # --- Guardar metadata + visualizaciones ---
    metadata = {
        "tarea": "segmentacion_semantica_tejidos",
        "dataset": "DFUTissue (Padded, particion oficial 78/16/16)",
        "arquitectura": f"{arch} + {encoder} (encoder ImageNet)",
        "parametros_M": round(sum(p.numel() for p in modelo.parameters()) / 1e6, 3),
        "clases": CLASES,
        "epocas_max": epocas, "epocas_corridas": len(historial),
        "mejor_epoca": mejor_epoca, "batch": batch, "lr": lr, "semilla": semilla,
        "dispositivo": dispositivo.type,
        "minutos_entrenamiento": round((time.time() - t0) / 60, 1),
        "val": met_va_final, "test": met_te,
        "referencias_magnitud": {
            "DFUTissue_Unet_MiT-b3": "Dice medio ~0.85",
            "Kabir2025_FPN_VGG16_LOOCV": "Dice ~0.82",
        },
        "guardado_en": datetime.now().isoformat(),
    }
    with open(os.path.join(ruta_exp, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ruta_exp, "historial.json"), "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    _guardar_visualizaciones(modelo, ds_te, dispositivo, ruta_exp, n=8)

    print(f"\n[+] Experimento guardado en: {os.path.relpath(ruta_exp, RAIZ_REPO)}")
    return ruta_exp, met_te


def _guardar_visualizaciones(modelo, ds, dispositivo, ruta_exp, n=8):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    modelo.eval()
    n = min(n, len(ds))
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    _MEAN = np.array([0.485, 0.456, 0.406]); _STD = np.array([0.229, 0.224, 0.225])
    with torch.no_grad():
        for i in range(n):
            x, y = ds[i]
            pred = modelo(x.unsqueeze(0).to(dispositivo)).argmax(1)[0].cpu().numpy()
            img = (x.numpy().transpose(1, 2, 0) * _STD + _MEAN).clip(0, 1)
            for j, (dato, titulo) in enumerate([
                (img, "imagen"), (colorear(y.numpy()) / 255, "anotación"),
                (colorear(pred) / 255, "predicción")]):
                ax = axes[i, j] if n > 1 else axes[j]
                ax.imshow(dato); ax.axis("off")
                if i == 0:
                    ax.set_title(titulo, fontsize=10)
    fig.suptitle("DFUTissue -- test (rojo=Fibrina, verde=Granulación, azul=Callo)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ruta_exp, "muestras_test.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="FPN", choices=["FPN", "Unet", "UnetPlusPlus", "DeepLabV3Plus"])
    p.add_argument("--encoder", default="mobilenet_v2")
    p.add_argument("--epocas", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--paciencia", type=int, default=40)
    args = p.parse_args()
    entrenar(arch=args.arch, encoder=args.encoder, epocas=args.epocas, batch=args.batch,
             lr=args.lr, paciencia=args.paciencia)
