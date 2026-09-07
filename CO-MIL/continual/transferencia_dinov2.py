"""
=========================================================================================
EL SALTO DE DOMINIO CON CARACTERÍSTICAS DE FOUNDATION MODEL (DINOv2)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE.
CÓMO EJECUTAR:  python CO-MIL/continual/transferencia_dinov2.py

Experimento A (7-sep). Repite `transferencia.py` (el salto DFUTissue -> lote
mexicano) pero cambiando el extractor congelado: MobileNetV2-ImageNet -> DINOv2
(auto-supervisado, 142 M de imágenes). Hipótesis: DINOv2 transfiere mejor entre
poblaciones -> el salto se reduce.

Con MobileNetV2: F1 DFUTissue 0.74 -> F1 mexicano 0.30 (salto 0.44).
                 naive secuencial: DFUTissue 0.74 -> 0.02 (olvido catastrófico).
                 latent replay: DFUTissue 0.76.
=========================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
_SEG = os.path.join(os.path.dirname(_DIR), "segmentacion")
_MODELS = os.path.join(os.path.dirname(_DIR), "Models")
for _p in (_DIR, _SEG, _MODELS):
    if _p not in sys.path:
        sys.path.append(_p)

from attention_mil import AttentionAggregator, BagClassifier
from dataset_seg import DFUTissueSeg
from datos_continual import (DIR_CACHE, RAIZ_REPO, _MEAN, _MEX_A_DFU, _STD, _reconstruir_roi)
from estrategias import entrenar_cabeza, evaluar_cabeza
import torch.nn as nn

DIR_SALIDA = os.path.join(RAIZ_REPO, "Pesos_Entrenados", "continual")
RUTA_DFU = os.path.join(DIR_CACHE, "dinov2_dfutissue.pt")
RUTA_MEX = os.path.join(DIR_CACHE, "dinov2_mexicano.pt")
MIN_PIXELES = 64


class CabezaMIL(nn.Module):
    def __init__(self, dim, num_clases=3):
        super().__init__()
        self.attention = AttentionAggregator(L=dim, num_classes=num_clases)
        self.classifier = BagClassifier(L=dim, num_classes=num_clases)

    def forward(self, H, mask):
        z, a = self.attention(H, mask)
        return self.classifier(z), a


def _dinov2(dispositivo, entrada=224):
    import timm
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                          img_size=entrada, dynamic_img_size=True).to(dispositivo).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def _tokens(vit, x, dispositivo, entrada=224):
    if x.shape[-1] != entrada:
        x = F.interpolate(x, entrada, mode="bilinear", align_corners=False)
    f = vit.forward_features(x.to(dispositivo))
    return f[:, vit.num_prefix_tokens:, :].cpu()          # [B, 256, 384]


def cachear(dispositivo, forzar=False):
    if os.path.exists(RUTA_DFU) and os.path.exists(RUTA_MEX) and not forzar:
        return
    os.makedirs(DIR_CACHE, exist_ok=True)
    vit = _dinov2(dispositivo)

    # --- DFUTissue ---
    Hs, Ys = [], []
    for split in ("train", "val", "test"):
        ds = DFUTissueSeg(split, augment=False)
        for i in range(len(ds)):
            x, y = ds[i]
            Hs.append(_tokens(vit, x.unsqueeze(0), dispositivo)[0])
            pres = torch.zeros(3)
            for c in (1, 2, 3):
                if int((y == c).sum()) >= MIN_PIXELES:
                    pres[c - 1] = 1.0
            Ys.append(pres)
    torch.save({"X": torch.stack(Hs), "Y": torch.stack(Ys)}, RUTA_DFU)
    print(f"[dinov2-dfu] {len(Hs)} bolsas X={tuple(torch.stack(Hs).shape)}")

    # --- lote mexicano ---
    import glob
    from PIL import Image as _Im
    patron = os.path.join(RAIZ_REPO, "APP_generador_bolsas", "**",
                          "Bolsas_MIL_Procesadas", "224px", "*__roi_*.pt")
    Hm, Ym = [], []
    for f in sorted(glob.glob(patron, recursive=True)):
        d = torch.load(f, weights_only=False)
        meta = d["spatial_metadata"]
        img = _reconstruir_roi(d["X"], meta["grid_shape"], meta["patch_size"]).clamp(0, 1)
        arr = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        arr = np.asarray(_Im.fromarray(arr).resize((224, 224), _Im.BILINEAR))
        x = torch.from_numpy(np.ascontiguousarray(
            ((arr.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)))
        Hm.append(_tokens(vit, x.unsqueeze(0), dispositivo)[0])
        y12 = d["Y"].float()
        Ym.append(torch.tensor([float(y12[i]) for i in _MEX_A_DFU]))
    torch.save({"X": torch.stack(Hm), "Y": torch.stack(Ym)}, RUTA_MEX)
    print(f"[dinov2-mex] {len(Hm)} ROIs X={tuple(torch.stack(Hm).shape)} "
          f"positivos/clase {torch.stack(Ym).sum(0).tolist()}")


def _f1(modelo, X, Y, disp):
    m = torch.ones(len(X), X.shape[1], dtype=torch.bool)
    return evaluar_cabeza(modelo, X, m, Y, disp)["f1_macro"]


def _entrenar(modelo, X, Y, disp, epocas):
    M = torch.ones(len(X), X.shape[1], dtype=torch.bool)
    C = torch.ones(len(X), 3, dtype=torch.bool)
    npos = Y.sum(0).clamp(min=1.0)
    entrenar_cabeza(modelo, X, M, Y, C, epocas=epocas, dispositivo=disp,
                    pos_weight=(len(Y) - npos) / npos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", default="42,1,7,2,3")
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--frac_test_mex", type=float, default=0.4)
    p.add_argument("--forzar_cache", action="store_true")
    args = p.parse_args()
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semillas = [int(s) for s in args.semillas.split(",")]

    cachear(disp, forzar=args.forzar_cache)
    dfu = torch.load(RUTA_DFU); mex = torch.load(RUTA_MEX)
    Xd, Yd, Xm, Ym = dfu["X"], dfu["Y"], mex["X"], mex["Y"]
    dim = Xd.shape[-1]
    dfu_tr, dfu_te = np.arange(78), np.arange(94, len(Xd))
    X = torch.cat([Xd, Xm]); Y = torch.cat([Yd, Ym]); off = len(Xd)
    print(f"DINOv2-small | dim {dim} | DFUTissue {len(Xd)} (tr {len(dfu_tr)}/te {len(dfu_te)}) | "
          f"mexicano {len(Xm)}\n")

    salto, runs = [], []
    for s in semillas:
        rng = np.random.default_rng(s)
        idxm = off + rng.permutation(len(Xm))
        nte = int(round(len(Xm) * args.frac_test_mex))
        mex_te, mex_tr = idxm[:nte], idxm[nte:]

        torch.manual_seed(s)
        modelo = CabezaMIL(dim).to(disp)
        _entrenar(modelo, X[dfu_tr], Y[dfu_tr], disp, args.epocas)
        fd = _f1(modelo, X[dfu_te], Y[dfu_te], disp)
        fm = _f1(modelo, X[mex_te], Y[mex_te], disp)
        salto.append((s, fd, fm))
        print(f"[semilla {s}] DFUTissue -> F1 DFUTissue {fd:.3f} | F1 mexicano {fm:.3f}  (salto {fd - fm:+.3f})")

        # continual: naive y latent replay (búfer del vector medio, 10/clase)
        for modo in ("naive", "latente"):
            torch.manual_seed(s)
            modelo = CabezaMIL(dim).to(disp)
            _entrenar(modelo, X[dfu_tr], Y[dfu_tr], disp, args.epocas)
            R00, R01 = _f1(modelo, X[dfu_te], Y[dfu_te], disp), _f1(modelo, X[mex_te], Y[mex_te], disp)
            if modo == "latente":
                buf_idx, Yb = [], Y[dfu_tr]
                for c in range(3):
                    pos = dfu_tr[Yb[:, c].numpy() == 1]; rng.shuffle(pos)
                    buf_idx += list(pos[:10])
                buf_idx = sorted(set(buf_idx))
                Hbuf = X[buf_idx].mean(1, keepdim=True).expand(-1, X.shape[1], -1)
                Xtr2 = torch.cat([X[mex_tr], Hbuf]); Ytr2 = torch.cat([Y[mex_tr], Y[buf_idx]])
            else:
                Xtr2, Ytr2 = X[mex_tr], Y[mex_tr]
            _entrenar(modelo, Xtr2, Ytr2, disp, args.epocas)
            R10, R11 = _f1(modelo, X[dfu_te], Y[dfu_te], disp), _f1(modelo, X[mex_te], Y[mex_te], disp)
            runs.append({"modo": modo, "semilla": s, "final_dfu": R10, "final_mex": R11,
                         "olvido_dfu": R00 - R10})

    sd = np.array([[a, b] for _, a, b in salto])
    print(f"\n=== EL SALTO con DINOv2 (media sobre {len(semillas)} semillas) ===")
    print(f"  F1 DFUTissue          {sd[:, 0].mean():.3f} ± {sd[:, 0].std():.3f}")
    print(f"  F1 mexicano           {sd[:, 1].mean():.3f} ± {sd[:, 1].std():.3f}")
    print(f"  salto                 {(sd[:, 0] - sd[:, 1]).mean():+.3f}   "
          f"(con MobileNetV2 era +0.44)")
    print("\n=== DFUTissue -> mexicano en secuencia ===")
    for modo in ("naive", "latente"):
        r = [x for x in runs if x["modo"] == modo]
        g = lambda k: np.mean([x[k] for x in r])
        print(f"  {modo:>8}  final DFUTissue {g('final_dfu'):.3f}  final mexicano {g('final_mex'):.3f}  "
              f"olvido DFU {g('olvido_dfu'):+.3f}")

    ruta = os.path.join(DIR_SALIDA, "dinov2_transfer_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ruta, exist_ok=True)
    json.dump({"salto": salto, "runs": runs, "config": vars(args)},
              open(os.path.join(ruta, "resultados.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[+] {os.path.relpath(ruta, RAIZ_REPO)}")


if __name__ == "__main__":
    main()
