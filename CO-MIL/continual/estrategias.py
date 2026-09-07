"""
=========================================================================================
CABEZA MIL Y ESTRATEGIAS DE APRENDIZAJE CONTINUO
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL.

La cabeza reutiliza los bloques de CO-MIL/Models/attention_mil.py (atención con
compuertas + clasificador por clase), sobre características ya cacheadas (el
extractor MobileNetV2 está congelado y fuera de este módulo).

Cada bolsa se representa como (H [N,1280], mask_inst [N] bool). Cada muestra lleva
además un mask_col [num_clases] bool que dice qué columnas de la etiqueta se
supervisan en ese paso:
  - modo DOMAIN-incremental  -> mask_col siempre todo verdadero (las 3 clases).
  - modo CLASS-incremental   -> en el paso k solo la columna de la clase k (y, en
    el búfer, las columnas con que se guardó cada exemplar).

Estrategias: naive (cota inferior) · joint (cota superior) · ExperienceReplay
(búfer de bolsas completas) · LatentReplay (búfer del vector medio 1280-D: 64x
menos almacenamiento, sin imágenes).
=========================================================================================
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(os.path.dirname(_DIR), "Models")
for _p in (_DIR, _MODELS):
    if _p not in sys.path:
        sys.path.append(_p)

from attention_mil import AttentionAggregator, BagClassifier  # noqa: E402

from datos_continual import DIM_FEATURES, N_INSTANCIAS, NUM_CLASES  # noqa: E402


class CabezaMIL(nn.Module):
    """Arquitectura CoMIL: K ramas de atención + K clasificadores independientes
    (uno por tejido). El aislamiento por clase la hace robusta al olvido."""

    def __init__(self, dim=DIM_FEATURES, num_clases=NUM_CLASES):
        super().__init__()
        self.attention = AttentionAggregator(L=dim, num_classes=num_clases)
        self.classifier = BagClassifier(L=dim, num_classes=num_clases)

    def forward(self, H, mask):
        z, attn = self.attention(H, mask)
        return self.classifier(z), attn


class CabezaMILCompartida(nn.Module):
    """Ablación: UNA rama de atención + UN clasificador de 3 salidas. Sin aislamiento
    por clase -> el olvido catastrófico sí aparece (es el caso clásico de cabeza única
    en class-incremental). Sirve para mostrar que la robustez de CabezaMIL viene del
    diseño K-rama, no del backbone congelado."""

    def __init__(self, dim=DIM_FEATURES, num_clases=NUM_CLASES, D=128):
        super().__init__()
        self.att_V = nn.Sequential(nn.Linear(dim, D), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(dim, D), nn.Sigmoid())
        self.att_w = nn.Linear(D, 1)
        self.classifier = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(256, num_clases))

    def forward(self, H, mask):
        a = self.att_w(self.att_V(H) * self.att_U(H)).squeeze(-1)          # [B,N]
        a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)
        z = torch.bmm(a.unsqueeze(1), H).squeeze(1)                        # [B,dim]
        return self.classifier(z), a


def nueva_cabeza(semilla: int, dispositivo, tipo="kbranch") -> nn.Module:
    torch.manual_seed(semilla)
    cls = CabezaMILCompartida if tipo == "compartida" else CabezaMIL
    return cls().to(dispositivo)


def _pad_a_N(H1: torch.Tensor):
    b, k, dim = H1.shape
    H = torch.zeros(b, N_INSTANCIAS, dim)
    H[:, :k] = H1
    mask = torch.zeros(b, N_INSTANCIAS, dtype=torch.bool)
    mask[:, :k] = True
    return H, mask


# ---------------------------------------------------------------------------
# Entrenamiento / evaluación
# ---------------------------------------------------------------------------
def entrenar_cabeza(modelo, H, mask, Y, mask_col, epocas=30, lr=1e-3, batch=64,
                    dispositivo="cpu", pos_weight=None):
    """mask_col [M, num_clases] bool: qué columnas de la etiqueta entran en la pérdida."""
    modelo.train()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none",
                               pos_weight=pos_weight.to(dispositivo) if pos_weight is not None else None)
    H, mask, Y, mask_col = (t.to(dispositivo) for t in (H, mask, Y, mask_col.float()))
    M = H.shape[0]
    for _ in range(epocas):
        perm = torch.randperm(M, device=dispositivo)
        for k in range(0, M, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            logits, _ = modelo(H[idx], mask[idx])
            perdida_el = bce(logits, Y[idx]) * mask_col[idx]
            (perdida_el.sum() / mask_col[idx].sum().clamp(min=1.0)).backward()
            opt.step()
    return modelo


@torch.no_grad()
def evaluar_cabeza(modelo, H, mask, Y, dispositivo="cpu") -> dict:
    modelo.eval()
    logits, _ = modelo(H.to(dispositivo), mask.to(dispositivo))
    pred = (logits.cpu() > 0).float()
    Y = Y.float()
    f1 = {}
    for c in range(Y.shape[1]):
        if Y[:, c].sum() == 0:
            f1[c] = float("nan"); continue
        tp = float((pred[:, c] * Y[:, c]).sum())
        fp = float((pred[:, c] * (1 - Y[:, c])).sum())
        fn = float(((1 - pred[:, c]) * Y[:, c]).sum())
        f1[c] = tp / (tp + 0.5 * (fp + fn) + 1e-9)
    presentes = [v for v in f1.values() if v == v]
    return {"f1_por_clase": f1,
            "f1_macro": float(np.mean(presentes)) if presentes else 0.0,
            "acc_hamming": float((pred == Y).float().mean()), "n": int(Y.shape[0])}


# ---------------------------------------------------------------------------
# Estrategias
# ---------------------------------------------------------------------------
class Estrategia:
    nombre = "base"

    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self._vistos = []

    def _bolsas(self, idx, columnas):
        H = self.X[idx]
        mask = torch.ones(H.shape[:2], dtype=torch.bool)
        mc = torch.zeros(len(idx), NUM_CLASES, dtype=torch.bool)
        mc[:, list(columnas)] = True
        return H, mask, self.Y[idx], mc

    def datos_entrenamiento(self, idx_lote, columnas):
        raise NotImplementedError

    def actualizar_buffer(self, idx_lote, columnas):
        self._vistos = list(self._vistos) + list(idx_lote)


class Naive(Estrategia):
    nombre = "naive"

    def datos_entrenamiento(self, idx_lote, columnas):
        return self._bolsas(np.asarray(idx_lote), columnas)


class Joint(Estrategia):
    """Cota superior: acumula datos y columnas vistas."""
    nombre = "joint"

    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._cols = set()

    def datos_entrenamiento(self, idx_lote, columnas):
        self._cols |= set(columnas)
        idx = np.array(list(self._vistos) + list(idx_lote))
        return self._bolsas(idx, sorted(self._cols))


class _ReplayBase(Estrategia):
    """Búfer con presupuesto POR CLASE. Guarda, por exemplar, con qué columnas se etiquetó."""

    def __init__(self, X, Y, buffer_por_clase=10, semilla=0):
        super().__init__(X, Y)
        self.k = buffer_por_clase
        self.rng = np.random.default_rng(semilla)
        self._buffer = {}          # idx_global -> set(columnas conocidas)

    def _bolsas_buffer(self):
        raise NotImplementedError

    def datos_entrenamiento(self, idx_lote, columnas):
        Hn, Mn, Yn, Cn = self._bolsas(np.asarray(idx_lote), columnas)
        if not self._buffer:
            return Hn, Mn, Yn, Cn
        Hb, Mb, Yb, Cb = self._bolsas_buffer()
        return (torch.cat([Hn, Hb]), torch.cat([Mn, Mb]),
                torch.cat([Yn, Yb]), torch.cat([Cn, Cb]))

    def actualizar_buffer(self, idx_lote, columnas):
        super().actualizar_buffer(idx_lote, columnas)
        idx_lote = np.asarray(idx_lote)
        for c in columnas:
            pos = idx_lote[self.Y[idx_lote][:, c].numpy() == 1]
            self.rng.shuffle(pos)
            for p in pos[:self.k]:
                self._buffer.setdefault(int(p), set()).add(c)

    def _idx_cols(self):
        idx = np.array(sorted(self._buffer))
        mc = torch.zeros(len(idx), NUM_CLASES, dtype=torch.bool)
        for r, p in enumerate(idx):
            mc[r, sorted(self._buffer[p])] = True
        return idx, mc


class ExperienceReplay(_ReplayBase):
    nombre = "er"

    def _bolsas_buffer(self):
        idx, mc = self._idx_cols()
        return self.X[idx], torch.ones(len(idx), N_INSTANCIAS, dtype=torch.bool), self.Y[idx], mc


class LatentReplay(_ReplayBase):
    nombre = "latente"

    def _bolsas_buffer(self):
        idx, mc = self._idx_cols()
        H, mask = _pad_a_N(self.X[idx].mean(dim=1, keepdim=True))
        return H, mask, self.Y[idx], mc


ESTRATEGIAS = {c.nombre: c for c in (Naive, Joint, ExperienceReplay, LatentReplay)}


def construir_estrategia(nombre, X, Y, buffer_por_clase=10, semilla=0):
    cls = ESTRATEGIAS[nombre]
    if issubclass(cls, _ReplayBase):
        return cls(X, Y, buffer_por_clase=buffer_por_clase, semilla=semilla)
    return cls(X, Y)
