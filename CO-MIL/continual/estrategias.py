"""
=========================================================================================
CABEZA MIL Y ESTRATEGIAS DE APRENDIZAJE CONTINUO  (prototipo mínimo)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL.

La cabeza reutiliza los bloques de CO-MIL/Models/attention_mil.py (atención con
compuertas + clasificador por clase), operando sobre las características ya
cacheadas: el extractor MobileNetV2 está congelado y fuera de este módulo.

Cada bolsa se representa como (H [N,1280], mask [N] bool). Las bolsas del caché
tienen N=64 con máscara toda verdadera; las del búfer de repetición latente tienen
1 instancia real (rellenada a 64 con máscara para que la atención la ignore).

Estrategias del prototipo mínimo:
  - Naive        : solo el lote actual (cota inferior; muestra olvido).
  - Joint        : unión de todos los lotes vistos (cota superior).
  - ExperienceReplay : lote actual + búfer de bolsas COMPLETAS, presupuesto por clase.
  - LatentReplay : igual, pero el búfer guarda solo el vector medio (1280-D) de cada
                   bolsa -> 64x menos almacenamiento, sin imágenes (privacy-safe).
                   Configuración "titular" del capítulo.

Pendiente 2ª pasada: LwF, NCM, barrido de búfer, ablación de olvido por capa.
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


# ---------------------------------------------------------------------------
# Cabeza MIL (lo único entrenable)
# ---------------------------------------------------------------------------
class CabezaMIL(nn.Module):
    def __init__(self, dim=DIM_FEATURES, num_clases=NUM_CLASES):
        super().__init__()
        self.attention = AttentionAggregator(L=dim, num_classes=num_clases)
        self.classifier = BagClassifier(L=dim, num_classes=num_clases)

    def forward(self, H, mask):
        z, attn = self.attention(H, mask)
        return self.classifier(z), attn


def nueva_cabeza(semilla: int, dispositivo) -> CabezaMIL:
    torch.manual_seed(semilla)
    return CabezaMIL().to(dispositivo)


def _pad_a_N(H1: torch.Tensor):
    """[B,k,dim] -> ([B,N,dim] con ceros de relleno, [B,N] bool con k verdaderos)."""
    b, k, dim = H1.shape
    H = torch.zeros(b, N_INSTANCIAS, dim)
    H[:, :k] = H1
    mask = torch.zeros(b, N_INSTANCIAS, dtype=torch.bool)
    mask[:, :k] = True
    return H, mask


# ---------------------------------------------------------------------------
# Entrenamiento / evaluación de la cabeza sobre un conjunto de bolsas
# ---------------------------------------------------------------------------
def entrenar_cabeza(modelo, H, mask, Y, epocas=30, lr=1e-3, batch=64,
                    dispositivo="cpu", pos_weight=None):
    modelo.train()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dispositivo)
                               if pos_weight is not None else None)
    H, mask, Y = H.to(dispositivo), mask.to(dispositivo), Y.to(dispositivo)
    M = H.shape[0]
    for _ in range(epocas):
        perm = torch.randperm(M, device=dispositivo)
        for k in range(0, M, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            logits, _ = modelo(H[idx], mask[idx])
            bce(logits, Y[idx]).backward()
            opt.step()
    return modelo


@torch.no_grad()
def evaluar_cabeza(modelo, H, mask, Y, dispositivo="cpu") -> dict:
    modelo.eval()
    logits, _ = modelo(H.to(dispositivo), mask.to(dispositivo))
    pred = (logits.cpu() > 0).float()
    Y = Y.float()
    f1s = []
    for c in range(Y.shape[1]):
        if Y[:, c].sum() == 0:
            continue
        tp = float((pred[:, c] * Y[:, c]).sum())
        fp = float((pred[:, c] * (1 - Y[:, c])).sum())
        fn = float(((1 - pred[:, c]) * Y[:, c]).sum())
        f1s.append(tp / (tp + 0.5 * (fp + fn) + 1e-9))
    return {"f1_macro": float(np.mean(f1s)) if f1s else 0.0,
            "acc_hamming": float((pred == Y).float().mean()), "n": int(Y.shape[0])}


# ---------------------------------------------------------------------------
# Estrategias
# ---------------------------------------------------------------------------
class Estrategia:
    nombre = "base"

    def __init__(self, X, Y):
        self.X, self.Y = X, Y                 # [110,64,1280] / [110,3] cacheadas
        self._vistos = []

    def _bolsas(self, idx):
        H = self.X[idx]
        mask = torch.ones(H.shape[:2], dtype=torch.bool)
        return H, mask, self.Y[idx]

    def datos_entrenamiento(self, idx_train_lote):
        raise NotImplementedError

    def actualizar_buffer(self, idx_train_lote):
        self._vistos = list(self._vistos) + list(idx_train_lote)


class Naive(Estrategia):
    nombre = "naive"

    def datos_entrenamiento(self, idx_train_lote):
        return self._bolsas(np.asarray(idx_train_lote))


class Joint(Estrategia):
    nombre = "joint"

    def datos_entrenamiento(self, idx_train_lote):
        return self._bolsas(np.array(list(self._vistos) + list(idx_train_lote)))


class _ReplayBase(Estrategia):
    """Búfer con presupuesto POR CLASE (no reservoir global)."""

    def __init__(self, X, Y, buffer_por_clase=10, semilla=0):
        super().__init__(X, Y)
        self.k = buffer_por_clase
        self.rng = np.random.default_rng(semilla)
        self._buffer_idx = []

    def _bolsas_buffer(self):
        raise NotImplementedError

    def datos_entrenamiento(self, idx_train_lote):
        Hn, Mn, Yn = self._bolsas(np.asarray(idx_train_lote))
        if not self._buffer_idx:
            return Hn, Mn, Yn
        Hb, Mb, Yb = self._bolsas_buffer()
        return torch.cat([Hn, Hb], 0), torch.cat([Mn, Mb], 0), torch.cat([Yn, Yb], 0)

    def actualizar_buffer(self, idx_train_lote):
        super().actualizar_buffer(idx_train_lote)
        cand = np.array(self._vistos)
        Yc = self.Y[cand].numpy()
        elegidos = set()
        for c in range(NUM_CLASES):
            pos = cand[Yc[:, c] == 1]
            if len(pos):
                self.rng.shuffle(pos)
                elegidos.update(pos[:self.k].tolist())
        self._buffer_idx = sorted(elegidos)


class ExperienceReplay(_ReplayBase):
    nombre = "er"

    def _bolsas_buffer(self):
        return self._bolsas(np.array(self._buffer_idx))


class LatentReplay(_ReplayBase):
    """Guarda solo el vector medio de la bolsa -> bolsa de 1 instancia (rellenada)."""
    nombre = "latente"

    def _bolsas_buffer(self):
        idx = np.array(self._buffer_idx)
        v = self.X[idx].mean(dim=1, keepdim=True)      # [B,1,dim]
        H, mask = _pad_a_N(v)
        return H, mask, self.Y[idx]


ESTRATEGIAS = {c.nombre: c for c in (Naive, Joint, ExperienceReplay, LatentReplay)}


def construir_estrategia(nombre, X, Y, buffer_por_clase=10, semilla=0):
    cls = ESTRATEGIAS[nombre]
    if issubclass(cls, _ReplayBase):
        return cls(X, Y, buffer_por_clase=buffer_por_clase, semilla=semilla)
    return cls(X, Y)
