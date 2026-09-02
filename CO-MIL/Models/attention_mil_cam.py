"""
=========================================================================================
ARQUITECTURA ESTILO CAM (Co-MIL) — instancias = celdas del mapa de features nativo
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL.

Por qué existe (23-ago-2026): el prototipo de factibilidad
(_proceso_claude/scripts/prototipo_cam_feasibility.py) confirmó, sin entrenar nada, que
el mapa de features nativo de MobileNetV2 sobre el ROI completo (~1/32 de la resolución
de entrada) muestra más estructura espacial real que la malla actual de parches
recortados a mano, con MENOS forward passes (1 en vez de N). Este módulo implementa esa
idea de verdad, entrenable.

Diseño clave: SOLO cambia cómo se generan las instancias (InstanceEncoderCAM). El
agregador de atención (AttentionAggregator) y el clasificador (BagClassifier) se
IMPORTAN de attention_mil.py sin ninguna modificación -- ya estaban diseñados para
operar sobre un conjunto genérico de vectores de instancia, sin importar su origen. No
se duplica esa lógica aquí, se reutiliza.

Limitación de diseño aceptada a propósito por ahora: cada ROI tiene un tamaño físico
distinto (una lesión alargada vs. una compacta), así que su mapa de features nativo
también varía de tamaño (N' instancias distinto por bolsa). Por eso CoMILNetworkCAM
procesa UNA bolsa a la vez (sin agrupar varias en un tensor de lote fijo, a diferencia
de CoMILNetwork que si podía por tener parches de tamaño uniforme) -- ver
entrenar_comil_cam.py, que usa DataLoader con batch_size=1 en vez de torchmil.collate_fn.
=========================================================================================
"""

import sys
import os
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

_DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(_DIRECTORIO_ACTUAL)

from attention_mil import AttentionAggregator, BagClassifier


class InstanceEncoderCAM(nn.Module):
    """[Bloque 1, variante CAM] En vez de recibir N parches recortados y codificar
    cada uno por separado, recibe el ROI COMPLETO y lo pasa una sola vez por el
    extractor convolucional de MobileNetV2. Su mapa de salida (antes del pooling
    global) ya es una grilla espacial de vectores de 1280 canales -- cada celda de esa
    grilla ES una instancia MIL, del mismo modo que antes cada parche era una
    instancia, pero sin haber tenido que recortar nada a mano ni pagar un forward pass
    por parche."""

    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features
        self.embedding_dim = base_model.last_channel  # 1280, igual que InstanceEncoder

    def forward(self, imagen: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """imagen: [3, H, W] -- una sola bolsa, sin dimensión de batch (ver
        CoMILNetworkCAM para por qué no se agrupan varias bolsas juntas)."""
        mapa = self.features(imagen.unsqueeze(0))  # [1, 1280, H', W'], H'≈H/32
        _, canales, alto_nativo, ancho_nativo = mapa.shape
        instancias = mapa.view(canales, alto_nativo * ancho_nativo).permute(1, 0)  # [N', 1280]
        return instancias, (alto_nativo, ancho_nativo)


class CoMILNetworkCAM(nn.Module):
    """Orquestador estilo CAM. `AttentionAggregator` y `BagClassifier` son los MISMOS
    de attention_mil.py, sin ningún cambio -- la única diferencia real con
    CoMILNetwork es de dónde salen las instancias."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = InstanceEncoderCAM()
        self.attention = AttentionAggregator(L=self.encoder.embedding_dim, num_classes=num_classes)
        self.classifier = BagClassifier(L=self.encoder.embedding_dim, num_classes=num_classes)

    def forward(self, imagen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """imagen: [3, H, W] de UNA bolsa. Devuelve (logits [1, K], pesos_atencion
        [1, K, N'], grilla_nativa (alto', ancho')) -- la grilla nativa se devuelve
        para poder redibujar el heatmap de atención con la forma espacial correcta,
        igual que grid_shape en las bolsas por parches."""
        instancias, grilla_nativa = self.encoder(imagen)  # [N', 1280]
        H_bag = instancias.unsqueeze(0)  # [1, N', 1280]
        mask = torch.ones(1, instancias.shape[0], dtype=torch.bool, device=imagen.device)
        z, pesos_atencion = self.attention(H_bag, mask)
        logits = self.classifier(z)
        return logits, pesos_atencion, grilla_nativa
