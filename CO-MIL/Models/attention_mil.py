"""
=========================================================================================
ARQUITECTURA NEURONAL: Co-MIL (Atención MIML Dinámica)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL.

Evolución al paradigma MIML (Multi-Instancia Multi-Etiqueta) real.
En lugar de generar un solo peso de atención global por parche, la red genera 
K pesos independientes (donde K es la cantidad de tejidos anotados en el dataset). 
Esto permite que la red construya un mapa de calor específico para la Fibrina, 
otro distinto para la Granulación, etc.
=========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Tuple

class InstanceEncoder(nn.Module):
    """[Bloque 1] Extractor Visual (MobileNetV2 Congelable)."""
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = base_model.last_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return pooled.view(pooled.size(0), -1) 

class AttentionAggregator(nn.Module):
    """
    [Bloque 2] Atención MIML con Compuertas.
    Emite tensores de forma [Batch, num_classes, N], garantizando un 
    mapa de calor anatómico por cada tejido clínico.
    """
    def __init__(self, L: int = 1280, D: int = 128, num_classes: int = 3):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        
        # EL CAMBIO CRÍTICO: La última capa proyecta a 'num_classes', no a 1.
        self.attention_weights = nn.Linear(D, num_classes)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A_V = self.attention_V(H)  # [Batch, N, D]
        A_U = self.attention_U(H)  # [Batch, N, D]
        
        # A tiene forma [Batch, N, num_classes]
        A = self.attention_weights(A_V * A_U) 
        
        # Transponemos para agrupar por clase: [Batch, num_classes, N]
        A = A.transpose(1, 2)
        
        # Expandimos la máscara para aplicarla a todos los mapas de atención
        mask_exp = mask.unsqueeze(1) # [Batch, 1, N]
        A = A.masked_fill(~mask_exp, float('-inf'))
        
        # Normalización probabilística (Softmax) sobre la dimensión N de los parches
        A_probs = F.softmax(A, dim=2) 
        
        # Multiplicación matricial masiva:
        # [Batch, num_classes, N] x [Batch, N, L] = [Batch, num_classes, L]
        # Genera un vector 'z' (resumen) INDEPENDIENTE para cada clase
        z = torch.bmm(A_probs, H) 
        
        return z, A_probs

class BagClassifier(nn.Module):
    """
    [Bloque 3] Cabezal MIML Desacoplado.
    Cada tejido es juzgado por su propio Perceptrón Multicapa evaluando su propio mapa.
    """
    def __init__(self, L: int = 1280, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        # Creamos dinámicamente K redes neuronales independientes
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(L, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1) # Salida de 1 Logit por tejido
            ) for _ in range(num_classes)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z tiene forma [Batch, num_classes, L]
        logits = []
        for i in range(self.num_classes):
            z_i = z[:, i, :] # Extraemos el resumen exclusivo del tejido 'i'
            logit_i = self.classifiers[i](z_i) # [Batch, 1]
            logits.append(logit_i)
            
        # Concatenamos todos los logits en el tensor final [Batch, num_classes]
        return torch.cat(logits, dim=1)

class CoMILNetwork(nn.Module):
    """Orquestador Global de la Red"""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = InstanceEncoder()
        self.attention = AttentionAggregator(L=self.encoder.embedding_dim, num_classes=num_classes)
        self.classifier = BagClassifier(L=self.encoder.embedding_dim, num_classes=num_classes)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Batch, N, C, H, W = X.shape
        X_flat = X.view(Batch * N, C, H, W)
        H_flat = self.encoder(X_flat)
        H_bag = H_flat.view(Batch, N, -1)
        z, attention_probs = self.attention(H_bag, mask)
        logits = self.classifier(z)
        return logits, attention_probs