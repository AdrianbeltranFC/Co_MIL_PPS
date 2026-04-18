"""
=========================================================================================
ARQUITECTURA NEURONAL: Co-MIL (Atención con Compuertas)
=========================================================================================
TIPO DE SCRIPT: MÓDULO ESTRUCTURAL (NO SE EJECUTA DIRECTAMENTE).
Es importado para instanciar la red en memoria.

Implementación en Programación Orientada a Objetos (POO) del Aprendizaje Multinstancia.
Divide la matemática en tres bloques lógicos (Encoder, Atención, Clasificador) para 
permitir la escalabilidad y el reemplazo de componentes (ej. cambiar MobileNet por ResNet)
sin alterar el algoritmo base de agregación probabilística.
=========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Tuple

class InstanceEncoder(nn.Module):
    """
    [Bloque 1] Extractor Visual (Backbone).
    Transforma la matriz de píxeles en un vector matemático denso (Embedding).
    """
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Se anula el clasificador de ImageNet original para retener solo las convoluciones
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = base_model.last_channel # 1280 canales para MobileNetV2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        # Aplana el tensor para entregarlo al módulo de atención
        return pooled.view(pooled.size(0), -1) 

class AttentionAggregator(nn.Module):
    """
    [Bloque 2] Gated Attention (Atención con Compuertas).
    Asigna un coeficiente de relevancia (peso alfa) a cada parche clínico.
    """
    def __init__(self, L: int = 1280, D: int = 128, K: int = 1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A_V = self.attention_V(H)  
        A_U = self.attention_U(H)  # Compuerta lógica para mitigar ruido de fondo
        
        A = self.attention_weights(A_V * A_U).squeeze(-1) # Ecuación cruda: [Batch, N]
        
        # Blindaje Matemático: Anulación de tensores fantasma (Zero-Padding)
        # Al forzar el valor a -infinito, la función Softmax lo convierte en un 0 exacto.
        A = A.masked_fill(~mask, float('-inf'))
        A_probs = F.softmax(A, dim=1) 
        
        # Suma ponderada: Colapsa los N parches en un solo vector global de la úlcera (Z)
        A_probs_expanded = A_probs.unsqueeze(1) 
        z = torch.bmm(A_probs_expanded, H).squeeze(1) 
        
        return z, A_probs

class BagClassifier(nn.Module):
    """
    [Bloque 3] Cabezal Multietiqueta.
    Mapea la patología resumida (Z) hacia el vector de diagnóstico MIML.
    """
    def __init__(self, L: int = 1280, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(L, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Regularización para evitar sobreajuste
            nn.Linear(256, num_classes) # Escalabilidad para aceptar N tejidos dinámicos
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Entrega los Logits crudos (la activación Sigmoide se aplica en BCEWithLogitsLoss)
        return self.classifier(z)

class CoMILNetwork(nn.Module):
    """
    Orquestador Arquitectónico. Unifica los 3 bloques lógicos.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = InstanceEncoder()
        self.attention = AttentionAggregator(L=self.encoder.embedding_dim)
        self.classifier = BagClassifier(L=self.encoder.embedding_dim, num_classes=num_classes)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Batch, N, C, H, W = X.shape
        
        X_flat = X.view(Batch * N, C, H, W)
        H_flat = self.encoder(X_flat)
        
        H_bag = H_flat.view(Batch, N, -1)
        z, attention_probs = self.attention(H_bag, mask)
        
        logits = self.classifier(z)
        
        return logits, attention_probs