import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Tuple

class InstanceEncoder(nn.Module):
    """
    Módulo Extractor a Nivel de Instancia (Instance Encoder).
    Procesa cada parche de forma individual extrayendo características profundas.
    """
    def __init__(self):
        super().__init__()
        # Cargamos MobileNetV2 pre-entrenado
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Extraemos solo las capas convolucionales (anulando 'classifier')
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV2 tiene 1280 canales de salida por defecto
        self.embedding_dim = base_model.last_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape esperado: [Batch * N, 3, 224, 224]
        features = self.features(x)
        pooled = self.pool(features)
        # Aplanar para obtener los vectores espaciales (embeddings)
        return pooled.view(pooled.size(0), -1) 

class AttentionAggregator(nn.Module):
    """
    Módulo Agregador de Atención con Compuertas (Gated Attention).
    Calcula pesos probabilísticos ignorando matemáticamente los parches de relleno.
    """
    def __init__(self, L: int = 1280, D: int = 128, K: int = 1):
        super().__init__()
        self.L = L
        
        # Rama Tanh para características complejas
        self.attention_V = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )
        # Rama Sigmoide (Gate) para filtrar fondo irrelevante
        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )
        # Proyección final al peso escalar
        self.attention_weights = nn.Linear(D, K)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H shape: [Batch, N, L]
        # mask shape: [Batch, N]
        
        A_V = self.attention_V(H)  # [Batch, N, D]
        A_U = self.attention_U(H)  # [Batch, N, D]
        
        # Producto elemento a elemento (Hadamard) y proyección final
        A = self.attention_weights(A_V * A_U) # [Batch, N, 1]
        A = A.squeeze(-1) # [Batch, N]
        
        # --- FILTRADO DE PADDING ---
        # Anulamos matemáticamente los pesos del padding reemplazándolos por -infinito.
        # Al pasar por la función Softmax, exp(-inf) se vuelve exactamente 0.0
        A = A.masked_fill(~mask, float('-inf'))
        
        A_probs = F.softmax(A, dim=1) # [Batch, N]
        
        # Representación global z mediante suma ponderada
        # Añadimos dimensiones para realizar el producto punto por lotes
        A_probs_expanded = A_probs.unsqueeze(1) # [Batch, 1, N]
        z = torch.bmm(A_probs_expanded, H)      # [Batch, 1, L]
        
        return z.squeeze(1), A_probs

class BagClassifier(nn.Module):
    """
    Módulo Clasificador a Nivel de Bolsa (Bag Classifier).
    Perceptrón Multicapa (MLP) para emitir los logits de la clasificación MIML.
    """
    def __init__(self, L: int = 1280, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(L, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Entrega los logits que serán evaluados por BCEWithLogitsLoss
        return self.classifier(z)

class CoMILNetwork(nn.Module):
    """
    Arquitectura Global Co-MIL. Orquesta el flujo completo de la información.
    """
    def __init__(self):
        super().__init__()
        self.encoder = InstanceEncoder()
        self.attention = AttentionAggregator(L=self.encoder.embedding_dim)
        self.classifier = BagClassifier(L=self.encoder.embedding_dim, num_classes=3)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Batch, N, C, H, W = X.shape
        
        # 1. Plegar dimensiones para procesar todas las instancias como un lote masivo
        X_flat = X.view(Batch * N, C, H, W)
        
        # 2. Extracción de características
        H_flat = self.encoder(X_flat)
        
        # 3. Desplegar dimensiones de vuelta a [Batch, N, L]
        H_bag = H_flat.view(Batch, N, -1)
        
        # 4. Agregación MIL (Se aplica la máscara de anulación aquí)
        z, attention_probs = self.attention(H_bag, mask)
        
        # 5. Clasificación Multi-Etiqueta
        logits = self.classifier(z)
        
        return logits, attention_probs