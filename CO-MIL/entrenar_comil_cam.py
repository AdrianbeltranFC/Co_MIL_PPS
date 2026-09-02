"""
=========================================================================================
ENTRENAMIENTO — VARIANTE ESTILO CAM (Co-MIL)
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (NÚCLEO DE ENTRENAMIENTO, variante experimental).
CÓMO EJECUTAR: python CO-MIL/entrenar_comil_cam.py

Por qué existe (23-ago-2026): réplica de entrenar_comil.py, pero usando
CoMILDatasetCAM + CoMILNetworkCAM en vez de CoMILDataset + CoMILNetwork -- ver los
docstrings de esos dos módulos para el porqué del rediseño (motivado por
_proceso_claude/scripts/prototipo_cam_feasibility.py). Se mantiene como script
SEPARADO de entrenar_comil.py (no una bandera --cam ahí) porque el bucle de
entrenamiento es distinto en un punto real: aquí no hay collate_fn ni máscara de
padding -- cada bolsa es su propia imagen de tamaño distinto, así que se entrena una
bolsa a la vez (batch_size=1) en vez de por lotes de tamaño fijo.

Reutiliza sin duplicar: calcular_pos_weights y congelar_backbone se importan
directamente de entrenar_comil.py (funcionan igual sobre cualquier dataset/modelo que
tenga la misma forma, no hace falta reescribirlos).
=========================================================================================
"""

import json
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

_DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(_DIRECTORIO_ACTUAL)

from dataset_cam import CoMILDatasetCAM
from Models.attention_mil_cam import CoMILNetworkCAM
from entrenar_comil import calcular_pos_weights, congelar_backbone
import catalogo_tejidos
import experimentos


def entrenar_modelo_cam(etiqueta_experimento: str = "cam"):
    RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUTA_DATASET_ROOT = os.path.join(RAIZ_REPO, "APP_generador_bolsas", "Dataset_Experto_100")
    # Se reconstruye el ROI a partir de las bolsas de 224px (la resolución de origen
    # sin submuestrear más), sin importar qué tan fina vaya a quedar la grilla nativa
    # de instancias -- esa la decide el propio tamaño del ROI reensamblado, no esta
    # carpeta.
    RUTA_BOLSAS = os.path.join(RUTA_DATASET_ROOT, "Bolsas_MIL_Procesadas", "224px")
    RUTA_MANIFEST = os.path.join(RUTA_DATASET_ROOT, "Bolsas_MIL_Procesadas", "splits_manifest.json")
    RUTA_PESOS_DIR = os.path.join(RAIZ_REPO, "Pesos_Entrenados")
    EPOCHS = 30
    LEARNING_RATE = 1e-4

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== INICIANDO MOTOR Co-MIL (ESTILO CAM) EN DISPOSITIVO: {dispositivo.type.upper()} ===")

    if not os.path.exists(RUTA_MANIFEST):
        print(f"[!] Error: no existe {RUTA_MANIFEST}. Corre primero particionar_dataset.py.")
        return

    try:
        dataset = CoMILDatasetCAM(
            pt_folder=RUTA_BOLSAS,
            manifest_path=RUTA_MANIFEST,
            split="train",
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[!] Error: {e}")
        return

    problemas_etiquetas = catalogo_tejidos.auditar_etiquetas_no_reconocidas(
        RUTA_BOLSAS, dataset.class_catalog, dataset.renombres
    )
    if problemas_etiquetas:
        print(f"\n[!] ALERTA: {len(problemas_etiquetas)} bolsa(s) con etiquetas que no matchean "
              "el catálogo vigente (se están ignorando en silencio):")
        for archivo, etiquetas in list(problemas_etiquetas.items())[:10]:
            print(f"    {archivo}: {etiquetas}")

    print(f"-> Entrenando sobre el split 'train': {len(dataset)} bolsas "
          "(sin aumento de datos todavía en esta variante -- ver limitaciones al final).")
    # batch_size=1: cada ROI reconstruido tiene un tamaño físico distinto (una lesión
    # alargada da una imagen muy distinta de una compacta), así que no se pueden
    # apilar varias bolsas en un tensor de lote fijo como sí se hacía con parches de
    # tamaño uniforme. Se entrena una bolsa a la vez.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    meta_info = dataset.get_metadata(0)
    num_tejidos = len(meta_info["class_names"])
    print(f"-> Arquitectura configurada dinámicamente para {num_tejidos} clases: {meta_info['class_names']}")

    modelo = CoMILNetworkCAM(num_classes=num_tejidos).to(dispositivo)
    congelar_backbone(modelo)

    pesos_clase = calcular_pos_weights(dataset, num_tejidos).to(dispositivo)
    criterio_loss = nn.BCEWithLogitsLoss(pos_weight=pesos_clase)
    optimizador = AdamW(filter(lambda p: p.requires_grad, modelo.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"\n[+] Iniciando entrenamiento para {EPOCHS} épocas...\n")
    historial_loss = []

    for epoch in range(EPOCHS):
        modelo.train()
        loss_acumulada = 0.0
        barra_batches = tqdm(dataloader, desc=f"Época {epoch+1:02d}/{EPOCHS}")

        for batch in barra_batches:
            imagen = batch["imagen"][0].to(dispositivo)  # [3, H, W] -- una sola bolsa
            batch_Y = batch["Y"].to(dispositivo)  # [1, num_clases], ya con dimensión de lote=1

            optimizador.zero_grad()
            logits, _, _ = modelo(imagen)
            loss = criterio_loss(logits, batch_Y)
            loss.backward()
            optimizador.step()

            loss_acumulada += loss.item()
            barra_batches.set_postfix({"Loss": f"{loss.item():.4f}"})

        loss_promedio = loss_acumulada / len(dataloader)
        historial_loss.append(loss_promedio)
        print(f" -> Fin Época {epoch+1:02d} | Loss Promedio: {loss_promedio:.4f}")

    print("\n=== ENTRENAMIENTO ESTILO CAM FINALIZADO CON ÉXITO ===")

    os.makedirs(RUTA_PESOS_DIR, exist_ok=True)
    ruta_experimento = experimentos.crear_carpeta_experimento(RUTA_PESOS_DIR, etiqueta=etiqueta_experimento)
    ruta_modelo = os.path.join(ruta_experimento, "modelo.pth")

    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": modelo.state_dict(),
        "optimizer_state_dict": optimizador.state_dict(),
        "loss": historial_loss[-1],
        "num_classes": num_tejidos,
        "class_names": meta_info["class_names"],
        "ruta_bolsas": RUTA_BOLSAS,
        "ruta_manifest": RUTA_MANIFEST,
        "ruta_experimento": ruta_experimento,
        "arquitectura": "cam",
    }, ruta_modelo)

    with open(RUTA_MANIFEST, "r", encoding="utf-8") as f:
        manifiesto = json.load(f)
    conteo_imagenes_por_split = {}
    for info in manifiesto.get("imagenes", {}).values():
        conteo_imagenes_por_split[info["split"]] = conteo_imagenes_por_split.get(info["split"], 0) + 1

    experimentos.guardar_metadata(ruta_experimento, {
        "arquitectura": "cam (InstanceEncoderCAM: 1 forward pass por ROI completo, "
                        "instancias = celdas del mapa de features nativo de MobileNetV2)",
        "epochs": EPOCHS,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "loss_final": historial_loss[-1],
        "num_classes": num_tejidos,
        "class_names": meta_info["class_names"],
        "ruta_bolsas_origen_roi": RUTA_BOLSAS,
        "ruta_manifest": RUTA_MANIFEST,
        "bolsas_entrenamiento": len(dataset),
        "imagenes_por_split": conteo_imagenes_por_split,
        "dispositivo": dispositivo.type,
        "limitaciones_conocidas": [
            "Sin aumento de datos todavía (CoMILDatasetCAM no lo implementa aún).",
            "batch_size=1 (SGD por bolsa, no por lotes) porque cada ROI reconstruido "
            "tiene un tamaño físico distinto -- no comparable 1:1 con la corrida de "
            "parches (batch_size=8) en cuanto a dinámica de optimización.",
        ],
    })
    experimentos.marcar_como_mas_reciente(RUTA_PESOS_DIR, ruta_experimento)

    print(f"[+] Experimento guardado en: {ruta_experimento}")
    print(f"    -> Modelo: {ruta_modelo}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenamiento Co-MIL, variante estilo CAM")
    parser.add_argument("--etiqueta", default="cam", help="Etiqueta para la carpeta del experimento")
    args = parser.parse_args()
    entrenar_modelo_cam(etiqueta_experimento=args.etiqueta)
