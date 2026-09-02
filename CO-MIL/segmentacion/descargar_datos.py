"""
=========================================================================================
DESCARGA DE DATASETS PÚBLICOS DE SEGMENTACIÓN DE TEJIDOS DE HERIDA
=========================================================================================
TIPO DE SCRIPT: EJECUTABLE (utilidad de datos).
CÓMO EJECUTAR: python CO-MIL/segmentacion/descargar_datos.py

Descarga a datasets_publicos/ (ignorado por git) los conjuntos de datos públicos que
se usan como base mientras se anota el dataset propio (ver documentación/propuesta_
dataset_tejidos_UPD.pdf y la bitácora Parte IV):

  - DFUTissue (Dhar et al., 2024, arXiv:2406.16012) -- 110 imágenes DFU anotadas por
    píxel en 3 tejidos (fibrina, granulación, callo) + fondo, con partición oficial
    78/16/16. Repo: github.com/uwm-bigdata/DFUTissueSegNet. Uso: citar el paper.
    Mapa de clases (Palette/palette_colorCode.txt):
        0 = fondo, 1 = Fibrina (rojo), 2 = Granulación (verde), 3 = Callo (azul)

  - WoundTissue (Kabir et al., 2025, arXiv:2502.10652) -- SOLO se descarga la MUESTRA
    de 13 imágenes que está en el repo público; el dataset completo (147 imágenes,
    6 tejidos) hay que solicitarlo a los autores. Licencia CC BY-NC-SA 4.0 (no
    comercial, share-alike). Repo: github.com/akabircs/WoundTissue.

No se usa `git clone` porque el `git` de este equipo tiene un problema de CA/SSL;
se baja el tarball vía urllib (que sí funciona).
=========================================================================================
"""

import io
import os
import sys
import tarfile
import urllib.request

RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DESTINO = os.path.join(RAIZ_REPO, "datasets_publicos")


def _bajar_tarball(url: str) -> tarfile.TarFile:
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    print(f"    descargando {url}")
    data = urllib.request.urlopen(req, timeout=180).read()
    print(f"    {len(data) / 1e6:.1f} MB")
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def descargar_dfutissue(incluir_no_etiquetadas: int = 0) -> None:
    """DFUTissue: parte etiquetada (Original + Padded + Palette + txt de partición).
    `incluir_no_etiquetadas` = cuántas de las 600 imágenes sin etiqueta traer (0 = ninguna)."""
    print("[DFUTissue]")
    tf = _bajar_tarball(
        "https://codeload.github.com/uwm-bigdata/DFUTissueSegNet/tar.gz/refs/heads/main"
    )
    root = tf.getnames()[0].split("/")[0]
    n = 0
    for m in tf.getmembers():
        if f"{root}/DFUTissue/Labeled/" in m.name and m.isfile():
            m.name = m.name.replace(f"{root}/", "", 1)
            tf.extract(m, DESTINO)
            n += 1
    print(f"    parte etiquetada: {n} archivos")
    if incluir_no_etiquetadas:
        sin = [
            m for m in tf.getmembers()
            if f"{root}/DFUTissue/Unlabeled/" in m.name and m.isfile()
        ][:incluir_no_etiquetadas]
        for m in sin:
            m.name = m.name.replace(f"{root}/", "", 1)
            tf.extract(m, DESTINO)
        print(f"    muestra sin etiqueta: {len(sin)} archivos")


def descargar_woundtissue_muestra() -> None:
    """WoundTissue: solo la muestra de 13 imágenes del repo público."""
    print("[WoundTissue -- MUESTRA de 13 imágenes]")
    print("    El dataset completo (147 img) se solicita a los autores del paper arXiv:2502.10652.")
    tf = _bajar_tarball(
        "https://codeload.github.com/akabircs/WoundTissue/tar.gz/refs/heads/main"
    )
    root = tf.getnames()[0].split("/")[0]
    dest = os.path.join(DESTINO, "WoundTissue_muestra")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for m in tf.getmembers():
        if m.isfile():
            m.name = m.name.replace(f"{root}/", "", 1)
            tf.extract(m, dest)
            n += 1
    print(f"    {n} archivos en {os.path.relpath(dest, RAIZ_REPO)}")


if __name__ == "__main__":
    os.makedirs(DESTINO, exist_ok=True)
    n_sin = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    descargar_dfutissue(incluir_no_etiquetadas=n_sin)
    descargar_woundtissue_muestra()
    print("\n[+] Listo. Datos en datasets_publicos/ (ignorado por git).")
