import os
import shutil

def preparar_dataset():
    # --- TUS RUTAS EXACTAS ---
    ruta_origen = r"C:\Users\silvi\OneDrive\Documents\Co_MIL_PPS\UDP"
    
    # Rutas de destino (se crearán automáticamente en tu carpeta Co_MIL_PPS)
    ruta_base = r"C:\Users\silvi\OneDrive\Documents\Co_MIL_PPS"
    ruta_experto = os.path.join(ruta_base, "Dataset_Experto_100")
    ruta_adrian = os.path.join(ruta_base, "Dataset_Adrian_100")
    ruta_compartido = os.path.join(ruta_base, "Dataset_Compartido_68")

    # 1. Crear las carpetas de destino si no existen
    os.makedirs(ruta_experto, exist_ok=True)
    os.makedirs(ruta_adrian, exist_ok=True)
    os.makedirs(ruta_compartido, exist_ok=True)

    # 2. Obtener y filtrar archivos de imagen .jpg
    archivos = [f for f in os.listdir(ruta_origen) if f.lower().endswith('.jpg')]
    
    # 3. Ordenar los archivos alfabéticamente para mantener un orden lógico constante
    archivos.sort()

    total_imagenes = len(archivos)
    print(f"Se encontraron {total_imagenes} imágenes .jpg en la carpeta de origen.\n")
    
    if total_imagenes != 268:
        print(f"⚠️ ADVERTENCIA: Esperabas 268 imágenes, pero se encontraron {total_imagenes}.")
        print("El script continuará, pero verifica tu carpeta original por si falta algo.\n")

    # 4. Iterar, renombrar y copiar a las carpetas correspondientes
    cont_experto = 0
    cont_adrian = 0
    cont_compartido = 0

    for i, archivo_original in enumerate(archivos):
        # Generar nuevo nombre: UPD_001.jpg, UPD_002.jpg, etc.
        numero_formateado = f"{i + 1:03d}"  
        nuevo_nombre = f"UPD_{numero_formateado}.jpg"
        ruta_completa_original = os.path.join(ruta_origen, archivo_original)

        # Lógica de distribución:
        # Imágenes 1 a 100 -> Experto
        if i < 100:
            ruta_destino = os.path.join(ruta_experto, nuevo_nombre)
            cont_experto += 1
        # Imágenes 101 a 200 -> Adrián
        elif i < 200:
            ruta_destino = os.path.join(ruta_adrian, nuevo_nombre)
            cont_adrian += 1
        # Imágenes 201 en adelante -> Compartido (las 68 restantes)
        else:
            ruta_destino = os.path.join(ruta_compartido, nuevo_nombre)
            cont_compartido += 1

        # Copiar el archivo con el nuevo nombre (copy2 conserva metadatos de la foto)
        shutil.copy2(ruta_completa_original, ruta_destino)
        print(f"[{nuevo_nombre}] copiado a su carpeta respectiva.")

    # 5. Resumen final
    print("\n" + "="*50)
    print("¡PROCESO COMPLETADO CON ÉXITO!")
    print("="*50)
    print(f"Carpeta del Experto:   {cont_experto} imágenes.")
    print(f"Carpeta de Adrián:     {cont_adrian} imágenes.")
    print(f"Carpeta Compartida:    {cont_compartido} imágenes.")
    print("="*50)
    print(f"Total procesadas:      {cont_experto + cont_adrian + cont_compartido}")

if __name__ == "__main__":
    preparar_dataset()