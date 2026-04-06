"""
=========================================================================================
HERRAMIENTA DE PREPROCESAMIENTO Y ETIQUETADO DÉBIL PARA Co-MIL (MIML)
=========================================================================================

CONTEXTO ARQUITECTÓNICO Y FLUJO DE DATOS:
Este software es el puente entre la captura fotográfica cruda en las clínicas y el 
entrenamiento matemático del modelo Co-MIL. Resuelve la "Paradoja de Resolución" (imágenes 
de más de 12MP que desbordarían la memoria RAM si se procesan completas) mediante el 
siguiente flujo de procesamiento:

1. Macrolocalización Interactiva (ROI): El usuario delimita la úlcera con un Bounding Box. 
   Esto descarta macroscópicamente el ruido excesivo de fondo innecesario (sábanas, paredes).
2. Anotación Débil (MIML): Generación del vector "Y" global para clasificación multi-etiqueta 
   (Granulación, Fibrina, Callo). No se exige segmentación densa a nivel píxel.
3. Expansión Inteligente (Smart Crop): Para evitar el clásico "Zero-Padding" que introduce 
   bordes negros artificiales (y arruina la extracción de gradientes de MobileNet V2), 
   el Bounding Box original se expande matemáticamente absorbiendo tejido real circundante 
   hasta alcanzar un múltiplo exacto de 224x224 px. 
4. Reflection Padding: Si la úlcera está en el límite físico de la foto y no se puede expandir, 
   se aplica un efecto espejo para rellenar los píxeles faltantes, manteniendo la derivada 
   continua y evitando confundir a los filtros convolucionales.
5. Extracción de Bolsas (Unfold) y Serialización: Se extraen los parches de 224x224 y se 
   empaquetan junto con la etiqueta Y en un archivo '.pt' binario nativo de PyTorch.

Nota: Puedes correr el código pegando esto en tu terminal :)    python CO-MIL/generador_bolsas.py
=========================================================================================
"""

import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms

# ---------------------------------------------------------
# CONFIGURACIÓN VISUAL DEL ENTORNO
# ---------------------------------------------------------
# Se establece un tema oscuro para reducir la fatiga visual del operario 
# durante largas sesiones de revisión clínica.
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class EtiquetadorCoMIL(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Co-MIL Tool - Análisis Multi-Etiqueta UPD")
        self.geometry("1200x850")
        
        # ---------------------------------------------------------
        # INICIALIZACIÓN DE VARIABLES DE ESTADO INTERNO
        # ---------------------------------------------------------
        # Gestión de archivos
        self.image_paths = []         # Lista con las rutas absolutas de todas las imágenes cargadas.
        self.current_idx = 0          # Índice numérico para iterar sobre la lista de imágenes.
        
        # Gestión de imágenes en memoria
        self.current_image = None     # Guarda el objeto de imagen PIL original (en máxima resolución).
        self.tk_image = None          # Guarda la versión comprimida/renderizada para la pantalla (UI).
        
        # Variables espaciales para el Bounding Box
        self.rect = None              # Referencia al objeto gráfico del rectángulo en el Canvas.
        self.start_x = None           # Coordenada X inicial al hacer clic (en el espacio de la UI).
        self.start_y = None           # Coordenada Y inicial al hacer clic (en el espacio de la UI).
        self.bbox = None              # Tupla final (x1, y1, x2, y2) mapeada a la resolución original.
        
        # Variables de calibración espacial (Mapeo UI <-> Original)
        self.scale_factor = 1.0       # Proporción de escalado. Ej: 0.5 significa que la UI muestra la foto a la mitad.
        self.img_offset_x = 0         # Píxeles de margen negro horizontal si la imagen no llena el Canvas.
        self.img_offset_y = 0         # Píxeles de margen negro vertical si la imagen no llena el Canvas.
        
        # Variables lógicas para el Vector Multietiqueta (MIML)
        # 1 = Tejido presente en la bolsa, 0 = Tejido ausente.
        self.var_granulacion = tk.IntVar()
        self.var_fibrina = tk.IntVar()
        self.var_callo = tk.IntVar()
        
        # Renderizado de la interfaz
        self.setup_ui()
        
    def setup_ui(self):
        """
        Construye la interfaz gráfica utilizando un sistema de cuadrícula (Grid).
        El diseño es asimétrico e invertido (herramientas a la derecha, visor a la izquierda) 
        para optimizar la ergonomía al trazar las cajas con el ratón.
        """
        # Configuración de pesos: El visor de imágenes (columna 0) toma todo el espacio extra si se agranda la ventana.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        
        # =========================================================
        # PANEL CENTRAL IZQUIERDO (Visor Interactivo de Imágenes)
        # =========================================================
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        
        # Se utiliza un tk.Canvas nativo porque permite operaciones de dibujo de bajo nivel 
        # mucho más rápidas y precisas que los widgets empaquetados.
        self.canvas = tk.Canvas(self.main_frame, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Vinculación de eventos de ratón para el Bounding Box interactivo
        self.canvas.bind("<ButtonPress-1>", self.on_press)    # Inicio del trazo
        self.canvas.bind("<B1-Motion>", self.on_drag)         # Actualización en tiempo real del tamaño
        self.canvas.bind("<ButtonRelease-1>", self.on_release) # Cierre del trazo y cálculo matemático

        # =========================================================
        # PANEL LATERAL DERECHO (Sidebar de Control y Etiquetado)
        # =========================================================
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=1, sticky="nsew") 
        self.sidebar.grid_rowconfigure(6, weight=1) # Empuja los botones de acción hacia el fondo
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Co-MIL Lab", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Carpeta", command=self.load_folder)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)
        
        self.lbl_info = ctk.CTkLabel(self.sidebar, text="Imágenes: 0/0\nResolución: N/A", text_color="gray")
        self.lbl_info.grid(row=2, column=0, padx=20, pady=10)
        
        # Sección de Checkboxes para la Anotación Débil
        self.lbl_etiquetas = ctk.CTkLabel(self.sidebar, text="Anotación Global (Y):", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_etiquetas.grid(row=3, column=0, padx=20, pady=(30, 5), sticky="w")
        
        self.chk_granulacion = ctk.CTkCheckBox(self.sidebar, text="Tejido Granulación", variable=self.var_granulacion)
        self.chk_granulacion.grid(row=4, column=0, padx=20, pady=10, sticky="w")
        
        self.chk_fibrina = ctk.CTkCheckBox(self.sidebar, text="Fibrina", variable=self.var_fibrina)
        self.chk_fibrina.grid(row=5, column=0, padx=20, pady=10, sticky="w")
        
        self.chk_callo = ctk.CTkCheckBox(self.sidebar, text="Tejido Calloso", variable=self.var_callo)
        self.chk_callo.grid(row=6, column=0, padx=20, pady=10, sticky="w")
        
        # Botones de flujo de trabajo
        self.btn_next = ctk.CTkButton(self.sidebar, text="Saltar Imagen", command=self.next_image, 
                                      fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_next.grid(row=7, column=0, padx=20, pady=10)
        
        self.btn_save = ctk.CTkButton(self.sidebar, text="Procesar y Guardar", command=self.process_and_save, 
                                      fg_color="#28a745", hover_color="#218838")
        self.btn_save.grid(row=8, column=0, padx=20, pady=(10, 30))

    # =========================================================
    # LÓGICA DE CARGA Y GESTIÓN DE MEMORIA VISUAL
    # =========================================================
    def load_folder(self):
        """Abre el diálogo del SO y construye el índice de imágenes a procesar."""
        folder_path = filedialog.askdirectory()
        if not folder_path: return 
        
        # Tolera múltiples formatos de compresión fotográfica de los dispositivos móviles
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        
        self.current_idx = 0 
        if self.image_paths:
            self.load_image() 
        else:
            messagebox.showwarning("Aviso", "No se encontraron imágenes en esta carpeta.")

    def load_image(self):
        """
        Carga la imagen cruda en memoria, calcula el factor de escala y la renderiza
        en la interfaz gráfica asegurando que no se pierda la proporción matemática real.
        """
        if self.current_idx >= len(self.image_paths):
            messagebox.showinfo("Fin", "Procesamiento completado para toda la carpeta.")
            return
            
        path = self.image_paths[self.current_idx]
        # Se fuerza la conversión a RGB para estandarizar el canal tensorial (C=3) más adelante.
        self.current_image = Image.open(path).convert('RGB')
        
        # Reseteo del estado lógico para evitar arrastrar etiquetas a la siguiente imagen.
        self.bbox = None
        self.var_granulacion.set(0)
        self.var_fibrina.set(0)
        self.var_callo.set(0)
        if self.rect: self.canvas.delete(self.rect) 
        
        # Sincronización de tareas de la interfaz para obtener tamaños reales del contenedor.
        self.update_idletasks() 
        canvas_w = self.main_frame.winfo_width() - 20
        canvas_h = self.main_frame.winfo_height() - 20
        if canvas_w < 100: canvas_w, canvas_h = 800, 600 
        
        # Cálculo del Scale Factor: Relación entre los píxeles reales y los píxeles de pantalla.
        img_w, img_h = self.current_image.size
        ratio = min(canvas_w/img_w, canvas_h/img_h)
        self.scale_factor = ratio 
        
        # Redimensionamiento temporal para la UI usando filtro LANCZOS (alta calidad antialiasing).
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        img_resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all") 
        
        # Cálculo de los offsets para centrar la imagen matemáticamente en el Canvas.
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image, tags="img")
        
        # Almacenamiento de offsets para corregir las coordenadas del ratón posteriormente.
        self.img_offset_x = x_offset
        self.img_offset_y = y_offset
        self.lbl_info.configure(text=f"Imagen: {self.current_idx + 1} / {len(self.image_paths)}\nOrig: {img_w}x{img_h} px")

    # =========================================================
    # LÓGICA DE EVENTOS E INTERACCIÓN ESPACIAL
    # =========================================================
    def on_press(self, event):
        """Registra el punto de inicio del Bounding Box."""
        self.start_x, self.start_y = event.x, event.y
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="#00ffcc", width=3)

    def on_drag(self, event):
        """Actualiza la renderización geométrica del recuadro dinámicamente."""
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        """
        Cierra la interacción y ejecuta el Mapeo Inverso.
        Traduce las coordenadas de los clics en pantalla a los índices exactos 
        de los píxeles en la imagen cruda de alta resolución (ej. 12MP).
        """
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        
        # Paso 1: Restar los márgenes de centrado (offset) de la UI.
        x1, x2 = x1 - self.img_offset_x, x2 - self.img_offset_x
        y1, y2 = y1 - self.img_offset_y, y2 - self.img_offset_y
        
        # Paso 2: Restricción (clipping) para asegurar que no se guarden coordenadas negativas.
        x1, y1 = max(0, x1), max(0, y1)
        
        # Paso 3: División por el factor de escala para proyectar la caja a la dimensión real.
        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)
        
        self.bbox = (orig_x1, orig_y1, orig_x2, orig_y2)

    # =========================================================
    # NÚCLEO MATEMÁTICO: EXPANSIÓN INTELIGENTE Y EXTRACCIÓN MIL
    # =========================================================
    def smart_expansion_and_extraction(self, original_img, bbox, patch_size=224):
        """
        Transmuta el recorte macroscópico irregular en una Bolsa X de tensores estandarizados (N x C x 224 x 224).
        
        Metodología:
        Para evitar introducir bordes negros artificiales (Zero Padding), este algoritmo 
        calcula cuántos píxeles faltan para lograr un múltiplo de 224 y expande las coordenadas 
        del Bounding Box para capturar tejido real (el cual será filtrado naturalmente por 
        el módulo Gated Attention de la arquitectura Co-MIL).
        
        Edge Case: Si la expansión colisiona con el límite absoluto de la fotografía, 
        emplea 'Reflection Padding' para generar una transición suave sin gradientes abruptos.
        """
        img_w, img_h = original_img.size
        x1, y1, x2, y2 = bbox
        
        # Dimensiones en píxeles de la caja trazada por el usuario.
        w_current = x2 - x1
        h_current = y2 - y1
        
        # Cálculo del tamaño objetivo inmediato superior que sea múltiplo del patch_size (224).
        # math.ceil redondea siempre hacia arriba.
        w_target = math.ceil(w_current / patch_size) * patch_size
        h_target = math.ceil(h_current / patch_size) * patch_size
        
        # Píxeles totales que se deben agregar a la imagen para cumplir la restricción matemática.
        diff_w = w_target - w_current
        diff_h = h_target - h_current
        
        # Expansión Simétrica: Se divide el faltante a la mitad para expandir el centro de la herida 
        # equitativamente hacia afuera (arriba, abajo, izquierda, derecha).
        expand_left = diff_w // 2
        expand_right = diff_w - expand_left
        expand_top = diff_h // 2
        expand_bottom = diff_h - expand_top
        
        # Nuevas coordenadas geométricas virtuales.
        new_x1 = x1 - expand_left
        new_y1 = y1 - expand_top
        new_x2 = x2 + expand_right
        new_y2 = y2 + expand_bottom
        
        # --- PREVENCIÓN DE DESBORDAMIENTO (OUT OF BOUNDS) ---
        # Si la nueva coordenada expandida cae fuera del tamaño de la foto real, 
        # truncamos la coordenada al borde y almacenamos la cantidad de píxeles perdidos 
        # en variables de padding para rellenarlos mediante espejeo más adelante.
        pad_left = pad_right = pad_top = pad_bottom = 0
        
        if new_x1 < 0:
            pad_left = abs(new_x1)
            new_x1 = 0
        if new_y1 < 0:
            pad_top = abs(new_y1)
            new_y1 = 0
        if new_x2 > img_w:
            pad_right = new_x2 - img_w
            new_x2 = img_w
        if new_y2 > img_h:
            pad_bottom = new_y2 - img_h
            new_y2 = img_h
            
        # 1. Recorte biológico puro: Se extrae la sub-imagen optimizada.
        cropped_img = original_img.crop((new_x1, new_y1, new_x2, new_y2))
        
        # Conversión del objeto PIL a un Tensor de PyTorch (rango normalizado 0.0 - 1.0).
        transform = transforms.ToTensor()
        tensor_img = transform(cropped_img)
        C, H, W = tensor_img.shape
        
        # 2. Resolución del Edge Case: Aplicación de Reflection Padding.
        # Solo se ejecuta si la expansión chocó con algún borde. El efecto espejo copia 
        # la textura de la piel hacia afuera, manteniendo la derivada continua y 
        # evitando que MobileNet V2 detecte líneas rectas falsas.
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            pad_transform = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            #unsqueeze y squeeze añaden y quitan la dimensión del batch artificialmente requerida por Pad2d.
            tensor_img = pad_transform(tensor_img.unsqueeze(0)).squeeze(0)
            
        # 3. Creación de la Bolsa (Unfold):
        # El tensor 3D se fragmenta en N parches independientes utilizando una ventana deslizante.
        # unfold(1, ...) opera sobre la altura, unfold(2, ...) opera sobre el ancho.
        patches = tensor_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        
        # Se aplana la cuadrícula bidimensional a una secuencia lineal de parches (N).
        # contiguous() garantiza la alineación contigua en la memoria de la GPU.
        patches = patches.contiguous().view(C, -1, patch_size, patch_size)
        
        # Reordenamiento de dimensiones al estándar de PyTorch para lotes:
        # [Instancias (N), Canales (3), Alto (224), Ancho (224)]
        patches = patches.permute(1, 0, 2, 3) 
        
        return patches

    # =========================================================
    # LÓGICA DE SERIALIZACIÓN Y GUARDADO DE DATOS (DATASET I/O)
    # =========================================================
    def process_and_save(self):
        """
        Ejecuta el pipeline tensorial, empaqueta la Bolsa X y la Etiqueta Y en un diccionario 
        y serializa el objeto directamente a disco rígido en formato binario de PyTorch.
        """
        # Validación de integridad: Evita procesar clics accidentales.
        if not self.bbox or (self.bbox[2]-self.bbox[0] < 10):
            messagebox.showwarning("Atención", "Traza una caja delimitadora sobre la úlcera.")
            return
            
        # Extracción del vector MIML ingresado por el usuario.
        vector_y = [self.var_granulacion.get(), self.var_fibrina.get(), self.var_callo.get()]
        
        # Verificación de falsos negativos.
        if sum(vector_y) == 0:
            if not messagebox.askyesno("Confirmación", "No marcaste tejidos. ¿Guardar como caso negativo?"):
                return
                
        # Llamada al núcleo matemático para generar el tensor [N, 3, 224, 224]
        bolsa_X = self.smart_expansion_and_extraction(self.current_image, self.bbox, patch_size=224)
        
        # Conversión del vector lógico a un Tensor de punto flotante para cálculo de pérdida (BCEWithLogitsLoss).
        y_tensor = torch.tensor(vector_y, dtype=torch.float32)
        
        # Construcción del diccionario estructurado que será consumido por el Custom Dataset Class.
        data_dict = {
            'X': bolsa_X,               
            'Y': y_tensor,              
            'original_file': self.image_paths[self.current_idx] 
        }
        
        # Creación de la estructura de directorios persistente.
        out_dir = os.path.join(os.path.dirname(self.image_paths[0]), "Bolsas_MIL_Procesadas")
        os.makedirs(out_dir, exist_ok=True)
        
        # Serialización de los tensores.
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        out_path = os.path.join(out_dir, f"{base_name}_bag.pt")
        torch.save(data_dict, out_path)
        
        print(f"[{base_name}] -> Bolsa óptima generada: {bolsa_X.shape[0]} parches (Expansión Inteligente).")
        
        # Avance automático en el flujo de trabajo.
        self.next_image()

    def next_image(self):
        """Iterador simple para avanzar en la cola de imágenes."""
        self.current_idx += 1
        self.load_image()

# Punto de entrada de ejecución del script
if __name__ == "__main__":
    app = EtiquetadorCoMIL()
    app.mainloop()