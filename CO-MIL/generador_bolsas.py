"""
=========================================================================================
HERRAMIENTA DE PREPROCESAMIENTO Y ETIQUETADO DÉBIL PARA Co-MIL (MIML)
=========================================================================================

CONTEXTO ARQUITECTÓNICO Y FLUJO DE DATOS:
Este software es el puente entre la captura fotográfica cruda en las clínicas y el 
entrenamiento matemático del modelo Co-MIL. Resuelve la "Paradoja de Resolución" mediante el 
siguiente flujo de procesamiento:

1. Macrolocalización Interactiva (ROI): Delimitación de la úlcera para descartar ruido.
2. Anotación Débil Dinámica (MIML): Generación del vector "Y" global multietiqueta.
   Se soporta la agregación de nuevos tejidos dinámicos durante la sesión clínica.
3. Expansión Inteligente (Smart Crop): Expansión matemática del Bounding Box para 
   alcanzar un múltiplo exacto del tamaño de parche (224, 112, 56 px).
4. Reflection Padding: Efecto espejo en los límites físicos de la foto.
5. Preservación Espacial y Extracción: Empaquetado del tensor X, la etiqueta Y y 
   los metadatos espaciales (grid_shape) necesarios para la reconstrucción de heatmaps.

Ejecución: python CO-MIL/generador_bolsas.py
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

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class EtiquetadorCoMIL(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Co-MIL Tool - Análisis Multi-Etiqueta UPD (WSSS)")
        self.geometry("1250x880")
        
        # ---------------------------------------------------------
        # ESTADO INTERNO Y RUTAS
        # ---------------------------------------------------------
        self.image_paths = []         
        self.current_idx = 0          
        self.processed_files = set()  # Almacena imágenes ya anotadas en sesiones previas
        
        self.current_image = None     
        self.tk_image = None          
        
        self.rect = None              
        self.start_x = None           
        self.start_y = None           
        self.bbox = None              
        
        self.scale_factor = 1.0       
        self.img_offset_x = 0         
        self.img_offset_y = 0         
        
        # ---------------------------------------------------------
        # VARIABLES DINÁMICAS (MIML Y TAMAÑO DE PARCHE)
        # ---------------------------------------------------------
        self.patch_size = 224 # Resolución arquitectónica fijada
        
        # Diccionario dinámico para soportar clases emergentes en la clínica (ej. Tendón)
        self.label_vars = {
            "Tejido Granulación": tk.IntVar(),
            "Fibrina": tk.IntVar(),
            "Tejido Calloso": tk.IntVar()
        }
        self.checkboxes = [] # Referencias visuales a los checkboxes
        
        self.setup_ui()
        
    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        
        # =========================================================
        # PANEL CENTRAL IZQUIERDO (Visor Interactivo)
        # =========================================================
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        
        self.canvas = tk.Canvas(self.main_frame, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)    
        self.canvas.bind("<B1-Motion>", self.on_drag)         
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # =========================================================
        # PANEL LATERAL DERECHO (Control y Etiquetado Dinámico)
        # =========================================================
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=1, sticky="nsew") 
        self.sidebar.grid_rowconfigure(9, weight=1) 
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Co-MIL Lab", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Carpeta Dataset", command=self.load_folder)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)
        
        self.lbl_info = ctk.CTkLabel(self.sidebar, text="Imágenes: 0/0\nPendientes: 0", text_color="gray")
        self.lbl_info.grid(row=2, column=0, padx=20, pady=5)
        
        # --- Configuración de Resolución Adaptativa ---
        self.lbl_patch = ctk.CTkLabel(self.sidebar, text="Resolución de Instancia (px):", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_patch.grid(row=3, column=0, padx=20, pady=(15, 5), sticky="w")
        
        nota_resolucion = (
            "NOTA ESTRUCTURAL:\n"
            "La extracción inicial se fija\n"
            "a 224x224 px (MobileNet V2).\n"
            "Para probar dimensiones menores,\n"
            "utiliza el reprocesador de dataset."
        )
        self.lbl_nota = ctk.CTkLabel(self.sidebar, text=nota_resolucion, font=ctk.CTkFont(size=11, slant="italic"), text_color="#00ffcc", justify="left")
        self.lbl_nota.grid(row=4, column=0, padx=20, pady=(5, 5), sticky="w")

        # --- Etiquetado Multiclase Dinámico ---
        self.lbl_etiquetas = ctk.CTkLabel(self.sidebar, text="Anotación Global (Vector Y):", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_etiquetas.grid(row=5, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.labels_scrollframe = ctk.CTkScrollableFrame(self.sidebar, height=180)
        self.labels_scrollframe.grid(row=6, column=0, padx=20, pady=5, sticky="nsew")
        self.render_checkboxes()
        
        # Agregador de nuevos tejidos
        self.frame_add_class = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_add_class.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        self.entry_new_class = ctk.CTkEntry(self.frame_add_class, placeholder_text="Nuevo tejido...", width=140)
        self.entry_new_class.pack(side=tk.LEFT, padx=(0, 5))
        self.btn_add_class = ctk.CTkButton(self.frame_add_class, text="+", width=40, command=self.add_custom_class)
        self.btn_add_class.pack(side=tk.LEFT)

        # --- Flujo de Trabajo ---
        self.btn_next = ctk.CTkButton(self.sidebar, text="Saltar a Pendiente", command=self.jump_to_next_unprocessed, 
                                      fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_next.grid(row=10, column=0, padx=20, pady=10)
        
        self.btn_save = ctk.CTkButton(self.sidebar, text="Extraer Bolsa y Guardar", command=self.process_and_save, 
                                      fg_color="#28a745", hover_color="#218838")
        self.btn_save.grid(row=11, column=0, padx=20, pady=(10, 30))

    # =========================================================
    # LÓGICA DE UI DINÁMICA Y PERSISTENCIA
    # =========================================================
    def render_checkboxes(self):
        """Renderiza dinámicamente el vector de clases en la interfaz."""
        for chk in self.checkboxes:
            chk.destroy()
        self.checkboxes.clear()
        
        for idx, (label_name, var) in enumerate(self.label_vars.items()):
            chk = ctk.CTkCheckBox(self.labels_scrollframe, text=label_name, variable=var)
            chk.grid(row=idx, column=0, padx=10, pady=8, sticky="w")
            self.checkboxes.append(chk)

    def add_custom_class(self):
        """Permite al experto agregar un tejido raro para posterior análisis de proporciones."""
        new_class = self.entry_new_class.get().strip()
        if new_class and new_class not in self.label_vars:
            self.label_vars[new_class] = tk.IntVar()
            self.render_checkboxes()
            self.entry_new_class.delete(0, tk.END)

    def get_output_dir(self):
        """Retorna el directorio aislado por resolución (ej. Bolsas_MIL_Procesadas/224px)."""
        base_dir = os.path.dirname(self.image_paths[0])
        return os.path.join(base_dir, "Bolsas_MIL_Procesadas", f"{self.patch_size}px")

    def scan_processed_files(self):
        """Escanea el directorio de salida actual para habilitar la reanudación del trabajo."""
        self.processed_files.clear()
        out_dir = self.get_output_dir()
        if os.path.exists(out_dir):
            for file in os.listdir(out_dir):
                if file.endswith("_bag.pt"):
                    base_name = file.replace("_bag.pt", "")
                    self.processed_files.add(base_name)
            
    def jump_to_next_unprocessed(self):
        """Localiza la primera imagen que no exista en el set de archivos procesados."""
        for i, path in enumerate(self.image_paths):
            base_name = os.path.splitext(os.path.basename(path))[0]
            if base_name not in self.processed_files:
                self.current_idx = i
                self.load_image()
                return
        messagebox.showinfo("Completado", "Todas las imágenes para esta resolución han sido procesadas.")

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return 
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        
        if self.image_paths:
            self.scan_processed_files()
            self.jump_to_next_unprocessed() 
        else:
            messagebox.showwarning("Aviso", "No se encontraron imágenes válidas.")

    def update_info_label(self):
        pendientes = len(self.image_paths) - len(self.processed_files)
        self.lbl_info.configure(text=f"Imagen: {self.current_idx + 1} / {len(self.image_paths)}\nPendientes: {pendientes}")

    def load_image(self):
        if self.current_idx >= len(self.image_paths):
            return
            
        path = self.image_paths[self.current_idx]
        self.current_image = Image.open(path).convert('RGB')
        
        self.bbox = None
        for var in self.label_vars.values():
            var.set(0) # Limpieza lógica
        if self.rect: self.canvas.delete(self.rect) 
        
        self.update_idletasks() 
        canvas_w = self.main_frame.winfo_width() - 20
        canvas_h = self.main_frame.winfo_height() - 20
        if canvas_w < 100: canvas_w, canvas_h = 800, 600 
        
        img_w, img_h = self.current_image.size
        ratio = min(canvas_w/img_w, canvas_h/img_h)
        self.scale_factor = ratio 
        
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        img_resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all") 
        
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image, tags="img")
        
        self.img_offset_x = x_offset
        self.img_offset_y = y_offset
        self.update_info_label()

    # =========================================================
    # LÓGICA DE EVENTOS ESPACIALES (Mapeo Inverso)
    # =========================================================
    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="#00ffcc", width=3)

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        
        x1, x2 = x1 - self.img_offset_x, x2 - self.img_offset_x
        y1, y2 = y1 - self.img_offset_y, y2 - self.img_offset_y
        x1, y1 = max(0, x1), max(0, y1)
        
        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)
        
        self.bbox = (orig_x1, orig_y1, orig_x2, orig_y2)

    # =========================================================
    # NÚCLEO MATEMÁTICO: PRESERVACIÓN TOPOLÓGICA Y MIL EXTRACCIÓN
    # =========================================================
    def smart_expansion_and_extraction(self, original_img, bbox, patch_size=224):
        """
        Transmuta el recorte en una Bolsa X y extrae su geometría (grid_shape).
        Retorna la matriz de tensores y un diccionario con los metadatos espaciales
        esenciales para la generación de mapas de calor anatómicos (Heatmaps WSSS).
        """
        img_w, img_h = original_img.size
        x1, y1, x2, y2 = bbox
        
        w_current, h_current = x2 - x1, y2 - y1
        
        w_target = math.ceil(w_current / patch_size) * patch_size
        h_target = math.ceil(h_current / patch_size) * patch_size
        
        diff_w = w_target - w_current
        diff_h = h_target - h_current
        
        expand_left, expand_right = diff_w // 2, diff_w - (diff_w // 2)
        expand_top, expand_bottom = diff_h // 2, diff_h - (diff_h // 2)
        
        new_x1 = x1 - expand_left
        new_y1 = y1 - expand_top
        new_x2 = x2 + expand_right
        new_y2 = y2 + expand_bottom
        
        pad_left = pad_right = pad_top = pad_bottom = 0
        if new_x1 < 0: pad_left, new_x1 = abs(new_x1), 0
        if new_y1 < 0: pad_top, new_y1 = abs(new_y1), 0
        if new_x2 > img_w: pad_right, new_x2 = new_x2 - img_w, img_w
        if new_y2 > img_h: pad_bottom, new_y2 = new_y2 - img_h, img_h
            
        cropped_img = original_img.crop((new_x1, new_y1, new_x2, new_y2))
        
        transform = transforms.ToTensor()
        tensor_img = transform(cropped_img)
        C, H, W = tensor_img.shape
        
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            pad_transform = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            tensor_img = pad_transform(tensor_img.unsqueeze(0)).squeeze(0)
            H, W = tensor_img.shape[1], tensor_img.shape[2] # Actualizamos dimensiones tras padding
            
        # Extracción topológica: ¿Cuántos parches hay a lo alto y ancho?
        grid_h = H // patch_size
        grid_w = W // patch_size
        
        # Despliegue a la bolsa
        patches = tensor_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(C, -1, patch_size, patch_size)
        patches = patches.permute(1, 0, 2, 3) 
        
        # Diccionario de metadatos espaciales preservados para WSSS
        spatial_metadata = {
            'grid_shape': (grid_h, grid_w),
            'patch_size': patch_size,
            'user_bbox': bbox, # CRÍTICO: Guarda el trazo original del experto
            'bbox_expanded': (new_x1, new_y1, new_x2, new_y2),
            'padding_applied': (pad_left, pad_right, pad_top, pad_bottom)
        }
        
        return patches, spatial_metadata

    # =========================================================
    # SERIALIZACIÓN ESTRUCTURADA
    # =========================================================
    def process_and_save(self):
        if not self.bbox or (self.bbox[2]-self.bbox[0] < 10):
            messagebox.showwarning("Atención", "Traza una caja delimitadora sobre la úlcera.")
            return
            
        # Captura dinámica de las etiquetas
        vector_y = [var.get() for var in self.label_vars.values()]
        class_names = list(self.label_vars.keys())
        
        if sum(vector_y) == 0:
            if not messagebox.askyesno("Confirmación", "No marcaste tejidos. ¿Guardar como caso negativo?"):
                return
                
        bolsa_X, spatial_meta = self.smart_expansion_and_extraction(self.current_image, self.bbox, patch_size=self.patch_size)
        y_tensor = torch.tensor(vector_y, dtype=torch.float32)
        
        # Nuevo empaquetado optimizado para torchmil y análisis anatómico
        data_dict = {
            'X': bolsa_X,               
            'Y': y_tensor,              
            'class_names': class_names, # Mapeo lógico de las clases
            'spatial_metadata': spatial_meta, # Vital para reconstruir la forma de la herida
            'original_file': self.image_paths[self.current_idx] 
        }
        
        out_dir = self.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        out_path = os.path.join(out_dir, f"{base_name}_bag.pt")
        torch.save(data_dict, out_path)
        
        print(f"[{base_name}] -> Bolsa ({self.patch_size}px) generada. Grid: {spatial_meta['grid_shape']}")
        
        # Persistencia: Agregamos a procesados y saltamos al siguiente pendiente
        self.processed_files.add(base_name)
        self.jump_to_next_unprocessed()

if __name__ == "__main__":
    app = EtiquetadorCoMIL()
    app.mainloop()