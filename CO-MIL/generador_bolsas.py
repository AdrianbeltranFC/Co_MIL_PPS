"""
=========================================================================================
HERRAMIENTA CLÍNICA DE PREPROCESAMIENTO Y ETIQUETADO DÉBIL PARA Co-MIL (MIML)
=========================================================================================
Descripción General:
Este software es el módulo de ingesta e interfaz gráfica de la arquitectura Co-MIL para 
el análisis de Úlceras de Pie Diabético (UPD). Su objetivo es permitir a expertos médicos 
delimitar zonas de interés (ROIs) en fotografías clínicas de alta resolución y asignarles 
múltiples etiquetas (Multi-Instance Multi-Label) sin necesidad de segmentación a nivel píxel.

Flujo de Procesamiento Matemático y Lógico:
1. Macrolocalización: El usuario traza cajas delimitadoras sobre las regiones con úlceras.
   Se soportan múltiples ROIs independientes por fotografía.
2. Expansión Inteligente (Smart Crop): Para evitar el 'Zero-Padding' que destruye 
   gradientes, la caja trazada se expande matemáticamente para alcanzar dimensiones 
   múltiplos de 224x224 px, absorbiendo tejido real.
3. Reflection Padding: Si la expansión colisiona con los bordes de la foto, se aplica 
   un efecto espejo preservando la continuidad de la textura de la piel.
4. Extracción de Bolsas (Unfold): El tensor resultante se fragmenta en N parches de 224px.
5. Serialización (.pt): Se empaqueta la malla de tensores (Bolsa X), el vector de 
   etiquetas (Y) y los metadatos espaciales en un binario nativo de PyTorch, listo 
   para el entrenamiento de la red.

Características de la Interfaz:
- Autoguardado silencioso de sesiones (archivos JSON) para no perder progreso.
- Panel lateral deslizable para adaptarse a laptops con pantallas pequeñas.
- "Visión de Parches" (Modo Investigador) independiente y minimizable para inspección visual.
- Registro de autoría (Firma del anotador) y manifiestos en CSV/XLSX.
- Flexibilidad para renombrar clases base mediante una ventana modal independiente.

Uso: 
    python generador_bolsas.py
=========================================================================================
"""

import csv
import json
import math
import os
import sys
import tkinter as tk
import threading
import unicodedata
import zipfile
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import torch
import torchvision.transforms as transforms

# Intento de importación de openpyxl para crear Excel de forma nativa
try:
    from openpyxl import Workbook, load_workbook
except ImportError:
    Workbook = None
    load_workbook = None

import catalogo_tejidos
from catalogo_tejidos import limpiar_texto as clean_label_text, normalizar_clave as normalize_label_key

# Configuración visual de la librería gráfica
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BBox = Tuple[int, int, int, int]


def get_app_dir() -> str:
    """Retorna el directorio de la aplicación, compatible con PyInstaller (.exe)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


class EtiquetadorCoMIL(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Co-MIL Tool - Análisis Clínico Multi-Etiqueta (UPD)")
        self.geometry("1380x930")

        # Nombres de archivos de registro y manifiesto
        self.app_dir = get_app_dir()
        self.registry_filename = "custom_classes.json"
        self.dataset_catalog_filename = "class_catalog.json"
        self.session_dirname = "_autosave_sessions"
        self.manifest_csv_filename = "manifest_bolsas.csv"
        self.manifest_xlsx_filename = "manifest_bolsas.xlsx"

        # ---------------------------------------------------------
        # ESTADO INTERNO Y RUTAS
        # ---------------------------------------------------------
        self.image_paths: List[str] = []
        self.current_idx = 0
        self.processed_files = set()
        self.dataset_input_dir: Optional[str] = None
        self.dataset_output_root: Optional[str] = None

        self.current_image: Optional[Image.Image] = None
        self.tk_image = None

        self.pending_regions: List[Dict[str, object]] = []
        self.selected_region_idx: Optional[int] = None

        # Variables espaciales del Canvas
        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.bbox: Optional[BBox] = None
        self.draft_rect_id = None
        self.region_canvas_ids: List[int] = []

        self.scale_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.rendered_image_size = (0, 0)
        self.suspend_autosave = False
        self.autosave_job = None
        self.autosave_delay_ms = 650
        self.manifest_worker = None
        self.min_roi_size_px = 20
        self.min_roi_size_canvas = 10

        # ---------------------------------------------------------
        # CATÁLOGO DE CLASES
        # ---------------------------------------------------------
        self.patch_size = 224
        # Lista maestra de clases base, definida en catalogo_tejidos.py (fuente
        # única compartida con dataset.py). Al compilar el .exe, estas clases
        # serán las que el experto verá por defecto.
        self.base_classes = list(catalogo_tejidos.CLASES_BASE)

        self.class_catalog = self.load_app_class_catalog()
        self.custom_classes = self.extract_custom_classes(self.class_catalog)
        self.label_vars: Dict[str, tk.IntVar] = {}
        self.checkboxes = []
        self.entry_vars: Dict[int, tk.StringVar] = {} # Usado en modal de renombrado
        self._rebuild_label_vars()

        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_app_close)

    # =========================================================
    # PERSISTENCIA DE CLASES Y CATÁLOGO
    # =========================================================
    def get_registry_file_path(self) -> str:
        return os.path.join(self.app_dir, self.registry_filename)

    def get_dataset_catalog_path(self) -> Optional[str]:
        if not self.dataset_output_root:
            return None
        return os.path.join(self.dataset_output_root, self.dataset_catalog_filename)

    def dedupe_labels(self, labels: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for label in labels:
            cleaned = clean_label_text(label)
            if not cleaned: continue
            key = normalize_label_key(cleaned)
            if key in seen: continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def extract_custom_classes(self, class_catalog: List[str]) -> List[str]:
        base_keys = {normalize_label_key(name) for name in self.base_classes}
        return [name for name in class_catalog if normalize_label_key(name) not in base_keys]

    def load_catalog_from_file(self, file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            return list(self.base_classes)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return list(self.base_classes)

        if isinstance(data, dict):
            if isinstance(data.get("class_catalog"), list):
                labels = data["class_catalog"]
            else:
                labels = list(self.base_classes) + list(data.get("custom_classes", []))
        elif isinstance(data, list):
            labels = data
        else:
            labels = list(self.base_classes)

        return self.dedupe_labels(list(self.base_classes) + list(labels))

    def save_catalog_to_file(self, file_path: str):
        payload = {
            "version": 3,
            "base_classes": self.base_classes,
            "custom_classes": self.custom_classes,
            "class_catalog": self.class_catalog,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_app_class_catalog(self) -> List[str]:
        return self.load_catalog_from_file(self.get_registry_file_path())

    def save_app_class_catalog(self):
        try: self.save_catalog_to_file(self.get_registry_file_path())
        except Exception: pass

    def save_dataset_catalog(self):
        catalog_path = self.get_dataset_catalog_path()
        if not catalog_path: return
        try:
            os.makedirs(self.dataset_output_root, exist_ok=True)
            self.save_catalog_to_file(catalog_path)
        except Exception: pass

    def find_existing_label(self, candidate_label: str) -> Optional[str]:
        candidate_key = normalize_label_key(candidate_label)
        for label in self.class_catalog:
            if normalize_label_key(label) == candidate_key:
                return label
        return None

    def _rebuild_label_vars(self, previous_selection: Optional[Dict[str, int]] = None):
        previous_selection = previous_selection or {}
        self.label_vars = {}
        for class_name in self.class_catalog:
            value = int(bool(previous_selection.get(class_name, 0)))
            var = tk.IntVar(value=value)
            var.trace_add("write", self.on_label_selection_change)
            self.label_vars[class_name] = var

    def update_catalog(self, labels: List[str], preserve_selection: bool = True, preselected=None):
        previous_selection = {}
        if preserve_selection:
            previous_selection = {name: var.get() for name, var in self.label_vars.items()}
        if preselected:
            previous_selection.update(preselected)

        new_catalog = self.dedupe_labels(list(self.base_classes) + list(labels))
        if new_catalog == self.class_catalog: return False

        self.class_catalog = new_catalog
        self.custom_classes = self.extract_custom_classes(self.class_catalog)
        self._rebuild_label_vars(previous_selection)
        if hasattr(self, "labels_scrollframe"):
            self.render_checkboxes()

        self.save_app_class_catalog()
        self.save_dataset_catalog()
        self.sync_pending_regions_to_catalog()
        return True

    def labels_to_vector(self, labels: List[str]) -> torch.Tensor:
        vector = torch.zeros(len(self.class_catalog), dtype=torch.float32)
        label_keys = {normalize_label_key(label) for label in labels}
        for idx, class_name in enumerate(self.class_catalog):
            if normalize_label_key(class_name) in label_keys:
                vector[idx] = 1.0
        return vector

    def get_selected_labels(self) -> List[str]:
        return [name for name, var in self.label_vars.items() if var.get() == 1]

    def set_labels_from_list(self, labels: List[str]):
        label_keys = {normalize_label_key(label) for label in labels}
        for class_name, var in self.label_vars.items():
            var.set(1 if normalize_label_key(class_name) in label_keys else 0)

    def clear_label_selection(self):
        for var in self.label_vars.values(): var.set(0)

    def sync_pending_regions_to_catalog(self):
        for region in self.pending_regions:
            normalized_labels = []
            for label in region["labels"]:
                existing = self.find_existing_label(label)
                normalized_labels.append(existing or clean_label_text(label))
            region["labels"] = self.dedupe_labels(normalized_labels)
        if hasattr(self, "roi_listbox"):
            self.refresh_region_list()
            self.redraw_regions()

    def on_label_selection_change(self, *_args):
        if not self.suspend_autosave: self.schedule_autosave()

    # =========================================================
    # LÓGICA DE RENOMBRADO DE CLASES (VENTANA MODAL)
    # =========================================================
    def open_renamer_window(self):
        """Abre una ventana modal elegante para que el experto renombre etiquetas."""
        win_rename = ctk.CTkToplevel(self)
        win_rename.title("Renombrar Catálogo Maestro")
        win_rename.geometry("500x600")
        win_rename.grab_set() # Esta SÍ debe ser modal para no editar sobre la imagen al mismo tiempo

        lbl_info = ctk.CTkLabel(win_rename, text="Edita los nombres de las clases base.\nLos cambios se guardarán en el catálogo general.", font=ctk.CTkFont(weight="bold"))
        lbl_info.pack(pady=15, padx=20)

        scroll_renamer = ctk.CTkScrollableFrame(win_rename)
        scroll_renamer.pack(fill="both", expand=True, padx=20, pady=10)

        self.entry_vars = {}
        for idx, class_name in enumerate(self.base_classes):
            row = ctk.CTkFrame(scroll_renamer, fg_color="transparent")
            row.pack(fill="x", pady=5)
            
            e_var = tk.StringVar(value=class_name)
            self.entry_vars[idx] = e_var
            
            entry = ctk.CTkEntry(row, textvariable=e_var)
            entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
            
            btn = ctk.CTkButton(row, text="Guardar", width=60, fg_color="#0056b3", hover_color="#004494", command=lambda i=idx: self.apply_class_rename(i))
            btn.pack(side="right")
            
        btn_close = ctk.CTkButton(win_rename, text="Cerrar", command=win_rename.destroy, fg_color="#4B5563", hover_color="#374151")
        btn_close.pack(pady=15)

    def apply_class_rename(self, idx: int):
        """Aplica el renombrado de la clase base en el índice dado."""
        old_name = self.base_classes[idx]
        new_name = clean_label_text(self.entry_vars[idx].get())
        
        if not new_name or new_name == old_name:
            self.entry_vars[idx].set(old_name) 
            return

        if self.find_existing_label(new_name):
            messagebox.showwarning("Atención", f"El nombre '{new_name}' ya existe en el catálogo.")
            self.entry_vars[idx].set(old_name)
            return

        if not messagebox.askyesno("Confirmación", f"¿Seguro que quieres renombrar la clase base '{old_name}' a '{new_name}'?\n\nEsto actualizará el catálogo maestro y el manifiesto."):
            self.entry_vars[idx].set(old_name)
            return

        # Actualizar base_classes
        self.base_classes[idx] = new_name
        self.class_catalog = self.dedupe_labels(list(self.base_classes) + list(self.custom_classes))
        
        # Sincronizar UI (Checkboxes)
        current_sel = {name: var.get() for name, var in self.label_vars.items()}
        if current_sel.get(old_name, 0):
            current_sel[new_name] = current_sel.pop(old_name)
            
        self._rebuild_label_vars(current_sel)
        self.render_checkboxes()
        
        # Sincronizar datos de ROI pendientes
        for region in self.pending_regions:
            if old_name in region["labels"]:
                region["labels"] = [new_name if l == old_name else l for l in region["labels"]]
        self.refresh_region_list()

        self.save_app_class_catalog()
        # Registrar el renombrado para que las bolsas ya guardadas con el
        # nombre anterior se sigan resolviendo bien contra el catálogo vigente
        # (ver catalogo_tejidos.py / dataset.py).
        catalogo_tejidos.registrar_renombre(self.app_dir, old_name, new_name)
        messagebox.showinfo("Éxito", f"Clase renombrada a '{new_name}'")

    # =========================================================
    # AUTOGUARDADO DE SESIONES
    # =========================================================
    def schedule_autosave(self, immediate: bool = False):
        if self.suspend_autosave: return
        if immediate:
            self.flush_autosave_now()
            return
        self.cancel_pending_autosave()
        self.autosave_job = self.after(self.autosave_delay_ms, self._flush_autosave)

    def cancel_pending_autosave(self):
        if self.autosave_job is not None:
            try: self.after_cancel(self.autosave_job)
            except Exception: pass
            self.autosave_job = None

    def _flush_autosave(self):
        self.autosave_job = None
        self.save_session_state()

    def flush_autosave_now(self):
        if not self.suspend_autosave:
            self.cancel_pending_autosave()
            self.save_session_state()

    def on_app_close(self):
        self.flush_autosave_now()
        self.destroy()

    def get_session_dir(self) -> Optional[str]:
        if not self.dataset_output_root: return None
        return os.path.join(self.dataset_output_root, self.session_dirname)

    def get_current_image_base_name(self) -> Optional[str]:
        if not self.image_paths or self.current_idx >= len(self.image_paths): return None
        return os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]

    def get_session_file_path(self, image_base_name: Optional[str] = None) -> Optional[str]:
        session_dir = self.get_session_dir()
        if not session_dir: return None
        image_base_name = image_base_name or self.get_current_image_base_name()
        if not image_base_name: return None
        return os.path.join(session_dir, f"{image_base_name}.json")

    def serialize_bbox(self, bbox: Optional[BBox]):
        return [int(v) for v in bbox] if bbox else None

    def deserialize_bbox(self, bbox_data) -> Optional[BBox]:
        if not bbox_data or len(bbox_data) != 4: return None
        return tuple(int(v) for v in bbox_data)

    def save_session_state(self):
        session_path = self.get_session_file_path()
        if not session_path: return
        os.makedirs(os.path.dirname(session_path), exist_ok=True)
        payload = {
            "image_path": self.image_paths[self.current_idx],
            "pending_regions": [
                {"bbox": self.serialize_bbox(r["bbox"]), "labels": list(r["labels"])}
                for r in self.pending_regions
            ],
            "current_bbox": self.serialize_bbox(self.bbox),
            "current_labels": self.get_selected_labels(),
            "selected_region_idx": self.selected_region_idx,
            "saved_at": datetime.now().isoformat()
        }
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception: pass

    def restore_session_state(self):
        session_path = self.get_session_file_path()
        if not session_path or not os.path.exists(session_path): return False
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception: return False

        self.suspend_autosave = True
        try:
            self.pending_regions = [
                {"bbox": self.deserialize_bbox(r["bbox"]), "labels": self.dedupe_labels([self.find_existing_label(l) or clean_label_text(l) for l in r.get("labels", [])])}
                for r in data.get("pending_regions", []) if self.deserialize_bbox(r.get("bbox"))
            ]
            self.bbox = self.deserialize_bbox(data.get("current_bbox"))
            self.clear_label_selection()
            self.set_labels_from_list([self.find_existing_label(l) or clean_label_text(l) for l in data.get("current_labels", [])])
            
            self.selected_region_idx = data.get("selected_region_idx")
            if not isinstance(self.selected_region_idx, int) or self.selected_region_idx >= len(self.pending_regions):
                self.selected_region_idx = None

            self.refresh_region_list()
            self.redraw_regions()
            self.update_info_label()
        finally:
            self.suspend_autosave = False
        return True

    def clear_session_state(self, image_base_name: Optional[str] = None):
        session_path = self.get_session_file_path(image_base_name)
        if session_path and os.path.exists(session_path):
            try: os.remove(session_path)
            except OSError: pass

    # =========================================================
    # UI: DISEÑO CON BARRA DESLIZANTE, FIRMA, INSPECTOR Y RENOMBRADOR
    # =========================================================
    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        # Panel Central: Visor de imágenes
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")

        self.canvas = tk.Canvas(self.main_frame, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Panel Lateral Deslizable
        self.sidebar = ctk.CTkScrollableFrame(self, width=420, corner_radius=0)
        self.sidebar.grid(row=0, column=1, sticky="nsew")

        # Encabezado
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Co-MIL Tool", font=ctk.CTkFont(size=26, weight="bold"))
        self.logo_label.pack(pady=(20, 5))
        
        self.lbl_vsss = ctk.CTkLabel(self.sidebar, text="Visión Computacional & MIML Clínico", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray")
        self.lbl_vsss.pack(pady=(0, 15))

        # Firma del Experto
        self.entry_annotator = ctk.CTkEntry(self.sidebar, placeholder_text="Firma (Ej. Nombre Apellido)", justify="center", font=ctk.CTkFont(weight="bold"))
        self.entry_annotator.pack(fill="x", padx=20, pady=(0, 15))

        # Carga y Navegación
        self.btn_load = ctk.CTkButton(self.sidebar, text="Cargar Carpeta Dataset", command=self.load_folder)
        self.btn_load.pack(fill="x", padx=20, pady=10)

        self.lbl_info = ctk.CTkLabel(self.sidebar, text="Imágenes: 0/0\nEstado: Esperando", text_color="gray", justify="center")
        self.lbl_info.pack(pady=(0, 10))

        self.frame_nav = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_nav.pack(fill="x", padx=20, pady=(0, 15))
        self.frame_nav.grid_columnconfigure((0, 1), weight=1)

        self.btn_prev_image = ctk.CTkButton(self.frame_nav, text="Anterior", command=self.go_to_previous_image, fg_color="#374151", hover_color="#4B5563")
        self.btn_prev_image.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.btn_next_image = ctk.CTkButton(self.frame_nav, text="Siguiente", command=self.go_to_next_image, fg_color="#374151", hover_color="#4B5563")
        self.btn_next_image.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        self.btn_jump_pending = ctk.CTkButton(self.sidebar, text="Saltar a Próxima Pendiente", command=self.jump_to_next_unprocessed, fg_color="transparent", border_width=1, text_color="#DCE4EE")
        self.btn_jump_pending.pack(fill="x", padx=20, pady=(0, 15))

        # Anotación (Checkboxes) y Renombrador
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#4B5563").pack(fill="x", padx=20, pady=10)
        
        self.lbl_etiquetas = ctk.CTkLabel(self.sidebar, text="Anotación de la ROI actual:", font=ctk.CTkFont(size=15, weight="bold"))
        self.lbl_etiquetas.pack(anchor="w", padx=20, pady=(5, 5))

        # Botón para abrir modal de renombrado
        self.btn_renamer = ctk.CTkButton(self.sidebar, text="⚙️ Renombrar etiquetas del catálogo", fg_color="#374151", hover_color="#4B5563", command=self.open_renamer_window)
        self.btn_renamer.pack(fill="x", padx=20, pady=(0, 5))

        self.labels_scrollframe = ctk.CTkScrollableFrame(self.sidebar, height=200, fg_color="#1F2937", corner_radius=5)
        self.labels_scrollframe.pack(fill="x", padx=20, pady=5)
        self.render_checkboxes()

        self.frame_add_class = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_add_class.pack(fill="x", padx=20, pady=(5, 15))
        self.entry_new_class = ctk.CTkEntry(self.frame_add_class, placeholder_text="Añadir tejido raro...")
        self.entry_new_class.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_add_class = ctk.CTkButton(self.frame_add_class, text="+", width=40, command=self.add_custom_class)
        self.btn_add_class.pack(side="right")

        # ROIs Múltiples
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#4B5563").pack(fill="x", padx=20, pady=10)
        
        self.frame_roi_actions = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_roi_actions.pack(fill="x", padx=20, pady=5)
        self.frame_roi_actions.grid_columnconfigure((0, 1), weight=1)
        
        self.btn_add_roi = ctk.CTkButton(self.frame_roi_actions, text="Guardar ROI Local", command=self.add_or_update_region, fg_color="#2f9e44", hover_color="#2b8a3e")
        self.btn_add_roi.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.btn_clear_current = ctk.CTkButton(self.frame_roi_actions, text="Borrar Trazo", command=self.clear_current_region, fg_color="transparent", border_width=1)
        self.btn_clear_current.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        self.lbl_regions = ctk.CTkLabel(self.sidebar, text="ROIs guardadas (esta imagen):", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_regions.pack(anchor="w", padx=20, pady=(15, 5))

        self.roi_listbox = tk.Listbox(self.sidebar, height=4, bg="#202833", fg="#e5eef8", selectbackground="#0d6efd", relief="flat")
        self.roi_listbox.pack(fill="x", padx=20, pady=5)
        self.roi_listbox.bind("<<ListboxSelect>>", self.on_region_select)

        self.btn_remove_roi = ctk.CTkButton(self.sidebar, text="Eliminar ROI seleccionada", command=self.remove_selected_region, fg_color="#7b1f2b", hover_color="#982436")
        self.btn_remove_roi.pack(fill="x", padx=20, pady=(5, 10))

        # Modos y Finalización
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#4B5563").pack(fill="x", padx=20, pady=10)

        self.btn_inspect = ctk.CTkButton(self.sidebar, text="🔍 Visión de Parches (Inspector MIL)", command=self.show_patch_inspector, fg_color="#4B5563", hover_color="#374151")
        self.btn_inspect.pack(fill="x", padx=20, pady=(0, 15))
        self.btn_inspect.configure(state="disabled")

        self.btn_save = ctk.CTkButton(self.sidebar, text="PROCESAR IMAGEN & CONTINUAR", command=self.process_and_save, font=ctk.CTkFont(weight="bold"), fg_color="#0056b3", hover_color="#004494", height=45)
        self.btn_save.pack(fill="x", padx=20, pady=(10, 20))

        self.btn_exit = ctk.CTkButton(self.sidebar, text="SALIR & CERRAR APLICACIÓN", command=self.on_app_close, fg_color="#7b1f2b", hover_color="#982436")
        self.btn_exit.pack(fill="x", padx=20, pady=(5, 30))

    def render_checkboxes(self):
        for chk in self.checkboxes: chk.destroy()
        self.checkboxes.clear()
        for idx, (label_name, var) in enumerate(self.label_vars.items()):
            chk = ctk.CTkCheckBox(self.labels_scrollframe, text=label_name, variable=var)
            chk.grid(row=idx, column=0, padx=10, pady=8, sticky="w")
            self.checkboxes.append(chk)

    # =========================================================
    # LÓGICA DE ROIs MÚLTIPLES E INTERACCIÓN ESPACIAL
    # =========================================================
    def add_custom_class(self):
        new_class = clean_label_text(self.entry_new_class.get())
        if not new_class: return
        existing_name = self.find_existing_label(new_class)
        if existing_name:
            self.label_vars[existing_name].set(1)
            self.entry_new_class.delete(0, tk.END)
            return
        if self.update_catalog(self.class_catalog + [new_class], preselected={new_class: 1}):
            self.entry_new_class.delete(0, tk.END)

    def build_region_payload(self) -> Optional[Dict[str, object]]:
        if not self.bbox or (self.bbox[2] - self.bbox[0] < 10):
            messagebox.showwarning("Atención", "Traza una caja delimitadora válida sobre la úlcera.")
            return None
        selected = self.get_selected_labels()
        if not selected and not messagebox.askyesno("Confirmación", "No marcaste tejidos. ¿Guardar como caso negativo?"):
            return None
        return {"bbox": self.bbox, "labels": list(selected)}

    def add_or_update_region(self):
        region = self.build_region_payload()
        if not region: return
        if self.selected_region_idx is None:
            self.pending_regions.append(region)
        else:
            self.pending_regions[self.selected_region_idx] = region
        self.clear_current_region(reset_selection=True)
        self.refresh_region_list()
        self.schedule_autosave(immediate=True)

    def remove_selected_region(self):
        if self.selected_region_idx is not None and self.selected_region_idx < len(self.pending_regions):
            del self.pending_regions[self.selected_region_idx]
            self.clear_current_region(reset_selection=True)
            self.refresh_region_list()
            self.schedule_autosave(immediate=True)

    def clear_current_region(self, reset_selection: bool = True):
        self.bbox = None
        self.clear_label_selection()
        if reset_selection:
            self.selected_region_idx = None
            self.roi_listbox.selection_clear(0, tk.END)
            self.btn_inspect.configure(state="disabled")
        self.redraw_regions()
        self.schedule_autosave()

    def on_region_select(self, _event=None):
        selection = self.roi_listbox.curselection()
        if not selection: 
            self.btn_inspect.configure(state="disabled")
            return
        idx = int(selection[0])
        self.selected_region_idx = idx
        self.bbox = self.pending_regions[idx]["bbox"]
        self.set_labels_from_list(self.pending_regions[idx]["labels"])
        self.redraw_regions()
        self.btn_inspect.configure(state="normal")

    def refresh_region_list(self):
        self.roi_listbox.delete(0, tk.END)
        for idx, r in enumerate(self.pending_regions, 1):
            lbls = ", ".join(r["labels"][:3]) if r["labels"] else "Negativo"
            self.roi_listbox.insert(tk.END, f"ROI {idx:02d} | {lbls}")
        if self.selected_region_idx is not None:
            self.roi_listbox.selection_set(self.selected_region_idx)
            self.btn_inspect.configure(state="normal")
        else:
            self.btn_inspect.configure(state="disabled")

    # =========================================================
    # NAVEGACIÓN Y RENDERIZADO VISUAL
    # =========================================================
    def get_output_dir(self) -> str:
        return os.path.join(self.dataset_output_root, f"{self.patch_size}px") if self.dataset_output_root else ""

    def scan_processed_files(self):
        self.processed_files.clear()
        out_dir = self.get_output_dir()
        if not os.path.exists(out_dir): return
        for file in os.listdir(out_dir):
            if file.endswith("_bag.pt"):
                raw_name = file[:-7]
                base_name = raw_name.split("__roi_")[0] if "__roi_" in raw_name else raw_name
                self.processed_files.add(base_name)

    def jump_to_next_unprocessed(self):
        for idx, path in enumerate(self.image_paths):
            if os.path.splitext(os.path.basename(path))[0] not in self.processed_files:
                self.navigate_to_index(idx)
                return
        messagebox.showinfo("Completado", "Todas las imágenes procesadas.")

    def navigate_to_index(self, idx: int):
        if 0 <= idx < len(self.image_paths):
            self.flush_autosave_now()
            self.current_idx = idx
            self.load_image()

    def go_to_previous_image(self): self.navigate_to_index(self.current_idx - 1)
    def go_to_next_image(self): self.navigate_to_index(self.current_idx + 1)

    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)])
        if not self.image_paths:
            messagebox.showwarning("Aviso", "No se encontraron imágenes válidas.")
            return
        self.dataset_input_dir = folder
        self.dataset_output_root = os.path.join(folder, "Bolsas_MIL_Procesadas")
        self.scan_processed_files()
        self.jump_to_next_unprocessed()

    def update_info_label(self):
        pendientes = len(self.image_paths) - len(self.processed_files)
        base = self.get_current_image_base_name()
        estado = "✅ Procesada" if base in self.processed_files else "⏳ Pendiente"
        self.lbl_info.configure(text=f"Imagen: {self.current_idx + 1} / {len(self.image_paths)}\n{estado}\nFaltan: {pendientes}")

    def load_image(self):
        if self.current_idx >= len(self.image_paths): return
        self.current_image = Image.open(self.image_paths[self.current_idx]).convert("RGB")
        self.suspend_autosave = True
        self.pending_regions, self.selected_region_idx, self.bbox = [], None, None
        self.clear_label_selection()
        self.refresh_region_list()

        self.update_idletasks()
        cw, ch = max(800, self.main_frame.winfo_width() - 20), max(600, self.main_frame.winfo_height() - 20)
        img_w, img_h = self.current_image.size
        self.scale_factor = min(cw / img_w, ch / img_h)
        new_w, new_h = int(img_w * self.scale_factor), int(img_h * self.scale_factor)
        
        self.tk_image = ImageTk.PhotoImage(self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS))
        self.rendered_image_size = (new_w, new_h)
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        
        self.img_offset_x, self.img_offset_y = (cw - new_w) // 2, (ch - new_h) // 2
        self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor=tk.NW, image=self.tk_image, tags="img")

        self.redraw_regions()
        self.btn_inspect.configure(state="disabled") 

        if not self.restore_session_state(): self.save_session_state()
        self.update_info_label()
        self.suspend_autosave = False

    def image_to_canvas_bbox(self, bbox: BBox):
        return (self.img_offset_x + int(bbox[0] * self.scale_factor), self.img_offset_y + int(bbox[1] * self.scale_factor),
                self.img_offset_x + int(bbox[2] * self.scale_factor), self.img_offset_y + int(bbox[3] * self.scale_factor))

    def redraw_regions(self):
        if not hasattr(self, "canvas"): return
        for cid in self.region_canvas_ids: self.canvas.delete(cid)
        self.region_canvas_ids.clear()
        if self.draft_rect_id: self.canvas.delete(self.draft_rect_id); self.draft_rect_id = None

        for idx, region in enumerate(self.pending_regions):
            x1, y1, x2, y2 = self.image_to_canvas_bbox(region["bbox"])
            color = "#4da3ff" if idx == self.selected_region_idx else "#ffb347"
            self.region_canvas_ids.extend([
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if idx == self.selected_region_idx else 2),
                self.canvas.create_text(x1 + 5, y1 + 5, anchor=tk.NW, text=f"ROI {idx + 1:02d}", fill=color, font=("Arial", 10, "bold"))
            ])
        if self.bbox:
            x1, y1, x2, y2 = self.image_to_canvas_bbox(self.bbox)
            self.draft_rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ffcc", width=3, dash=(6, 4))
            
            if self.selected_region_idx is None:
                self.btn_inspect.configure(state="normal")
            else:
                 self.btn_inspect.configure(state="normal")

    def on_press(self, event):
        if self.selected_region_idx is not None:
            self.clear_current_region(reset_selection=True)

        self.start_x, self.start_y = event.x, event.y
        if self.draft_rect_id: self.canvas.delete(self.draft_rect_id)
        self.draft_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="#00ffcc", width=3, dash=(6, 4))

    def on_drag(self, event):
        if self.draft_rect_id: self.canvas.coords(self.draft_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        if abs(x2 - x1) < self.min_roi_size_canvas or abs(y2 - y1) < self.min_roi_size_canvas:
            self.bbox = None; self.redraw_regions(); return

        x1, y1 = max(0, x1 - self.img_offset_x), max(0, y1 - self.img_offset_y)
        x2, y2 = min(self.rendered_image_size[0], x2 - self.img_offset_x), min(self.rendered_image_size[1], y2 - self.img_offset_y)
        orig_bbox = (int(x1 / self.scale_factor), int(y1 / self.scale_factor), int(x2 / self.scale_factor), int(y2 / self.scale_factor))
        
        if (orig_bbox[2] - orig_bbox[0]) < self.min_roi_size_px or (orig_bbox[3] - orig_bbox[1]) < self.min_roi_size_px:
            self.bbox = None
        else:
            self.bbox = orig_bbox
        self.redraw_regions()
        self.schedule_autosave()

    # =========================================================
    # VISUALIZADOR DE PARCHES (PANEL DE ANÁLISIS PROFESIONAL Y ZOOM)
    # =========================================================
    def show_patch_inspector(self):
        roi_bbox: Optional[BBox] = None
        current_labels: List[str] = []
        is_saved_roi: bool = False
        saved_roi_idx: Optional[int] = None

        if self.selected_region_idx is not None and self.selected_region_idx < len(self.pending_regions):
            roi_data = self.pending_regions[self.selected_region_idx]
            roi_bbox = roi_data["bbox"]
            current_labels = roi_data["labels"]
            is_saved_roi = True
            saved_roi_idx = self.selected_region_idx
        elif self.bbox:
            roi_bbox = self.bbox
            current_labels = self.get_selected_labels()
            is_saved_roi = False

        if not roi_bbox:
            messagebox.showinfo("Aviso", "Traza una ROI o selecciona una ROI guardada para poder inspeccionarla.")
            self.btn_inspect.configure(state="disabled")
            return

        bolsa_x, meta = self.smart_expansion_and_extraction(self.current_image, roi_bbox, self.patch_size)
        grid_h, grid_w = meta["grid_shape"]
        num_patches = bolsa_x.shape[0]

        top = ctk.CTkToplevel(self)
        top.title(f"Inspector Co-MIL: Malla ({grid_h}x{grid_w}) - {num_patches} Parches")
        top.geometry("1100x850")
        
        # --- 1. Frame Superior (Controles de Zoom y Guardado) ---
        frame_controls = ctk.CTkFrame(top, fg_color="transparent")
        frame_controls.pack(side="top", fill="x", padx=20, pady=(10, 0))
        
        lbl_zoom = ctk.CTkLabel(frame_controls, text="Zoom de la Bolsa:", font=ctk.CTkFont(weight="bold"))
        lbl_zoom.pack(side="left", padx=(0, 10))

        # --- 2. Frame Inferior (Análisis Fijo) ---
        frame_analysis = ctk.CTkFrame(top, corner_radius=10, fg_color="#1a202c")
        frame_analysis.pack(side="bottom", fill="x", padx=20, pady=20)

        # --- 3. Frame Central (Canvas de la imagen) ---
        frame_grid = ctk.CTkFrame(top, corner_radius=0)
        frame_grid.pack(side="top", fill="both", expand=True, padx=20, pady=10)

        to_pil = transforms.ToPILImage()
        img_h_real, img_w_real = bolsa_x.shape[2], bolsa_x.shape[3]
        grid_img = Image.new('RGB', (grid_w * img_w_real, grid_h * img_h_real))

        for i in range(num_patches):
            row = i // grid_w
            col = i % grid_w
            patch_pil = to_pil(bolsa_x[i])
            grid_img.paste(patch_pil, (col * img_w_real, row * img_h_real))

        draw = ImageDraw.Draw(grid_img)
        line_color = (0, 255, 204)
        line_width = 4

        for row in range(1, grid_h):
            y = row * img_h_real
            draw.line([(0, y), (grid_img.width, y)], fill=line_color, width=line_width)

        for col in range(1, grid_w):
            x = col * img_w_real
            draw.line([(x, 0), (x, grid_img.height)], fill=line_color, width=line_width)
            
        for i in range(num_patches):
            row = i // grid_w
            col = i % grid_w
            x = col * img_w_real
            y = row * img_h_real
            draw.rectangle([x+5, y+5, x+55, y+30], fill=(0, 0, 0))
            draw.text((x + 10, y + 10), f"P-{i:02d}", fill=(255, 255, 255))

        cv_grid = tk.Canvas(frame_grid, bg="#1a1a1a", highlightthickness=0)
        sc_y = ctk.CTkScrollbar(frame_grid, orientation="vertical", command=cv_grid.yview)
        sc_y.pack(side="right", fill="y")
        sc_x = ctk.CTkScrollbar(frame_grid, orientation="horizontal", command=cv_grid.xview)
        sc_x.pack(side="bottom", fill="x")
        cv_grid.pack(side="left", fill="both", expand=True)
        cv_grid.configure(xscrollcommand=sc_x.set, yscrollcommand=sc_y.set)

        # Lógica de Zoom Adaptativo
        canvas_display_w, canvas_display_h = 1000, 450
        base_ratio = min(canvas_display_w / grid_img.width, canvas_display_h / grid_img.height)
        if base_ratio > 1.0: base_ratio = 1.0

        def update_zoom(val):
            scale = float(val)
            new_w = int(grid_img.width * base_ratio * scale)
            new_h = int(grid_img.height * base_ratio * scale)
            if new_w < 10 or new_h < 10: return
            
            img_res = grid_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img_res)
            cv_grid.delete("all")
            cv_grid.create_image(0, 0, anchor="nw", image=tk_img)
            cv_grid.image = tk_img 
            cv_grid.config(scrollregion=cv_grid.bbox("all"))

        slider_zoom = ctk.CTkSlider(frame_controls, from_=0.5, to=5.0, command=update_zoom)
        slider_zoom.set(1.0)
        slider_zoom.pack(side="left", fill="x", expand=True, padx=10)
        update_zoom(1.0) 

        # Función y botón para guardar la imagen del mallado
        base_name = self.get_current_image_base_name()
        
        def save_grid_image():
            roi_str = f"{saved_roi_idx + 1:02d}" if is_saved_roi else "TrazoActivo"
            default_name = f"{base_name}_Malla_ROI_{roi_str}.png"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=default_name,
                title="Guardar Malla de Parches",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
            )
            if filepath:
                grid_img.save(filepath)
                messagebox.showinfo("Guardado", f"Imagen guardada exitosamente en:\n{filepath}", parent=top)

        btn_save_img = ctk.CTkButton(frame_controls, text="💾 Guardar Malla como Imagen", fg_color="#2f9e44", hover_color="#2b8a3e", command=save_grid_image)
        btn_save_img.pack(side="right", padx=(10, 0))

        # --- Llenar Panel de Análisis ---
        lbl_panel_title = ctk.CTkLabel(frame_analysis, text="Análisis de Bolsa MIL & ROI Clínico", font=ctk.CTkFont(size=18, weight="bold"), text_color="#E2E8F0")
        lbl_panel_title.pack(anchor="w", pady=(15, 5), padx=20)

        ctk.CTkFrame(frame_analysis, height=2, fg_color="#374151").pack(fill="x", padx=20, pady=5)

        frame_cols = ctk.CTkFrame(frame_analysis, fg_color="transparent")
        frame_cols.pack(fill="x", padx=20, pady=5)
        
        col1 = ctk.CTkFrame(frame_cols, fg_color="transparent")
        col1.pack(side="left", fill="both", expand=True)
        
        col2 = ctk.CTkFrame(frame_cols, fg_color="transparent")
        col2.pack(side="left", fill="both", expand=True)

        def create_data_label(frame, text, val_text, color="#E2E8F0"):
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=text, font=ctk.CTkFont(weight="bold", size=13), text_color="gray").pack(side="left")
            ctk.CTkLabel(row, text=f" {val_text}", font=ctk.CTkFont(size=13), text_color=color).pack(side="left")

        annotator = self.entry_annotator.get().strip() or "Autor Desconocido"
        status_roi = f"ROI Guardada ({saved_roi_idx + 1:02d})" if is_saved_roi else "Trazo Activo (Al vuelo)"
        color_status = "#4da3ff" if is_saved_roi else "#00ffcc"
        
        orig_w, orig_h = self.current_image.size
        total_p_real_w = grid_w * self.patch_size
        total_p_real_h = grid_h * self.patch_size

        # Lógica para mostrar si hubo Reflection Padding
        pad_l, pad_r, pad_t, pad_b = meta["padding_applied"]
        if any([pad_l, pad_r, pad_t, pad_b]):
            pad_str = f"Sí aplicado (Izq:{pad_l} Der:{pad_r} Arr:{pad_t} Aba:{pad_b})"
            pad_color = "#ffb347" 
        else:
            pad_str = "No (Expansión pura)"
            pad_color = "#28a745"

        lbl_head1 = ctk.CTkLabel(col1, text="Datos Generales:", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_head1.pack(anchor="w", pady=(5, 5))
        
        create_data_label(col1, "Archivo Original:", f"{base_name}")
        create_data_label(col1, "Anotador Clínico:", f"{annotator}")
        create_data_label(col1, "Estado del Inspector:", f"{status_roi}", color_status)
        create_data_label(col1, "Resolución de Instancia:", f"{self.patch_size}x{self.patch_size} px")

        lbl_head2 = ctk.CTkLabel(col2, text="Análisis de Bolsa MIL (N):", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_head2.pack(anchor="w", pady=(5, 5))
        
        create_data_label(col2, "Nº Total de Instancias (Parches):", f"{num_patches}", color="#4da3ff")
        create_data_label(col2, "Estructura de Cuadrícula (Malla):", f"{grid_h}x{grid_w}")
        create_data_label(col2, "Dimens. Totales de Bolsa (Px Real):", f"{total_p_real_w}x{total_p_real_h}")
        create_data_label(col2, "Reflection Padding:", f"{pad_str}", pad_color)
        
        frame_labels = ctk.CTkFrame(frame_analysis, corner_radius=5, fg_color="#2D3748")
        frame_labels.pack(fill="x", padx=20, pady=(10, 15))
        
        lbl_tejidos = ctk.CTkLabel(frame_labels, text="Anotación Débil (Vector Y) - Tejidos Presentes en ROI:", font=ctk.CTkFont(size=13, weight="bold"))
        lbl_tejidos.pack(pady=(10, 0), padx=10, anchor="w")
        
        if current_labels:
            labels_text = ", ".join(current_labels)
            color_tejido = "#28a745"
        else:
            labels_text = "Negativo / Sin tejidos marcados"
            color_tejido = "gray"
            
        lbl_val_tejidos = ctk.CTkLabel(frame_labels, text=labels_text, font=ctk.CTkFont(size=14, slant="italic"), text_color=color_tejido, wraplength=900, justify="left")
        lbl_val_tejidos.pack(pady=(5, 10), padx=10, anchor="w")

        # ELIMINADO: top.grab_set() 
        # Esto permite que la ventana sea no modal (se puede minimizar o dejar en segundo plano)

    # =========================================================
    # NÚCLEO MATEMÁTICO MIL Y SERIALIZACIÓN TENSORIAL
    # =========================================================
    def smart_expansion_and_extraction(self, original_img: Image.Image, bbox: BBox, patch_size=224):
        """
        Transmuta el recorte macroscópico irregular en una Bolsa X de tensores estandarizados.
        Utiliza matemáticas para expandir el borde evitando píxeles negros que rompan
        los gradientes convolucionales. Si choca con el borde real, usa ReflectionPad2d.
        """
        img_w, img_h = original_img.size
        x1, y1, x2, y2 = bbox
        
        w_current, h_current = x2 - x1, y2 - y1
        diff_w = (math.ceil(w_current / patch_size) * patch_size) - w_current
        diff_h = (math.ceil(h_current / patch_size) * patch_size) - h_current

        expand_left, expand_top = diff_w // 2, diff_h // 2
        new_x1, new_y1 = x1 - expand_left, y1 - expand_top
        new_x2, new_y2 = x2 + (diff_w - expand_left), y2 + (diff_h - expand_top)

        pad_left = pad_top = pad_right = pad_bottom = 0
        if new_x1 < 0: pad_left, new_x1 = abs(new_x1), 0
        if new_y1 < 0: pad_top, new_y1 = abs(new_y1), 0
        if new_x2 > img_w: pad_right, new_x2 = new_x2 - img_w, img_w
        if new_y2 > img_h: pad_bottom, new_y2 = new_y2 - img_h, img_h

        cropped_img = original_img.crop((new_x1, new_y1, new_x2, new_y2))
        tensor_img = transforms.ToTensor()(cropped_img)

        # Resolución Estructural: El ReflectionPadding previene colapso de bordes geométricos falsos
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            pad_transform = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            tensor_img = pad_transform(tensor_img.unsqueeze(0)).squeeze(0)

        grid_h, grid_w = tensor_img.shape[1] // patch_size, tensor_img.shape[2] // patch_size
        patches = tensor_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(tensor_img.shape[0], -1, patch_size, patch_size).permute(1, 0, 2, 3)

        spatial_meta = {
            "grid_shape": (grid_h, grid_w),
            "patch_size": patch_size,
            "user_bbox": bbox,
            "bbox_expanded": (new_x1, new_y1, new_x2, new_y2),
            "padding_applied": (pad_left, pad_right, pad_top, pad_bottom),
        }
        return patches, spatial_meta

    def process_and_save(self):
        """Ejecuta el pipeline, inyecta la firma del autor y guarda en binario .pt"""
        annotator = self.entry_annotator.get().strip()
        if not annotator:
            messagebox.showerror(
                "Falta la firma",
                "Ingresa tu nombre en el campo de firma (arriba del panel lateral) antes de "
                "procesar la imagen.\n\nEsto es necesario para poder rastrear quién anotó cada "
                "bolsa generada.",
            )
            return

        if not messagebox.askyesno(
            "Confirmar anotador",
            f"¿Guardar esta imagen con la firma \"{annotator}\"?\n\n"
            "El campo de firma no se limpia solo al cambiar de imagen — confírmalo cada vez, "
            "sobre todo si más de una persona usa esta instalación.",
        ):
            self.entry_annotator.focus_set()
            return

        if self.bbox or self.get_selected_labels():
            if messagebox.askyesno("ROI sin guardar", "¿Tienes un trazo sin agregar. Deseas incluirlo antes de procesar la imagen?"):
                self.add_or_update_region()
            else:
                self.clear_current_region()

        if not self.pending_regions:
            messagebox.showwarning("Atención", "No hay ROIs guardadas. Añade al menos una zona para procesar.")
            return

        out_dir = self.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        base_name = self.get_current_image_base_name()
        total_rois = len(self.pending_regions)

        for region_idx, region in enumerate(self.pending_regions):
            bolsa_x, spatial_meta = self.smart_expansion_and_extraction(self.current_image, region["bbox"], self.patch_size)
            y_tensor = self.labels_to_vector(region["labels"])
            spatial_meta["roi_index"] = region_idx + 1
            spatial_meta["roi_total"] = total_rois
            spatial_meta["annotator"] = annotator 

            data_dict = {
                "X": bolsa_x,
                "Y": y_tensor,
                "class_names": list(self.class_catalog),
                "spatial_metadata": spatial_meta,
                "original_file": self.image_paths[self.current_idx],
                "roi_labels": list(region["labels"]),
            }

            fname = f"{base_name}_bag.pt" if total_rois == 1 else f"{base_name}__roi_{region_idx + 1:02d}_bag.pt"
            torch.save(data_dict, os.path.join(out_dir, fname))
            print(f"[{base_name}] ROI {region_idx + 1}/{total_rois} [{annotator}] -> Guardado ({bolsa_x.shape[0]} parches).")

        self.processed_files.add(base_name)
        self.clear_session_state(base_name)
        self.jump_to_next_unprocessed()


if __name__ == "__main__":
    app = EtiquetadorCoMIL()
    app.mainloop()