"""
=========================================================================================
HERRAMIENTA DE PREPROCESAMIENTO Y ETIQUETADO DEBIL PARA Co-MIL (MIML)
=========================================================================================

Flujo de trabajo:
1. Cargar una carpeta con imagenes clinicas.
2. Dibujar una o varias ROI sobre la imagen actual.
3. Marcar la anotacion debil correspondiente para cada ROI.
4. Guardar una sub-bolsa independiente por cada ROI detectada.

El catalogo de clases se mantiene persistente entre sesiones para evitar variaciones
accidentales en los nombres de tejidos.

Ejecucion:
    python CO-MIL/generador_bolsas.py
=========================================================================================
"""

import csv
import json
import math
import os
import sys
import tkinter as tk
import unicodedata
import zipfile
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

import customtkinter as ctk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms

try:
    from openpyxl import Workbook, load_workbook
except ImportError:
    Workbook = None
    load_workbook = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BBox = Tuple[int, int, int, int]


def get_app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def clean_label_text(label: str) -> str:
    return " ".join(str(label).strip().split())


def normalize_label_key(label: str) -> str:
    cleaned = clean_label_text(label)
    folded = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    return folded.casefold()


class EtiquetadorCoMIL(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Co-MIL Tool - Analisis Multi-Etiqueta UPD")
        self.geometry("1380x930")

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

        # ---------------------------------------------------------
        # CATALOGO DE CLASES
        # ---------------------------------------------------------
        self.patch_size = 224
        self.base_classes = [
            "Tejido Granulacion",
            "Fibrina",
            "Tejido Calloso",
        ]
        self.class_catalog = self.load_app_class_catalog()
        self.custom_classes = self.extract_custom_classes(self.class_catalog)
        self.label_vars: Dict[str, tk.IntVar] = {}
        self.checkboxes = []
        self._rebuild_label_vars()

        self.setup_ui()

    # =========================================================
    # PERSISTENCIA DE CLASES
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
            if not cleaned:
                continue
            key = normalize_label_key(cleaned)
            if key in seen:
                continue
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
        except Exception as exc:
            print(f"[Co-MIL] No se pudo leer el catalogo {file_path}: {exc}")
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
            "version": 2,
            "base_classes": self.base_classes,
            "custom_classes": self.custom_classes,
            "class_catalog": self.class_catalog,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_app_class_catalog(self) -> List[str]:
        class_catalog = self.load_catalog_from_file(self.get_registry_file_path())
        print(f"[Co-MIL] Catalogo activo: {class_catalog}")
        return class_catalog

    def save_app_class_catalog(self):
        try:
            self.save_catalog_to_file(self.get_registry_file_path())
        except Exception as exc:
            print(f"[Co-MIL] Error al guardar catalogo global: {exc}")

    def save_dataset_catalog(self):
        catalog_path = self.get_dataset_catalog_path()
        if not catalog_path:
            return

        try:
            os.makedirs(self.dataset_output_root, exist_ok=True)
            self.save_catalog_to_file(catalog_path)
        except Exception as exc:
            print(f"[Co-MIL] Error al guardar catalogo del dataset: {exc}")

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
        changed = new_catalog != self.class_catalog
        if not changed:
            return False

        self.class_catalog = new_catalog
        self.custom_classes = self.extract_custom_classes(self.class_catalog)
        self._rebuild_label_vars(previous_selection)
        if hasattr(self, "labels_scrollframe"):
            self.render_checkboxes()
            self.update_catalog_label()

        self.save_app_class_catalog()
        self.save_dataset_catalog()
        self.sync_pending_regions_to_catalog()
        self.sync_existing_bags_to_catalog(notify=False)
        return True

    def update_catalog_label(self):
        if not hasattr(self, "lbl_catalog_status"):
            return
        total = len(self.class_catalog)
        custom = len(self.custom_classes)
        self.lbl_catalog_status.configure(
            text=f"Catalogo activo: {total} clases ({custom} personalizadas)"
        )

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
        for var in self.label_vars.values():
            var.set(0)

    def sync_pending_regions_to_catalog(self):
        normalized_regions = []
        for region in self.pending_regions:
            normalized_labels = []
            for label in region["labels"]:
                existing = self.find_existing_label(label)
                normalized_labels.append(existing or clean_label_text(label))
            normalized_regions.append(
                {
                    "bbox": region["bbox"],
                    "labels": self.dedupe_labels(normalized_labels),
                }
            )
        self.pending_regions = normalized_regions
        if hasattr(self, "roi_listbox"):
            self.refresh_region_list()
            self.redraw_regions()

    def on_label_selection_change(self, *_args):
        if self.suspend_autosave:
            return
        self.save_session_state()

    def get_session_dir(self) -> Optional[str]:
        if not self.dataset_output_root:
            return None
        return os.path.join(self.dataset_output_root, self.session_dirname)

    def get_current_image_base_name(self) -> Optional[str]:
        if not self.image_paths or self.current_idx >= len(self.image_paths):
            return None
        return os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]

    def get_session_file_path(self, image_base_name: Optional[str] = None) -> Optional[str]:
        session_dir = self.get_session_dir()
        if session_dir is None:
            return None
        image_base_name = image_base_name or self.get_current_image_base_name()
        if not image_base_name:
            return None
        return os.path.join(session_dir, f"{image_base_name}.json")

    def serialize_bbox(self, bbox: Optional[BBox]):
        if bbox is None:
            return None
        return [int(value) for value in bbox]

    def deserialize_bbox(self, bbox_data) -> Optional[BBox]:
        if not bbox_data or len(bbox_data) != 4:
            return None
        return tuple(int(value) for value in bbox_data)

    def save_session_state(self):
        if self.suspend_autosave or not self.dataset_output_root or not self.image_paths:
            return

        session_path = self.get_session_file_path()
        if not session_path:
            return

        session_dir = os.path.dirname(session_path)
        os.makedirs(session_dir, exist_ok=True)

        payload = {
            "version": 1,
            "image_path": self.image_paths[self.current_idx],
            "image_base_name": self.get_current_image_base_name(),
            "pending_regions": [
                {
                    "bbox": self.serialize_bbox(region["bbox"]),
                    "labels": list(region["labels"]),
                }
                for region in self.pending_regions
            ],
            "current_bbox": self.serialize_bbox(self.bbox),
            "current_labels": self.get_selected_labels(),
            "selected_region_idx": self.selected_region_idx,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        try:
            with open(session_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[Co-MIL] No se pudo guardar la sesion temporal: {exc}")

    def clear_session_state(self, image_base_name: Optional[str] = None):
        session_path = self.get_session_file_path(image_base_name)
        if session_path and os.path.exists(session_path):
            try:
                os.remove(session_path)
            except OSError as exc:
                print(f"[Co-MIL] No se pudo borrar la sesion temporal {session_path}: {exc}")

    def restore_session_state(self):
        session_path = self.get_session_file_path()
        if not session_path or not os.path.exists(session_path):
            return False

        try:
            with open(session_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"[Co-MIL] No se pudo restaurar la sesion temporal: {exc}")
            return False

        pending_regions = []
        for region in data.get("pending_regions", []):
            bbox = self.deserialize_bbox(region.get("bbox"))
            if bbox is None:
                continue
            labels = []
            for label in region.get("labels", []):
                labels.append(self.find_existing_label(label) or clean_label_text(label))
            labels = self.dedupe_labels(labels)
            pending_regions.append({"bbox": bbox, "labels": labels})

        current_bbox = self.deserialize_bbox(data.get("current_bbox"))
        current_labels = []
        for label in data.get("current_labels", []):
            current_labels.append(self.find_existing_label(label) or clean_label_text(label))
        current_labels = self.dedupe_labels(current_labels)
        selected_region_idx = data.get("selected_region_idx")
        if not isinstance(selected_region_idx, int) or not (0 <= selected_region_idx < len(pending_regions)):
            selected_region_idx = None

        self.suspend_autosave = True
        try:
            self.pending_regions = pending_regions
            self.selected_region_idx = selected_region_idx
            self.bbox = current_bbox
            self.clear_label_selection()
            self.set_labels_from_list(current_labels)
            self.refresh_region_list()
            self.redraw_regions()
            self.update_info_label()
        finally:
            self.suspend_autosave = False

        print(f"[Co-MIL] Sesion temporal restaurada desde {session_path}")
        return True

    def get_manifest_csv_path(self) -> Optional[str]:
        if not self.dataset_output_root:
            return None
        return os.path.join(self.dataset_output_root, self.manifest_csv_filename)

    def get_manifest_xlsx_path(self) -> Optional[str]:
        if not self.dataset_output_root:
            return None
        return os.path.join(self.dataset_output_root, self.manifest_xlsx_filename)

    def get_manifest_headers(self) -> List[str]:
        return [
            "timestamp",
            "original_image",
            "base_name",
            "roi_index",
            "roi_total",
            "output_file",
            "output_path",
            "labels",
            "num_positive_labels",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "patch_size",
            "grid_h",
            "grid_w",
            "num_instances",
            "class_catalog",
        ]

    def read_manifest_rows(self) -> List[Dict[str, str]]:
        manifest_csv_path = self.get_manifest_csv_path()
        if not manifest_csv_path or not os.path.exists(manifest_csv_path):
            return []

        try:
            with open(manifest_csv_path, "r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception as exc:
            print(f"[Co-MIL] No se pudo leer el manifiesto CSV: {exc}")
            return []

    def write_manifest_rows(self, rows: List[Dict[str, object]]):
        manifest_csv_path = self.get_manifest_csv_path()
        if not manifest_csv_path:
            return

        os.makedirs(self.dataset_output_root, exist_ok=True)
        headers = self.get_manifest_headers()

        with open(manifest_csv_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({header: row.get(header, "") for header in headers})

        manifest_xlsx_path = self.get_manifest_xlsx_path()
        if Workbook is not None and load_workbook is not None:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "manifest"
            sheet.append(headers)
            for row in rows:
                sheet.append([row.get(header, "") for header in headers])
            workbook.save(manifest_xlsx_path)
        else:
            self.write_xlsx_fallback(manifest_xlsx_path, headers, rows)

    def write_xlsx_fallback(self, output_path: str, headers: List[str], rows: List[Dict[str, object]]):
        def col_ref(index: int) -> str:
            label = ""
            index += 1
            while index:
                index, rem = divmod(index - 1, 26)
                label = chr(65 + rem) + label
            return label

        def inline_cell(cell_ref: str, value) -> str:
            text = escape("" if value is None else str(value))
            return (
                f'<c r="{cell_ref}" t="inlineStr">'
                f"<is><t>{text}</t></is>"
                f"</c>"
            )

        sheet_rows = []
        all_rows = [headers] + [[row.get(header, "") for header in headers] for row in rows]
        for row_idx, row_values in enumerate(all_rows, start=1):
            cells = []
            for col_idx, value in enumerate(row_values):
                cells.append(inline_cell(f"{col_ref(col_idx)}{row_idx}", value))
            sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

        sheet_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheetData>{"".join(sheet_rows)}</sheetData>'
            "</worksheet>"
        )

        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="manifest" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>"
        )

        workbook_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            "</Relationships>"
        )

        root_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
            'Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
            'Target="docProps/app.xml"/>'
            "</Relationships>"
        )

        content_types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" '
            'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>"
        )

        styles_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            "</styleSheet>"
        )

        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        core_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:title>Co-MIL Manifest</dc:title>"
            "<dc:creator>Co-MIL</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>'
            "</cp:coreProperties>"
        )

        app_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>Co-MIL</Application>"
            "</Properties>"
        )

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types_xml)
            archive.writestr("_rels/.rels", root_rels_xml)
            archive.writestr("docProps/core.xml", core_xml)
            archive.writestr("docProps/app.xml", app_xml)
            archive.writestr("xl/workbook.xml", workbook_xml)
            archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
            archive.writestr("xl/styles.xml", styles_xml)
            archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    def update_manifest_for_image(self, image_base_name: str, new_rows: List[Dict[str, object]]):
        existing_rows = self.read_manifest_rows()
        filtered_rows = [row for row in existing_rows if row.get("base_name") != image_base_name]
        filtered_rows.extend(new_rows)
        self.write_manifest_rows(filtered_rows)

    # =========================================================
    # CARGA Y ALINEACION DE BOLSAS
    # =========================================================
    def infer_class_names_from_data(self, data_dict) -> List[str]:
        vector_y = data_dict.get("Y")
        class_names = data_dict.get("class_names")
        if vector_y is None:
            return []

        vector_length = int(torch.as_tensor(vector_y).numel())
        if isinstance(class_names, list) and len(class_names) == vector_length:
            return [clean_label_text(name) for name in class_names]
        if vector_length == len(self.base_classes):
            return list(self.base_classes)
        if vector_length == len(self.class_catalog):
            return list(self.class_catalog)
        raise ValueError(
            f"No se pudo inferir el esquema de clases para un vector Y de longitud {vector_length}."
        )

    def align_vector_to_current_catalog(self, vector_y, source_class_names: List[str]) -> torch.Tensor:
        source_tensor = torch.as_tensor(vector_y, dtype=torch.float32).flatten()
        aligned = torch.zeros(len(self.class_catalog), dtype=torch.float32)

        missing_labels = []
        for index, class_name in enumerate(source_class_names):
            if index >= source_tensor.numel():
                break
            canonical_name = self.find_existing_label(class_name)
            if canonical_name is None:
                missing_labels.append(class_name)
                continue
            target_idx = self.class_catalog.index(canonical_name)
            aligned[target_idx] = source_tensor[index]

        if missing_labels:
            raise ValueError(
                "El vector Y contiene clases fuera del catalogo actual: "
                + ", ".join(sorted(set(missing_labels)))
            )
        return aligned

    def iter_existing_bag_paths(self) -> List[str]:
        if not self.dataset_output_root or not os.path.exists(self.dataset_output_root):
            return []
        bag_paths = []
        for root, _, files in os.walk(self.dataset_output_root):
            for file_name in files:
                if file_name.endswith("_bag.pt"):
                    bag_paths.append(os.path.join(root, file_name))
        return sorted(bag_paths)

    def harvest_catalog_from_existing_bags(self) -> List[str]:
        discovered_labels = []
        for bag_path in self.iter_existing_bag_paths():
            try:
                data = torch.load(bag_path, map_location="cpu")
                discovered_labels.extend(self.infer_class_names_from_data(data))
            except Exception as exc:
                print(f"[Co-MIL] No se pudo inspeccionar {bag_path}: {exc}")
        return self.dedupe_labels(discovered_labels)

    def sync_existing_bags_to_catalog(self, notify: bool = False):
        bag_paths = self.iter_existing_bag_paths()
        if not bag_paths:
            return 0, 0

        updated = 0
        skipped = 0
        for bag_path in bag_paths:
            try:
                data = torch.load(bag_path, map_location="cpu")
                source_names = self.infer_class_names_from_data(data)
                aligned_vector = self.align_vector_to_current_catalog(data["Y"], source_names)
                current_vector = torch.as_tensor(data["Y"], dtype=torch.float32).flatten()
                current_names = data.get("class_names", [])
                requires_update = (
                    current_vector.numel() != aligned_vector.numel()
                    or not torch.equal(current_vector, aligned_vector)
                    or current_names != self.class_catalog
                )
                if requires_update:
                    data["Y"] = aligned_vector
                    data["class_names"] = list(self.class_catalog)
                    torch.save(data, bag_path)
                    updated += 1
                else:
                    skipped += 1
            except Exception as exc:
                print(f"[Co-MIL] No se pudo sincronizar {bag_path}: {exc}")

        if notify and updated:
            messagebox.showinfo(
                "Catalogo actualizado",
                f"Se sincronizaron {updated} bolsas con el catalogo actual.",
            )
        return updated, skipped

    def bootstrap_dataset_catalog(self):
        dataset_labels = []
        catalog_path = self.get_dataset_catalog_path()
        if catalog_path and os.path.exists(catalog_path):
            dataset_labels.extend(self.load_catalog_from_file(catalog_path))

        dataset_labels.extend(self.harvest_catalog_from_existing_bags())
        if dataset_labels:
            self.update_catalog(dataset_labels, preserve_selection=False)
        self.save_dataset_catalog()
        self.sync_existing_bags_to_catalog(notify=False)

    # =========================================================
    # UI
    # =========================================================
    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")

        self.canvas = tk.Canvas(self.main_frame, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.sidebar = ctk.CTkFrame(self, width=370, corner_radius=0)
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.grid_rowconfigure(11, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="Co-MIL Lab",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(24, 8))

        self.btn_load = ctk.CTkButton(
            self.sidebar,
            text="Cargar Carpeta Dataset",
            command=self.load_folder,
        )
        self.btn_load.grid(row=1, column=0, padx=20, pady=8, sticky="ew")

        self.lbl_info = ctk.CTkLabel(
            self.sidebar,
            text="Imagenes: 0/0\nPendientes: 0\nROI cargadas: 0",
            text_color="gray",
            justify="left",
        )
        self.lbl_info.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="w")

        self.lbl_patch = ctk.CTkLabel(
            self.sidebar,
            text="Resolucion de Instancia (px): 224 fijo",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_patch.grid(row=3, column=0, padx=20, pady=(8, 4), sticky="w")

        nota_resolucion = (
            "Puedes dibujar una o varias ROI por imagen.\n"
            "Cada ROI se guarda como una sub-bolsa independiente\n"
            "con su propia anotacion debil."
        )
        self.lbl_nota = ctk.CTkLabel(
            self.sidebar,
            text=nota_resolucion,
            font=ctk.CTkFont(size=11),
            text_color="#00d4aa",
            justify="left",
        )
        self.lbl_nota.grid(row=4, column=0, padx=20, pady=(0, 12), sticky="w")

        self.lbl_etiquetas = ctk.CTkLabel(
            self.sidebar,
            text="Anotacion debil de la ROI actual:",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.lbl_etiquetas.grid(row=5, column=0, padx=20, pady=(6, 4), sticky="w")

        self.lbl_catalog_status = ctk.CTkLabel(
            self.sidebar,
            text="",
            text_color="#9fb3c8",
            justify="left",
        )
        self.lbl_catalog_status.grid(row=6, column=0, padx=20, pady=(0, 6), sticky="w")

        self.labels_scrollframe = ctk.CTkScrollableFrame(self.sidebar, height=180)
        self.labels_scrollframe.grid(row=7, column=0, padx=20, pady=4, sticky="nsew")
        self.render_checkboxes()
        self.update_catalog_label()

        self.frame_add_class = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_add_class.grid(row=8, column=0, padx=20, pady=(6, 10), sticky="ew")

        self.entry_new_class = ctk.CTkEntry(
            self.frame_add_class,
            placeholder_text="Nuevo tejido...",
            width=210,
        )
        self.entry_new_class.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_add_class = ctk.CTkButton(
            self.frame_add_class,
            text="+",
            width=40,
            command=self.add_custom_class,
        )
        self.btn_add_class.pack(side=tk.LEFT)

        self.lbl_regions = ctk.CTkLabel(
            self.sidebar,
            text="ROI pendientes en esta imagen:",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.lbl_regions.grid(row=9, column=0, padx=20, pady=(6, 4), sticky="w")

        self.roi_panel = ctk.CTkFrame(self.sidebar)
        self.roi_panel.grid(row=10, column=0, padx=20, pady=4, sticky="nsew")
        self.roi_panel.grid_columnconfigure(0, weight=1)
        self.roi_panel.grid_rowconfigure(0, weight=1)

        self.roi_listbox = tk.Listbox(
            self.roi_panel,
            height=7,
            bg="#202833",
            fg="#e5eef8",
            selectbackground="#0d6efd",
            selectforeground="white",
            activestyle="none",
            relief="flat",
            exportselection=False,
        )
        self.roi_listbox.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 8), sticky="nsew")
        self.roi_listbox.bind("<<ListboxSelect>>", self.on_region_select)

        self.btn_add_roi = ctk.CTkButton(
            self.roi_panel,
            text="Agregar ROI actual",
            command=self.add_or_update_region,
        )
        self.btn_add_roi.grid(row=1, column=0, padx=(10, 5), pady=(0, 10), sticky="ew")

        self.btn_remove_roi = ctk.CTkButton(
            self.roi_panel,
            text="Eliminar ROI",
            command=self.remove_selected_region,
            fg_color="#7b1f2b",
            hover_color="#982436",
        )
        self.btn_remove_roi.grid(row=1, column=1, padx=(5, 10), pady=(0, 10), sticky="ew")

        self.frame_actions = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_actions.grid(row=12, column=0, padx=20, pady=(8, 24), sticky="ew")
        self.frame_actions.grid_columnconfigure((0, 1), weight=1)

        self.btn_clear_current = ctk.CTkButton(
            self.frame_actions,
            text="Limpiar ROI actual",
            command=self.clear_current_region,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
        )
        self.btn_clear_current.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="ew")

        self.btn_next = ctk.CTkButton(
            self.frame_actions,
            text="Saltar a Pendiente",
            command=self.jump_to_next_unprocessed,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
        )
        self.btn_next.grid(row=0, column=1, padx=(5, 0), pady=0, sticky="ew")

        self.btn_save = ctk.CTkButton(
            self.sidebar,
            text="Guardar ROI(s) y pasar a la siguiente imagen",
            command=self.process_and_save,
            fg_color="#28a745",
            hover_color="#218838",
        )
        self.btn_save.grid(row=13, column=0, padx=20, pady=(0, 30), sticky="ew")

    def render_checkboxes(self):
        for checkbox in self.checkboxes:
            checkbox.destroy()
        self.checkboxes.clear()

        for idx, (label_name, var) in enumerate(self.label_vars.items()):
            checkbox = ctk.CTkCheckBox(self.labels_scrollframe, text=label_name, variable=var)
            checkbox.grid(row=idx, column=0, padx=10, pady=8, sticky="w")
            self.checkboxes.append(checkbox)

    def refresh_region_list(self):
        self.roi_listbox.delete(0, tk.END)
        for idx, region in enumerate(self.pending_regions, start=1):
            labels = region["labels"]
            preview = ", ".join(labels[:3]) if labels else "Caso negativo"
            if len(labels) > 3:
                preview += ", ..."
            x1, y1, x2, y2 = region["bbox"]
            item = f"ROI {idx:02d} | {preview} | ({x1},{y1})-({x2},{y2})"
            self.roi_listbox.insert(tk.END, item)

        if self.selected_region_idx is not None and self.selected_region_idx < len(self.pending_regions):
            self.roi_listbox.selection_set(self.selected_region_idx)

    # =========================================================
    # ANOTACION DE ROI
    # =========================================================
    def add_custom_class(self):
        new_class = clean_label_text(self.entry_new_class.get())
        if not new_class:
            messagebox.showwarning("Atencion", "Escribe un nombre de tejido antes de agregarlo.")
            return

        existing_name = self.find_existing_label(new_class)
        if existing_name is not None:
            self.entry_new_class.delete(0, tk.END)
            self.label_vars[existing_name].set(1)
            messagebox.showinfo(
                "Informacion",
                f"La clase '{existing_name}' ya existe en el catalogo activo.",
            )
            return

        changed = self.update_catalog(
            self.class_catalog + [new_class],
            preserve_selection=True,
            preselected={new_class: 1},
        )
        self.entry_new_class.delete(0, tk.END)
        if changed:
            messagebox.showinfo(
                "Catalogo actualizado",
                f"La clase '{new_class}' se agrego y quedara disponible en las siguientes imagenes.",
            )

    def build_region_payload(self) -> Optional[Dict[str, object]]:
        if not self.bbox or (self.bbox[2] - self.bbox[0] < 10) or (self.bbox[3] - self.bbox[1] < 10):
            messagebox.showwarning("Atencion", "Traza una caja delimitadora valida sobre la ulcera.")
            return None

        selected_labels = self.get_selected_labels()
        if not selected_labels:
            if not messagebox.askyesno(
                "Confirmacion",
                "No marcaste tejidos para esta ROI. Deseas guardarla como caso negativo?",
            ):
                return None

        return {
            "bbox": self.bbox,
            "labels": list(selected_labels),
        }

    def add_or_update_region(self):
        region = self.build_region_payload()
        if region is None:
            return

        if self.selected_region_idx is None:
            self.pending_regions.append(region)
        else:
            self.pending_regions[self.selected_region_idx] = region

        self.selected_region_idx = None
        self.clear_current_region(reset_selection=False)
        self.refresh_region_list()
        self.redraw_regions()
        self.update_info_label()
        self.save_session_state()

    def remove_selected_region(self):
        if self.selected_region_idx is None or self.selected_region_idx >= len(self.pending_regions):
            messagebox.showinfo("Informacion", "Selecciona una ROI de la lista para eliminarla.")
            return

        del self.pending_regions[self.selected_region_idx]
        self.selected_region_idx = None
        self.clear_current_region(reset_selection=False)
        self.refresh_region_list()
        self.redraw_regions()
        self.update_info_label()
        self.save_session_state()

    def clear_current_region(self, reset_selection: bool = True):
        self.bbox = None
        self.clear_label_selection()
        if reset_selection:
            self.selected_region_idx = None
            self.roi_listbox.selection_clear(0, tk.END)
        self.redraw_regions()
        self.save_session_state()

    def on_region_select(self, _event=None):
        selection = self.roi_listbox.curselection()
        if not selection:
            return

        idx = int(selection[0])
        if idx >= len(self.pending_regions):
            return

        region = self.pending_regions[idx]
        self.selected_region_idx = idx
        self.bbox = region["bbox"]
        self.set_labels_from_list(region["labels"])
        self.redraw_regions()
        self.save_session_state()

    # =========================================================
    # FLUJO DE DATOS
    # =========================================================
    def get_output_dir(self) -> str:
        if not self.dataset_output_root:
            raise RuntimeError("Primero carga una carpeta de imagenes.")
        return os.path.join(self.dataset_output_root, f"{self.patch_size}px")

    def base_name_from_bag_file(self, file_name: str) -> Optional[str]:
        if not file_name.endswith("_bag.pt"):
            return None
        raw_name = file_name[:-7]
        if raw_name.endswith("__roi"):
            return raw_name[:-5]
        if "__roi_" in raw_name:
            return raw_name.split("__roi_", 1)[0]
        return raw_name

    def scan_processed_files(self):
        self.processed_files.clear()
        out_dir = self.get_output_dir()
        if not os.path.exists(out_dir):
            return

        for file_name in os.listdir(out_dir):
            if not file_name.endswith("_bag.pt"):
                continue
            base_name = self.base_name_from_bag_file(file_name)
            if base_name:
                self.processed_files.add(base_name)

    def jump_to_next_unprocessed(self):
        for idx, path in enumerate(self.image_paths):
            base_name = os.path.splitext(os.path.basename(path))[0]
            if base_name not in self.processed_files:
                self.current_idx = idx
                self.load_image()
                return
        messagebox.showinfo("Completado", "Todas las imagenes para esta resolucion han sido procesadas.")

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        image_paths = [
            os.path.join(folder_path, file_name)
            for file_name in os.listdir(folder_path)
            if file_name.lower().endswith(valid_exts)
        ]
        if not image_paths:
            messagebox.showwarning("Aviso", "No se encontraron imagenes validas.")
            return

        self.dataset_input_dir = folder_path
        self.dataset_output_root = os.path.join(folder_path, "Bolsas_MIL_Procesadas")
        self.image_paths = sorted(image_paths, key=lambda path: os.path.basename(path).lower())
        self.bootstrap_dataset_catalog()
        self.scan_processed_files()
        self.jump_to_next_unprocessed()

    def update_info_label(self):
        pendientes = len(self.image_paths) - len(self.processed_files)
        self.lbl_info.configure(
            text=(
                f"Imagenes: {self.current_idx + 1 if self.image_paths else 0}/{len(self.image_paths)}\n"
                f"Pendientes: {pendientes}\n"
                f"ROI cargadas: {len(self.pending_regions)}"
            )
        )

    def load_image(self):
        if self.current_idx >= len(self.image_paths):
            return

        path = self.image_paths[self.current_idx]
        self.current_image = Image.open(path).convert("RGB")
        self.suspend_autosave = True
        self.pending_regions = []
        self.selected_region_idx = None
        self.bbox = None
        self.clear_label_selection()
        self.refresh_region_list()

        self.update_idletasks()
        canvas_w = self.main_frame.winfo_width() - 20
        canvas_h = self.main_frame.winfo_height() - 20
        if canvas_w < 100:
            canvas_w, canvas_h = 850, 650

        img_w, img_h = self.current_image.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        self.scale_factor = ratio

        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        img_resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.rendered_image_size = (new_w, new_h)
        self.canvas.delete("all")
        self.canvas.config(width=new_w, height=new_h)

        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image, tags="img")
        self.img_offset_x = x_offset
        self.img_offset_y = y_offset

        self.redraw_regions()
        self.update_info_label()
        restored = self.restore_session_state()
        self.suspend_autosave = False
        if not restored:
            self.save_session_state()

    # =========================================================
    # CANVAS Y ROI
    # =========================================================
    def image_to_canvas_bbox(self, bbox: BBox):
        x1, y1, x2, y2 = bbox
        return (
            self.img_offset_x + int(x1 * self.scale_factor),
            self.img_offset_y + int(y1 * self.scale_factor),
            self.img_offset_x + int(x2 * self.scale_factor),
            self.img_offset_y + int(y2 * self.scale_factor),
        )

    def redraw_regions(self):
        if not hasattr(self, "canvas"):
            return

        for canvas_id in self.region_canvas_ids:
            self.canvas.delete(canvas_id)
        self.region_canvas_ids = []

        if self.draft_rect_id:
            self.canvas.delete(self.draft_rect_id)
            self.draft_rect_id = None

        for idx, region in enumerate(self.pending_regions):
            x1, y1, x2, y2 = self.image_to_canvas_bbox(region["bbox"])
            is_selected = idx == self.selected_region_idx
            outline = "#4da3ff" if is_selected else "#ffb347"
            width = 3 if is_selected else 2
            rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, width=width)
            text_id = self.canvas.create_text(
                x1 + 8,
                y1 + 8,
                anchor=tk.NW,
                text=f"ROI {idx + 1}",
                fill=outline,
                font=("Segoe UI", 10, "bold"),
            )
            self.region_canvas_ids.extend([rect_id, text_id])

        if self.bbox is not None:
            x1, y1, x2, y2 = self.image_to_canvas_bbox(self.bbox)
            self.draft_rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="#00ffcc", width=3, dash=(6, 4)
            )

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.draft_rect_id:
            self.canvas.delete(self.draft_rect_id)
        self.draft_rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="#00ffcc",
            width=3,
            dash=(6, 4),
        )

    def on_drag(self, event):
        if self.draft_rect_id:
            self.canvas.coords(self.draft_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)

        x1, x2 = x1 - self.img_offset_x, x2 - self.img_offset_x
        y1, y2 = y1 - self.img_offset_y, y2 - self.img_offset_y
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.rendered_image_size[0], x2)
        y2 = min(self.rendered_image_size[1], y2)

        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)

        self.bbox = (orig_x1, orig_y1, orig_x2, orig_y2)
        self.redraw_regions()
        self.save_session_state()

    # =========================================================
    # EXTRACCION MIL
    # =========================================================
    def smart_expansion_and_extraction(self, original_img: Image.Image, bbox: BBox, patch_size=224):
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
        if new_x1 < 0:
            pad_left, new_x1 = abs(new_x1), 0
        if new_y1 < 0:
            pad_top, new_y1 = abs(new_y1), 0
        if new_x2 > img_w:
            pad_right, new_x2 = new_x2 - img_w, img_w
        if new_y2 > img_h:
            pad_bottom, new_y2 = new_y2 - img_h, img_h

        cropped_img = original_img.crop((new_x1, new_y1, new_x2, new_y2))
        tensor_img = transforms.ToTensor()(cropped_img)
        channels, height, width = tensor_img.shape

        if any([pad_left, pad_right, pad_top, pad_bottom]):
            pad_transform = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            tensor_img = pad_transform(tensor_img.unsqueeze(0)).squeeze(0)
            height, width = tensor_img.shape[1], tensor_img.shape[2]

        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = tensor_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(channels, -1, patch_size, patch_size)
        patches = patches.permute(1, 0, 2, 3)

        spatial_metadata = {
            "grid_shape": (grid_h, grid_w),
            "patch_size": patch_size,
            "user_bbox": bbox,
            "bbox_expanded": (new_x1, new_y1, new_x2, new_y2),
            "padding_applied": (pad_left, pad_right, pad_top, pad_bottom),
        }
        return patches, spatial_metadata

    # =========================================================
    # GUARDADO
    # =========================================================
    def ensure_current_region_is_queued(self):
        if self.bbox is None:
            return True

        has_selection = bool(self.get_selected_labels()) or self.selected_region_idx is not None
        if not has_selection and self.bbox is None:
            return True

        if not messagebox.askyesno(
            "ROI sin agregar",
            "La ROI actual aun no esta en la lista. Deseas agregarla antes de guardar?",
        ):
            return True

        region = self.build_region_payload()
        if region is None:
            return False

        if self.selected_region_idx is None:
            self.pending_regions.append(region)
        else:
            self.pending_regions[self.selected_region_idx] = region
            self.selected_region_idx = None

        self.clear_current_region(reset_selection=False)
        self.refresh_region_list()
        self.redraw_regions()
        return True

    def build_output_filename(self, base_name: str, region_idx: int, total_regions: int) -> str:
        if total_regions == 1:
            return f"{base_name}_bag.pt"
        return f"{base_name}__roi_{region_idx + 1:02d}_bag.pt"

    def build_manifest_row(
        self,
        base_name: str,
        file_name: str,
        out_path: str,
        region_idx: int,
        total_regions: int,
        region: Dict[str, object],
        spatial_meta: Dict[str, object],
        bolsa_x: torch.Tensor,
    ) -> Dict[str, object]:
        x1, y1, x2, y2 = region["bbox"]
        labels = list(region["labels"])
        grid_h, grid_w = spatial_meta["grid_shape"]
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "original_image": self.image_paths[self.current_idx],
            "base_name": base_name,
            "roi_index": region_idx + 1,
            "roi_total": total_regions,
            "output_file": file_name,
            "output_path": out_path,
            "labels": ", ".join(labels),
            "num_positive_labels": len(labels),
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "patch_size": self.patch_size,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "num_instances": int(bolsa_x.shape[0]),
            "class_catalog": ", ".join(self.class_catalog),
        }

    def process_and_save(self):
        if not self.ensure_current_region_is_queued():
            return

        if not self.pending_regions:
            messagebox.showwarning("Atencion", "Agrega al menos una ROI antes de guardar.")
            return

        if self.current_image is None:
            messagebox.showwarning("Atencion", "No hay imagen cargada para procesar.")
            return

        out_dir = self.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        self.save_dataset_catalog()

        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        total_regions = len(self.pending_regions)
        manifest_rows = []

        for region_idx, region in enumerate(self.pending_regions):
            bolsa_x, spatial_meta = self.smart_expansion_and_extraction(
                self.current_image,
                region["bbox"],
                patch_size=self.patch_size,
            )
            y_tensor = self.labels_to_vector(region["labels"])
            spatial_meta["roi_index"] = region_idx + 1
            spatial_meta["roi_total"] = total_regions

            data_dict = {
                "X": bolsa_x,
                "Y": y_tensor,
                "class_names": list(self.class_catalog),
                "spatial_metadata": spatial_meta,
                "original_file": self.image_paths[self.current_idx],
                "roi_labels": list(region["labels"]),
            }

            file_name = self.build_output_filename(base_name, region_idx, total_regions)
            out_path = os.path.join(out_dir, file_name)
            torch.save(data_dict, out_path)
            manifest_rows.append(
                self.build_manifest_row(
                    base_name,
                    file_name,
                    out_path,
                    region_idx,
                    total_regions,
                    region,
                    spatial_meta,
                    bolsa_x,
                )
            )
            print(
                f"[{base_name}] ROI {region_idx + 1}/{total_regions} -> "
                f"Bolsa ({self.patch_size}px) Grid: {spatial_meta['grid_shape']}"
            )

        self.update_manifest_for_image(base_name, manifest_rows)
        self.processed_files.add(base_name)
        self.clear_session_state(base_name)
        self.pending_regions = []
        self.selected_region_idx = None
        self.bbox = None
        self.refresh_region_list()
        self.redraw_regions()
        self.jump_to_next_unprocessed()


if __name__ == "__main__":
    app = EtiquetadorCoMIL()
    app.mainloop()
