import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms

class EtiquetadorCoMIL:
    def __init__(self, root):
        self.root = root
        self.root.title("Preprocesador Co-MIL - Úlceras Pie Diabético")
        
        # Variables de estado
        self.image_paths = []
        self.current_idx = 0
        self.current_image = None
        self.tk_image = None
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.bbox = None # (x1, y1, x2, y2)
        self.scale_factor = 1.0 # Para mapear clics en UI a resolución original
        
        # Variables de etiquetas (MIML)
        self.var_granulacion = tk.IntVar()
        self.var_fibrina = tk.IntVar()
        self.var_callo = tk.IntVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Panel Superior: Controles
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Button(control_frame, text="Cargar Carpeta de Imágenes", command=self.load_folder).pack(side=tk.LEFT)
        self.lbl_info = tk.Label(control_frame, text="Imágenes: 0/0")
        self.lbl_info.pack(side=tk.LEFT, padx=20)
        
        # Panel de Etiquetas (Vector Y)
        label_frame = tk.LabelFrame(control_frame, text="Anotación Débil Global (Y)")
        label_frame.pack(side=tk.LEFT, padx=20)
        tk.Checkbutton(label_frame, text="Granulación", variable=self.var_granulacion).pack(side=tk.LEFT)
        tk.Checkbutton(label_frame, text="Fibrina", variable=self.var_fibrina).pack(side=tk.LEFT)
        tk.Checkbutton(label_frame, text="Callo", variable=self.var_callo).pack(side=tk.LEFT)
        
        tk.Button(control_frame, text="Procesar y Guardar Bolsa", command=self.process_and_save, bg="lightblue").pack(side=tk.RIGHT)
        tk.Button(control_frame, text="Siguiente Ignorando", command=self.next_image).pack(side=tk.RIGHT, padx=10)

        # Panel Central: Canvas de la imagen
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Eventos del ratón para Bounding Box
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        self.current_idx = 0
        
        if self.image_paths:
            self.load_image()
        else:
            messagebox.showwarning("Aviso", "No se encontraron imágenes válidas.")

    def load_image(self):
        if self.current_idx >= len(self.image_paths):
            messagebox.showinfo("Fin", "Procesamiento completado.")
            return
            
        path = self.image_paths[self.current_idx]
        self.current_image = Image.open(path).convert('RGB')
        
        # Reiniciar variables
        self.bbox = None
        self.var_granulacion.set(0)
        self.var_fibrina.set(0)
        self.var_callo.set(0)
        if self.rect: self.canvas.delete(self.rect)
        
        # Redimensionar para la UI manteniendo el aspect ratio
        canvas_w, canvas_h = 1000, 700
        img_w, img_h = self.current_image.size
        
        ratio = min(canvas_w/img_w, canvas_h/img_h)
        self.scale_factor = ratio
        
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        img_resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        self.lbl_info.config(text=f"Imágenes: {self.current_idx + 1}/{len(self.image_paths)} | Res Original: {img_w}x{img_h}")

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        # Mapear coordenadas de UI a la imagen original
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        
        orig_x1 = int(x1 / self.scale_factor)
        orig_y1 = int(y1 / self.scale_factor)
        orig_x2 = int(x2 / self.scale_factor)
        orig_y2 = int(y2 / self.scale_factor)
        
        self.bbox = (orig_x1, orig_y1, orig_x2, orig_y2)

    def extract_patches_with_padding(self, cropped_img, patch_size=224):
        transform = transforms.ToTensor()
        tensor_img = transform(cropped_img) # [C, H, W]
        C, H, W = tensor_img.shape
        
        # Calcular padding necesario (Zero-padding)
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        
        pad_transform = torch.nn.ZeroPad2d((0, pad_w, 0, pad_h))
        padded_img = pad_transform(tensor_img)
        
        _, new_H, new_W = padded_img.shape
        
        # Generar bolsa extrayendo parches de 224x224
        patches = padded_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(C, -1, patch_size, patch_size)
        patches = patches.permute(1, 0, 2, 3) # [N, C, H, W]
        return patches

    def process_and_save(self):
        if not self.bbox:
            messagebox.showwarning("Atención", "Dibuja un rectángulo para aislar la úlcera (ROI).")
            return
            
        vector_y = [self.var_granulacion.get(), self.var_fibrina.get(), self.var_callo.get()]
        if sum(vector_y) == 0:
            if not messagebox.askyesno("Atención", "No seleccionaste ningún tejido. ¿Seguro que es una bolsa negativa?"):
                return
                
        # 1. Recortar ROI
        cropped_img = self.current_image.crop(self.bbox)
        
        # 2. Extraer parches con Zero-Padding
        bolsa_X = self.extract_patches_with_padding(cropped_img, patch_size=224)
        
        # 3. Guardar en formato tensor de PyTorch
        y_tensor = torch.tensor(vector_y, dtype=torch.float32)
        
        data_dict = {
            'X': bolsa_X,
            'Y': y_tensor,
            'original_file': self.image_paths[self.current_idx]
        }
        
        # Crear carpeta de salida
        out_dir = os.path.join(os.path.dirname(self.image_paths[0]), "Bolsas_MIL_Procesadas")
        os.makedirs(out_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        out_path = os.path.join(out_dir, f"{base_name}_bag.pt")
        
        torch.save(data_dict, out_path)
        print(f"Bolsa guardada: {out_path} | Tamaño: {bolsa_X.shape[0]} parches de 224x224")
        
        self.next_image()

    def next_image(self):
        self.current_idx += 1
        self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = EtiquetadorCoMIL(root)
    root.geometry("1000x800")
    root.mainloop()