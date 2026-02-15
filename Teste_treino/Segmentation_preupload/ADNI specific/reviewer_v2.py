import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- CONFIGURAÇÃO ---
BASE_DIR = r"D:\Tese_BDs\Defacer\ADNI_Structured"
REJECT_DIR = os.path.join(BASE_DIR, "_REJECTED")

class SpotReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Spot Reviewer v2 - Validação Segmentação Limpa")
        self.root.geometry("1000x900")
        
        if not os.path.exists(REJECT_DIR):
            os.makedirs(REJECT_DIR)

        self.qa_files = []
        self.current_index = 0
        
        self.lbl_info = tk.Label(root, text="A carregar...", font=("Arial", 14, "bold"))
        self.lbl_info.pack(pady=10)
        
        self.lbl_image = tk.Label(root)
        self.lbl_image.pack(expand=True)
        
        self.lbl_instructions = tk.Label(root, text="[<- REJEITAR]   |   [MANTER ->]", font=("Arial", 12))
        self.lbl_instructions.pack(pady=10, side=tk.BOTTOM)

        self.root.bind("<Left>", self.reject_exam)
        self.root.bind("<Right>", self.keep_exam)
        
        self.load_dataset()
        self.show_current()

    def load_dataset(self):
        print("A procurar imagens QA...")
        for root_dir, dirs, files in os.walk(BASE_DIR):
            if "_REJECTED" in root_dir: continue
                
            for file in files:
                # --- MUDANÇA PRINCIPAL: Procura o novo nome ---
                if file == "QA_Validation.png":
                    full_path = os.path.join(root_dir, file)
                    # O ficheiro está em: .../Exame/training_masks/QA_Validation.png
                    # Parent = training_masks, Grandparent = Exame
                    exam_folder = os.path.dirname(os.path.dirname(full_path))
                    
                    self.qa_files.append({
                        "img_path": full_path,
                        "exam_folder": exam_folder,
                        "name": os.path.basename(exam_folder)
                    })
        
        print(f"Encontrados {len(self.qa_files)} exames para rever.")

    def show_current(self):
        if self.current_index >= len(self.qa_files):
            messagebox.showinfo("Fim", "Revisão concluída!")
            self.root.quit()
            return

        data = self.qa_files[self.current_index]
        self.lbl_info.config(text=f"[{self.current_index + 1}/{len(self.qa_files)}] {data['name']}")
        
        try:
            img = Image.open(data["img_path"])
            img.thumbnail((900, 800), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.lbl_image.config(image=self.tk_img)
        except Exception as e:
            print(f"Erro imagem: {e}")
            self.current_index += 1
            self.show_current()

    def keep_exam(self, event=None):
        self.current_index += 1
        self.show_current()

    def reject_exam(self, event=None):
        data = self.qa_files[self.current_index]
        target = os.path.join(REJECT_DIR, data["name"])
        try:
            shutil.move(data["exam_folder"], target)
            print(f"REJEITADO: {data['name']}")
        except Exception as e:
            print(f"Erro ao mover: {e}")
        self.current_index += 1
        self.show_current()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpotReviewer(root)
    root.mainloop()