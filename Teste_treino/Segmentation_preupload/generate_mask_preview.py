import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import colors

#GERA AS MÁSCARAS DE PREVIEW PARA QA (ANTES DE ENVIAR PARA O CLUSTER)

# --- CONFIGURAÇÃO ---
BASE_DIR = r"D:\Tese_BDs\Defacer\ADNI_Structured"

def generate_qa_images():
    if not os.path.exists(BASE_DIR):
        print(f"Erro: Pasta não encontrada: {BASE_DIR}")
        return

    subjects = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    total = len(subjects)
    
    print(f"[QA GENERATOR] A gerar imagens para {total} exames...")
    print("Isto vai ser rápido...")

    count = 0

    for i, subj in enumerate(subjects):
        subj_path = os.path.join(BASE_DIR, subj)
        
        # Caminhos
        # Tenta encontrar o raw (pode ser raw.nii.gz ou outro nome se o conversor variou)
        raw_candidates = [f for f in os.listdir(subj_path) if f.endswith(".nii.gz") and "spot" not in f and "mask" not in f]
        if not raw_candidates: continue
        raw_path = os.path.join(subj_path, raw_candidates[0])

        mask_dir = os.path.join(subj_path, "spot_masks") # Confirma se a pasta se chama 'spot_masks' ou 'spot_final'
        if not os.path.exists(mask_dir): mask_dir = os.path.join(subj_path, "spot_final")
        
        mask_path = os.path.join(mask_dir, "spot_final_combined.nii.gz")
        qa_output = os.path.join(mask_dir, "QA_Check_Final.png")

        # Se não houver máscara, salta
        if not os.path.exists(mask_path):
            print(f"[{i+1}/{total}] SKIP: Máscara não encontrada para {subj}")
            continue
            
        # Se já existir imagem, salta (opcional)
        # if os.path.exists(qa_output): continue

        try:
            print(f"[{i+1}/{total}] A gerar imagem para: {subj}")
            
            # Carregar dados
            img_data = nib.load(raw_path).get_fdata()
            mask_data = nib.load(mask_path).get_fdata()

            # --- GERAÇÃO DA IMAGEM ---
            _create_qa_png(img_data, mask_data, qa_output, subj)
            count += 1

        except Exception as e:
            print(f" -> Erro: {e}")

    print("-" * 30)
    print(f"Concluído! {count} imagens geradas.")

def _create_qa_png(image, labels, save_path, file_name):
    # Mapa de cores: 
    # 0=Fundo, 1=Face(Cinza), 2=Orelhas(Roxo), 3=Boca(Azul), 4=Nariz(Amarelo), 5=Olhos(Vermelho)
    cmap = colors.ListedColormap(['none', 'gray', 'purple', 'blue', 'yellow', 'red'])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5] # Intervalos para as cores
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.suptitle(f"QA: {file_name}", fontsize=14)

    # Helper para encontrar o melhor slice (onde há mais máscara)
    def get_best_slice(label_id, axis):
        # axis: 0=Sagital, 1=Coronal, 2=Axial
        axes_sum = tuple(i for i in range(3) if i != axis)
        mask_counts = np.sum(labels == label_id, axis=axes_sum)
        if np.sum(mask_counts) == 0: 
            return int(image.shape[axis]/2)
        return np.argmax(mask_counts)

    # Órgãos para mostrar (Label ID, Nome)
    # Ajusta os IDs conforme o teu SpotSegmenter:
    # No último script: 1=Face, 2=Ears, 3=Mouth, 4=Nose, 5=Eyes
    organs = [
        (5, "Olhos (Red)"), 
        (4, "Nariz (Yel)"), 
        (2, "Orelhas (Pur)"), 
        (3, "Boca (Blu)")
    ]

    for col_idx, (lbl_id, name) in enumerate(organs):
        # Axial (Z)
        z = get_best_slice(lbl_id, 2)
        axes[0, col_idx].imshow(np.rot90(image[:, :, z]), cmap='gray')
        axes[0, col_idx].imshow(np.rot90(labels[:, :, z]), cmap=cmap, norm=norm, alpha=0.6)
        axes[0, col_idx].set_title(f"{name} - Axial")
        axes[0, col_idx].axis('off')

        # Sagital (X)
        x = get_best_slice(lbl_id, 0)
        axes[1, col_idx].imshow(np.rot90(image[x, :, :]), cmap='gray')
        axes[1, col_idx].imshow(np.rot90(labels[x, :, :]), cmap=cmap, norm=norm, alpha=0.6)
        axes[1, col_idx].set_title(f"Sagital")
        axes[1, col_idx].axis('off')

        # Coronal (Y)
        y = get_best_slice(lbl_id, 1)
        axes[2, col_idx].imshow(np.rot90(image[:, y, :]), cmap='gray')
        axes[2, col_idx].imshow(np.rot90(labels[:, y, :]), cmap=cmap, norm=norm, alpha=0.6)
        axes[2, col_idx].set_title(f"Coronal")
        axes[2, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100) # dpi=100 para ser mais rápido
    plt.close()

if __name__ == "__main__":
    generate_qa_images()