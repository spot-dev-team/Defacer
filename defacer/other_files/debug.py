import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scipy import ndimage

# --- CONFIGURAÇÃO ---
ADNI_DIR = "/home/andresousa615/defacer/ADNI_Lite_Cluster"
# Define aqui o tamanho que estás a usar no treino
DIM = (128, 128, 128)
N_CLASSES = 5

# --- FUNÇÕES DE AJUDA (Copiadas do teu código) ---
def robust_resize(volume, dim, order=1):
    if len(volume.shape) == 4: volume = volume[:, :, :, 0]
    # Normalização Min-Max para resize limpo
    min_v, max_v = np.min(volume), np.max(volume)
    volume = (volume - min_v) / (max_v - min_v + 1e-8)
    
    factors = (dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2])
    vol_res = ndimage.zoom(volume, factors, order=order)
    return vol_res

def remap_labels(mask):
    # Lógica do teu script
    # Assumindo: 0=Fundo, 1=Face(Resto), 2=Orelhas, 3=Boca, 4=Nariz, 5=Olhos
    # Mapear para: 0=Fundo, 1=Orelhas, 2=Boca, 3=Nariz, 4=Olhos
    
    final_mask = np.zeros_like(mask)
    
    # ATENÇÃO: Aqui vou imprimir o que entra para percebermos o erro
    unique_input = np.unique(mask)
    print(f"   DEBUG INTERNO: Valores na máscara original (antes remap): {unique_input}")
    
    final_mask[mask == 2] = 1 # Orelhas
    final_mask[mask == 3] = 2 # Boca
    final_mask[mask == 4] = 3 # Nariz
    final_mask[mask == 5] = 4 # Olhos
    
    return final_mask

# --- O TEU DATA GENERATOR (Simplificado para debug) ---
class DebugDataGenerator:
    def __init__(self, list_ids, base_dir):
        self.list_ids = list_ids
        self.base_dir = base_dir

    def get_sample(self, index):
        ID = self.list_ids[index]
        print(f"\n--- A processar Paciente: {ID} ---")
        subj_path = os.path.join(self.base_dir, ID)
        
        # 1. Load Raw
        raw_path = os.path.join(subj_path, "raw.nii.gz")
        if not os.path.exists(raw_path):
             cand = [f for f in os.listdir(subj_path) if f.endswith(".nii.gz") and "mask" not in f][0]
             raw_path = os.path.join(subj_path, cand)
        
        print(f"   Lendo imagem de: {raw_path}")
        img = nib.load(raw_path).get_fdata()
        
        # 2. Load Mask
        mask_path = os.path.join(subj_path, "training_masks", "mask_4_classes.nii.gz")
        print(f"   Lendo máscara de: {mask_path}")
        msk = nib.load(mask_path).get_fdata()

        # 3. Resize
        print("   A aplicar Resize...")
        img = robust_resize(img, DIM, order=1)
        # IMPORTANTE: Máscaras usam order=0 (vizinho mais próximo) para não criar valores decimais
        msk = robust_resize(msk, DIM, order=0)
        
        # 4. Remap
        print("   A aplicar Remap Labels...")
        msk = np.round(msk)
        msk = remap_labels(msk)

        # 5. Normalize Intensity
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        # 6. One-Hot (Simular o output final)
        msk_onehot = to_categorical(msk, num_classes=N_CLASSES)

        return img, msk, msk_onehot

# --- EXECUÇÃO ---
# 1. Encontrar um paciente real
all_adni = [f for f in os.listdir(ADNI_DIR) if os.path.isdir(os.path.join(ADNI_DIR, f))]
valid_adni = [f for f in all_adni if os.path.exists(os.path.join(ADNI_DIR, f, "training_masks", "mask_4_classes.nii.gz"))]

if not valid_adni:
    print("ERRO: Não encontrei nenhum paciente válido na pasta!")
    exit()

# Vamos testar com o primeiro da lista
test_id = valid_adni[0] 

# 2. Gerar dados
gen = DebugDataGenerator([test_id], ADNI_DIR)
img_final, msk_final, msk_onehot = gen.get_sample(0)

# 3. ANÁLISE CRÍTICA
print("\n" + "="*30)
print("RELATÓRIO DE DEBUG")
print("="*30)

unique_labels = np.unique(msk_final)
print(f"Valores únicos na máscara FINAL: {unique_labels}")

# Contagem de pixéis por classe
for val in unique_labels:
    count = np.sum(msk_final == val)
    print(f" -> Classe {val}: {count} pixéis")

print("-" * 30)

if len(unique_labels) == 1 and unique_labels[0] == 0:
    print("🚨 ERRO GRAVE DETETADO: A máscara final está VAZIA (só fundo)!")
    print("   Causa Provável: O 'remap_labels' eliminou tudo ou o resize estragou os valores.")
    print("   Ação: PARA O TREINO IMEDIATAMENTE.")
elif len(unique_labels) > 1:
    print(f"✅ A máscara parece válida. Tem {len(unique_labels)} classes.")
    if 4 in unique_labels:
        print("   (Inclui olhos - Classe 4)")
    else:
        print("   ⚠️ AVISO: Não encontrei a classe 4 (Olhos) neste exemplo.")
else:
    print("⚠️ Resultado estranho, verificar ficheiros.")

# 4. SALVAR PARA VISUALIZAR
print("\nSalvando ficheiros para inspeção visual...")
nib.save(nib.Nifti1Image(img_final, np.eye(4)), 'debug_img_128.nii.gz')
nib.save(nib.Nifti1Image(msk_final.astype(np.float32), np.eye(4)), 'debug_msk_128.nii.gz')
print("Ficheiros salvos: 'debug_img_128.nii.gz' e 'debug_msk_128.nii.gz'")
