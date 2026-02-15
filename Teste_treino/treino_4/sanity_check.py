import numpy as np
import nibabel as nib
import os
import tensorflow as tf
import random

# IMPORTANTE: Importar a classe do TEU script corrigido
# Certifica-te que o ficheiro do treino se chama 'train_replication_paper.py'
from train_adni_keras import DataGenerator

# --- CONFIGURAÇÃO ---
ADNI_DIR = r"C:\Tese\ADNI_Lite_Cluster"

def run_sanity_check():
    print("--- A INICIAR SANITY CHECK ---")
    
    # 1. Encontrar um paciente real
    all_patients = [f for f in os.listdir(ADNI_DIR) if os.path.isdir(os.path.join(ADNI_DIR, f))]
    valid_patients = [f for f in all_patients if os.path.exists(os.path.join(ADNI_DIR, f, "training_masks", "mask_4_classes.nii.gz"))]
    
    if not valid_patients:
        print("ERRO: Nenhum paciente encontrado.")
        return

    id= random.randint(0, len(valid_patients)-1)
    #018_S_0335__2007-05-22_09_18_480__I54979

    print(f"id: {id}")
    test_id = valid_patients[81]
    print(f"Paciente Teste: {test_id}")

    # 2. Instanciar o Gerador (Tal e qual como no treino)
    # Usamos augmentation=True para ver se as rotações não estragam a máscara
    gen = DataGenerator([test_id], ADNI_DIR, batch_size=1, augment=True, augmentation_factor=1)

    # 3. Pedir um Batch (Isto ativa o __getitem__, robust_resize, remap, etc.)
    X, y = gen.__getitem__(0)

    # X shape: (1, 128, 128, 128, 1)
    # y shape: (1, 128, 128, 128, 5) -> One-Hot Encoded

    # 4. ANÁLISE DOS DADOS
    print("\n--- RELATÓRIO DE DADOS ---")
    print(f"Imagem Shape: {X.shape}")
    print(f"Imagem Min/Max: {X.min():.4f} / {X.max():.4f} (Deve estar entre 0 e 1)")
    
    # Reverter One-Hot para ver as classes (0, 1, 2, 3, 4)
    y_argmax = np.argmax(y[0], axis=-1)
    unique_classes = np.unique(y_argmax)
    
    print(f"Classes encontradas na máscara final: {unique_classes}")
    
    # 5. VEREDITO AUTOMÁTICO
    if len(unique_classes) < 3:
        print("\n🚨🚨 VEREDITO: FALHOU! 🚨🚨")
        print("A máscara só tem fundo ou poucas classes. O erro de normalização persiste ou o remap falhou.")
    else:
        print("\n✅✅ VEREDITO: PASSOU! ✅✅")
        print("A máscara contém múltiplas classes (ex: Fundo, Orelhas, Olhos, etc).")

    # 6. SALVAR PARA PROVA VISUAL
    # Podes baixar estes ficheiros e abrir no ITK-SNAP para ter 100% certeza
    print("\n--- A SALVAR FICHEIROS DE PROVA ---")
    img_nii = nib.Nifti1Image(X[0, :, :, :, 0], np.eye(4))
    msk_nii = nib.Nifti1Image(y_argmax.astype(np.float32), np.eye(4))
    
    nib.save(img_nii, f'sanity_img_{id}.nii.gz')
    nib.save(msk_nii, f'sanity_mask_{id}.nii.gz')
    print("Guardado: sanity_img.nii.gz e sanity_mask.nii.gz")

if __name__ == "__main__":
    run_sanity_check()