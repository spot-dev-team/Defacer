import matplotlib.pyplot as plt
import numpy as np
import os
from train_adni_keras import AdniDataGenerator

BASE_DIR = r"C:\Tese\ADNI_Lite_Cluster"


def visualize_augmentation():
    print("--- A INICIAR DEBUG DE AUGMENTATION ---")
    
    # 1. Escolher um paciente real da tua pasta
    all_patients = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    if not all_patients:
        print("Erro: Não encontrei pacientes na pasta definida.")
        return
        
    patient_id = all_patients[0] # Pega no primeiro que encontrar
    print(f"A testar com o paciente: {patient_id}")
    
    # 2. Criar o Gerador em modo "Augment=True"
    # Vamos criar um gerador que só tem este paciente
    gen = AdniDataGenerator([patient_id], BASE_DIR, batch_size=1, augment=True)
    
    # 3. Gerar 5 variações do mesmo paciente
    # Como o augment=True, cada vez que pedimos o item [0], ele gera uma versão nova
    variations = []
    for i in range(5):
        X, y = gen[0] # Pede o batch 0
        variations.append((X[0], y[0])) # Guarda o volume (remove a dimensão do batch)
        print(f"Gerada variação {i+1}/5...")

    # 4. Visualizar (Fatia Central)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    slice_idx = 64 # Meio do volume (128/2)
    
    for i in range(5):
        vol_img = variations[i][0]
        vol_mask = variations[i][1]
        
        # O volume é (128, 128, 128, 1). Queremos a fatia Z=64
        img_slice = vol_img[:, :, slice_idx, 0]
        
        # A máscara é One-Hot (128, 128, 128, 5). 
        # Vamos converter para argmax para ver as cores (0, 1, 2, 3, 4)
        mask_slice = np.argmax(vol_mask, axis=-1)[:, :, slice_idx]

        # Linha 1: Imagem MRI
        axes[0, i].imshow(img_slice, cmap='gray')
        axes[0, i].set_title(f"Augmentation #{i+1}")
        axes[0, i].axis('off')
        
        # Linha 2: Máscara Correspondente
        axes[1, i].imshow(mask_slice, cmap='jet', interpolation='nearest')
        axes[1, i].set_title(f"Máscara #{i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('debug_aug.png')
    print("✅ Sucesso! Abre o ficheiro 'debug_aug.png' para ver as variações.")

if __name__ == "__main__":
    visualize_augmentation()
