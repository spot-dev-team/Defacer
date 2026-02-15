import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CAMINHOS (AJUSTA ISTO) ---
# Um ficheiro qualquer do ADNI (Treino)
FILE_ADNI = "D:\\Tese_BDs\\Defacer\\ADNI_Structured\\002_S_0413__2006-11-15_14_23_260__I30119\\raw.nii.gz"
# Um ficheiro qualquer do IXI (Teste)
FILE_IXI = "D:\\Tese_BDs\\Defacer\\IXI_Structured\\IXI002-Guys-0828-T1\\raw.nii.gz"

def check_alignment():
    print("--- A CARREGAR IMAGENS ---")
    img_adni = nib.load(FILE_ADNI)
    img_ixi = nib.load(FILE_IXI)
    
    # Dados brutos (como a U-Net os vê)
    data_adni = img_adni.get_fdata()
    data_ixi = img_ixi.get_fdata()
    
    # Metadados de Orientação
    ornt_adni = nib.aff2axcodes(img_adni.affine)
    ornt_ixi = nib.aff2axcodes(img_ixi.affine)
    
    print(f"Orientação ADNI: {ornt_adni}")
    print(f"Orientação IXI:  {ornt_ixi}")
    
    if ornt_adni != ornt_ixi:
        print("⚠️ ALERTA: As orientações são diferentes! É preciso reorientar o IXI.")

    # --- VISUALIZAÇÃO ---
    # Vamos ver a fatia central de cada eixo para garantir que a cabeça está "em pé"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # ADNI
    c_adni = [s // 2 for s in data_adni.shape]
    axes[0,0].imshow(np.rot90(data_adni[c_adni[0], :, :]), cmap='gray'); axes[0,0].set_title("ADNI (Sagital?)")
    axes[0,1].imshow(np.rot90(data_adni[:, c_adni[1], :]), cmap='gray'); axes[0,1].set_title("ADNI (Coronal?)")
    axes[0,2].imshow(np.rot90(data_adni[:, :, c_adni[2]]), cmap='gray'); axes[0,2].set_title("ADNI (Axial?)")
    
    # IXI
    c_ixi = [s // 2 for s in data_ixi.shape]
    axes[1,0].imshow(np.rot90(data_ixi[c_ixi[0], :, :]), cmap='gray'); axes[1,0].set_title("IXI (Sagital?)")
    axes[1,1].imshow(np.rot90(data_ixi[:, c_ixi[1], :]), cmap='gray'); axes[1,1].set_title("IXI (Coronal?)")
    axes[1,2].imshow(np.rot90(data_ixi[:, :, c_ixi[2]]), cmap='gray'); axes[1,2].set_title("IXI (Axial?)")
    
    plt.tight_layout()
    plt.savefig("orientacao_check.png")
    print("Imagem salva: orientacao_check.png")

if __name__ == "__main__":
    check_alignment()