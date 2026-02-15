import os
import shutil
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
SOURCE_DIR = r"D:\Tese_BDs\Defacer\segmentados"
TARGET_DIR = r"C:\Tese\IXI_Cluster"  # Pasta temporária para envio

def prepare_for_upload():
    if not os.path.exists(SOURCE_DIR):
        print("Pasta de origem não encontrada!")
        return

    # Limpar destino se já existir
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    patients = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    print(f"A analisar {len(patients)} pacientes...")

    count = 0
    
    for subj in tqdm(patients):
        src_subj_path = os.path.join(SOURCE_DIR, subj)
        
        # Ficheiros que queremos copiar
        # 1. O Volume Original
        raw_file = "raw.nii.gz"
        
        # 2. As Máscaras (Copia a de 3 e a de 4 classes para teres flexibilidade no cluster)
        mask_folder = "training_masks"
        masks_to_copy = ["mask_4_classes.nii.gz"]

        # Verificar se existem na origem
        src_raw = os.path.join(src_subj_path, raw_file)
        src_mask_dir = os.path.join(src_subj_path, mask_folder)
        
        # Só copiamos se tiver pelo menos o raw e a máscara de 3 classes
        if os.path.exists(src_raw) and os.path.exists(os.path.join(src_mask_dir, "mask_4_classes.nii.gz")):
            
            # Criar estrutura no destino
            dst_subj_path = os.path.join(TARGET_DIR, subj)
            dst_mask_dir = os.path.join(dst_subj_path, mask_folder)
            
            os.makedirs(dst_mask_dir, exist_ok=True)
            
            # Copiar RAW
            shutil.copy2(src_raw, os.path.join(dst_subj_path, raw_file))
            
            # Copiar MÁSCARAS
            for m in masks_to_copy:
                m_src = os.path.join(src_mask_dir, m)
                if os.path.exists(m_src):
                    shutil.copy2(m_src, os.path.join(dst_mask_dir, m))
            
            count += 1

    print("-" * 30)
    print(f"Preparação concluída!")
    print(f"Pacientes copiados: {count}")
    print(f"Pasta pronta para zipar: {TARGET_DIR}")

if __name__ == "__main__":
    prepare_for_upload()